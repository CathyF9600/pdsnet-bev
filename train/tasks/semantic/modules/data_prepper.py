#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import imp
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from common.avgmeter import *
from common.logger import Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax
import tasks.semantic.modules.adf as adf
import numpy as np

def keep_variance_fn(x):
    return x + 1e-3

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]

    return y_true


class SoftmaxHeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)

    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = torch.nn.functional.one_hot(targets, num_classes=20).permute(0,3,1,2).float()

        precision = 1 / (var + eps)
        return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNext" + suffix)


class Data_prepper():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None,uncertainty=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.uncertainty = uncertainty

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_update": 0,
                     "train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=1,
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=False)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if not self.uncertainty:
                self.model = SalsaNext(self.parser.get_n_classes())
            else:
                self.model = SalsaNextUncertainty(self.parser.get_n_classes())

        self.tb_logger = Logger(self.log + "/tb")

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        self.SoftmaxHeteroscedasticLoss = SoftmaxHeteroscedasticLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()
            self.SoftmaxHeteroscedasticLoss = nn.DataParallel(self.SoftmaxHeteroscedasticLoss).cuda()
        self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def prep(self):
        self.save_train_set(train_loader=self.parser.get_train_set())
        self.save_val_set(val_loader=self.parser.get_valid_set())
        return

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    def save_train_set(self, train_loader):
        cul_in_vol = None
        cul_proj_labels = None
        if not os.path.exists(os.path.join(self.datadir, 'range_proj_test', 'train')):
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'train'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'train', 'in_vol'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_mask'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_depth_vis'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_labels'))
        for i, (in_vol, _, proj_mask, _, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # if i != 2807: #  or i != 2636:
            #     continue
            # compute mean and avg frequencies
            try:
                if cul_in_vol == None:
                    cul_in_vol = torch.cat((in_vol[:,0][proj_mask.bool()].unsqueeze(0), in_vol[:,1][proj_mask.bool()].unsqueeze(0), in_vol[:,2][proj_mask.bool()].unsqueeze(0), in_vol[:,3][proj_mask.bool()].unsqueeze(0), in_vol[:,4][proj_mask.bool()].unsqueeze(0)), dim=0)
                else:
                    temp = torch.cat((in_vol[:,0][proj_mask.bool()].unsqueeze(0), in_vol[:,1][proj_mask.bool()].unsqueeze(0), in_vol[:,2][proj_mask.bool()].unsqueeze(0), in_vol[:,3][proj_mask.bool()].unsqueeze(0), in_vol[:,4][proj_mask.bool()].unsqueeze(0)), dim=0)
                    cul_in_vol = torch.cat((cul_in_vol, temp), axis=1)
                if cul_proj_labels == None:
                    cul_proj_labels = proj_labels[proj_mask.bool()]
                else:
                    cul_proj_labels = torch.cat((cul_proj_labels, proj_labels[proj_mask.bool()]), axis=0)
                # d1 = {'in_vol': in_vol, 'proj_mask': proj_mask, 'proj_labels': proj_labels}
                depth = (cv2.normalize(in_vol[0][0].cpu().numpy(), None, alpha=0, beta=1,
                                       norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
                out_img = cv2.applyColorMap(
                    depth, self.get_mpl_colormap('viridis')) * proj_mask[0].cpu().numpy()[..., None]
                out_img = out_img.astype(np.uint8)
                cv2.imwrite(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_depth_vis', path_name[0].split('.')[0]+".png"), out_img)
                np.save(os.path.join(self.datadir, 'range_proj_test', 'train', 'in_vol', path_name[0].split('.')[0]+".npy"), in_vol)
                np.save(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_mask', path_name[0].split('.')[0]+".npy"), proj_mask)
                np.save(os.path.join(self.datadir, 'range_proj_test', 'train', 'proj_labels', path_name[0].split('.')[0]+".npy"), proj_labels)
            except:
                print('something wrong here')

        for i in range(23):
            print(torch.where(cul_proj_labels==i)[0].shape[0] / cul_proj_labels.shape[0])
        print(cul_in_vol.mean(dim=1))
        print(cul_in_vol.std(dim=1))
        return

    def save_val_set(self, val_loader):
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        if not os.path.exists(os.path.join(self.datadir, 'range_proj_test', 'valid')):
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'valid'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'valid', 'in_vol'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'valid', 'proj_mask'))
            os.makedirs(os.path.join(self.datadir, 'range_proj_test', 'valid', 'proj_labels'))
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(val_loader):
            np.save(os.path.join(self.datadir, 'range_proj_test', 'valid', 'in_vol', path_name[0].split('.')[0]+".npy"), in_vol)
            np.save(os.path.join(self.datadir, 'range_proj_test', 'valid', 'proj_mask', path_name[0].split('.')[0]+".npy"), proj_mask)
            np.save(os.path.join(self.datadir, 'range_proj_test', 'valid', 'proj_labels', path_name[0].split('.')[0]+".npy"), proj_labels)
        return
