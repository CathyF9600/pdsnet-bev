import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, GtLaserScan
import torchvision

import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask, proj_labels

class Carla(Dataset):
  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True,
               transform=False,
               simple=False,
               multi_sensor=False,
               test=False):            # send ground truth?
    self.test = test
    # save deats
    self.sequence = sequences
    self.simple = simple
    if self.simple:
        self.root = os.path.join(root, "range_proj_high_res", self.sequence[0], "in_vol")
        self.label_root = os.path.join(root, "range_proj_high_res", self.sequence[0], "proj_labels")
    else:
        if not self.test:
            self.root = os.path.join(root, "sem_test", self.sequence[0], "kitti_velodyne") # kitti_velodyne
        else:
            print('using oracle')
            self.root = os.path.join(root, "sem_test", self.sequence[0], "nusc_velodyne")
        self.label_root = os.path.join(root, "sem_test", self.sequence[0], "velodyne")
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.transform = transform
    self.multi_sensor = multi_sensor

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # get files
    self.scan_files = list(os.listdir(self.root))

    self.scan_files = [self.root+'/'+ i for i in self.scan_files]

    # sort for correspondance
    self.scan_files.sort()

    if self.multi_sensor:
        self.scan_files_add = self.scan_files.copy()
        for entry in self.scan_files_add:
            entry = entry.replace('kitti', 'nusc')
            self.scan_files.append(entry)
    print("Using {} scans".format(len(self.scan_files)))

    self.label_files = list(os.listdir(self.label_root))
    self.label_files = [self.label_root+'/'+ i for i in self.label_files]
    self.label_files.sort()
    if self.multi_sensor:
        self.label_files += self.label_files.copy()

  def __getitem__(self, index):
    # if index !=  2807:
    #     return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

    # get item in tensor shape
    scan_file = self.scan_files[index]
    label_file = self.label_files[index]

    # open a laserscan
    DA = False
    flip_sign = False
    rot = False
    drop_points = False
    if self.transform:
        if random.random() > 0.5:
    #         if random.random() > 0.5:
    #             DA = True
            if random.random() > 0.5:
                flip_sign = True
    #         if random.random() > 0.5:
    #             rot = True
    #         drop_points = random.uniform(0, 0.5)

    scan = LaserScan(project=True,
                      H=self.sensor_img_H,
                      W=self.sensor_img_W,
                      fov_up=self.sensor_fov_up,
                      fov_down=self.sensor_fov_down,
                      DA=DA,
                      rot=rot,
                      flip_sign=flip_sign,
                      drop_points=drop_points)

    label = GtLaserScan(project=True,
                      H=self.sensor_img_H,
                      W=self.sensor_img_W,
                      fov_up=self.sensor_fov_up,
                      fov_down=self.sensor_fov_down,
                      DA=DA,
                      rot=rot,
                      flip_sign=flip_sign,
                      drop_points=drop_points)
    # open and obtain scan
    scan.open_scan(scan_file)
    label.open_scan(label_file)

    # make a tensor of the uncompressed data (with the max num points)
    proj = scan.bev_map[:-1] # scan.bev_map[-1] stores the number of points
    # (5, 160, 141) 5: 0123 + 4 reflectivity
    label_proj = label.bev_map[:-1]
    proj_mask = proj != 0
    label_proj_mask = label_proj != 0

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1]

    # return
    return proj, label_proj, proj_mask, label_proj_mask, path_seq, path_name

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,# shuffle training set?
               simple=False,
               multi_sensor=False,
               no_transform=False,
               oracle=False):
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train
    self.multi_sensor = multi_sensor

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    if no_transform:
        self.train_dataset = Carla(root=self.root,
                                   sequences=self.train_sequences,
                                   labels=self.labels,
                                   color_map=self.color_map,
                                   learning_map=self.learning_map,
                                   learning_map_inv=self.learning_map_inv,
                                   sensor=self.sensor,
                                   max_points=max_points,
                                   gt=self.gt,
                                   simple=simple,
                                   multi_sensor=self.multi_sensor)  # simple uses saved projected points
    else:
        if not oracle:
            self.train_dataset = Carla(root=self.root,
                                               sequences=self.train_sequences,
                                               labels=self.labels,
                                               color_map=self.color_map,
                                               learning_map=self.learning_map,
                                               learning_map_inv=self.learning_map_inv,
                                               sensor=self.sensor,
                                               max_points=max_points,
                                               transform=True,
                                               gt=self.gt,
                                               simple=simple,
                                               multi_sensor=self.multi_sensor) # simple uses saved projected points
        else:
            self.train_dataset = Carla(root=self.root,
                                               sequences=self.train_sequences,
                                               labels=self.labels,
                                               color_map=self.color_map,
                                               learning_map=self.learning_map,
                                               learning_map_inv=self.learning_map_inv,
                                               sensor=self.sensor,
                                               max_points=max_points,
                                               transform=True,
                                               gt=self.gt,
                                               simple=simple,
                                               multi_sensor=self.multi_sensor,
                                               test=True) # simple uses saved projected points

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = Carla(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt,
                                       simple=simple,
                                       multi_sensor=self.multi_sensor) # simple uses saved projected points

    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    self.test_dataset = Carla(root=self.root,
                                        sequences=self.valid_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=self.gt,
                                        simple=simple,
                                        multi_sensor=self.multi_sensor,
                                        test=True) # simple uses saved projected points

    self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    drop_last=True)
    assert len(self.testloader) > 0
    self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return Carla.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return Carla.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = Carla.map(label, self.learning_map_inv)
    # put label in color
    return Carla.map(label, self.color_map)