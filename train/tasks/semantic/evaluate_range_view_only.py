#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import shutil
from shutil import copyfile
import __init__ as booger
import yaml
from tasks.semantic.modules.eval_rv import *
from pip._vendor.distlib.compat import raw_input

from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextInterpolate import *
from tasks.semantic.modules.SalsaNextCompletion import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoder import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderStageTwo import *
from tasks.semantic.modules.SalsaNextCompletionStageTwo import *
from tasks.semantic.modules.pncnn import *
from tasks.semantic.modules.SalsaNextCompletionStageTwoFilteredWeights import *
from tasks.semantic.modules.SalsaNextCompletionAttention import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderMaskPredictionHead import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss import *
from tasks.semantic.modules.SalsaNextCompletionSingleDecoderMaskPredictionHeadStageTwo import *

#from tasks.semantic.modules.save_dataset_projected import *
import math
from decimal import Decimal

def remove_exponent(d):
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    if prefixes:
        millnames = ['']
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    result = '{:.{precision}f}'.format(n / 10**(3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return '{0}{dx}'.format(result, dx=millnames[millidx])


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='config/labels/semantic-kitti.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default="~/output",
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default="",
        help='If you want to give an aditional discriptive name'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default="",
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--completion', '-c',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--completion_single_decoder', '-cs',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--completion_single_decoder_stage_two', '-cs2',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--completion_stage_two', '-c2',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--completion_stage_two_filtered', '-c2f',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--pncnn', '-pncnn',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--pncnn_stage_two', '-pncnn2',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--ipbasic', '-ipbasic',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--bilateral_filtering', '-bf',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--interpolate', '-int',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--attn', '-attn',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Completion Version'
    )

    parser.add_argument(
        '--full_res', '-fr',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--edge_loss', '-el',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--completion_single_decoder_mask_prediction', '-csmp',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--completion_single_decoder_mask_prediction_stage_two', '-csmp2',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--perception_loss', '-pl',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--completion_single_decoder_stage_two_perception_loss', '-cs2pl',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Full Resolution Version'
    )

    parser.add_argument(
        '--oracle', '-ora',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Oracle Version (trained on nuscenes and inferred on nuscenes)'
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = FLAGS.log + '/logs/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + FLAGS.name
    if FLAGS.uncertainty:
        params = SalsaNextUncertainty(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion:
        params = SalsaNextCompletion(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.attn:
        params = SalsaNextCompletionAttention(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_single_decoder:
        params = SalsaNextCompletionSingleDecoder(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_single_decoder_mask_prediction:
        params = SalsaNextCompletionSingleDecoderMaskPredictionHead(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_single_decoder_stage_two_perception_loss:
        params = SalsaNextCompletionSingleDecoderStageTwoPerceptionLoss(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_single_decoder_stage_two:
        params = SalsaNextCompletionSingleDecoderStageTwo(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_stage_two:
        params = SalsaNextCompletionStageTwo(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_stage_two_filtered:
        params = SalsaNextCompletionStageTwoFilteredWeights(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.pncnn:
        params = PNCNN(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.interpolate:
        params = SalsaNextInterpolate(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    elif FLAGS.completion_single_decoder_mask_prediction_stage_two:
        params = SalsaNextCompletionSingleDecoderMaskPredictionHeadStageTwo(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    else:
        params = SalsaNext(20)
        pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("arch_cfg", FLAGS.arch_cfg)
    print("data_cfg", FLAGS.data_cfg)
    print("uncertainty", FLAGS.uncertainty)
    print("Total of Trainable Parameters: {}".format(millify(pytorch_total_params,2)))
    print("log", FLAGS.log)
    print("pretrained", FLAGS.pretrained)
    print("----------\n")
    # print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if FLAGS.pretrained == "":
            FLAGS.pretrained = None
            if os.path.isdir(FLAGS.log):
                if os.listdir(FLAGS.log):
                    answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
        else:
            FLAGS.log = FLAGS.pretrained
            print("Not creating new log file. Using pretrained directory")
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print("model folder exists! Using model from %s" % (FLAGS.pretrained))
        else:
            print("model folder doesnt exist! Start with random weights...")
    else:
        print("No pretrained directory found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
        copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    # create trainer and start the training
    evaluator = Eval_rv(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.pretrained,FLAGS.uncertainty,FLAGS.completion,FLAGS.completion_single_decoder, FLAGS.completion_single_decoder_stage_two, FLAGS.completion_stage_two,
                      FLAGS.pncnn,
                      FLAGS.pncnn_stage_two,
                      FLAGS.ipbasic,
                      FLAGS.bilateral_filtering,
                      FLAGS.interpolate,
                      FLAGS.completion_stage_two_filtered,
                      FLAGS.attn,
                      FLAGS.full_res,
                      FLAGS.edge_loss,
                      FLAGS.completion_single_decoder_mask_prediction,
                      FLAGS.completion_single_decoder_stage_two_perception_loss,
                      FLAGS.oracle,
                      FLAGS.completion_single_decoder_mask_prediction_stage_two)
    evaluator.eval()