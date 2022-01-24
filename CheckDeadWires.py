import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint
from FancyLoader import FancyLoader
from ModelFunctions import LSTMTagger, run_validation_pass
import random
import ROOT
from array import array
from larcv import larcv
import os
import argparse





def main():
    # test with:
    # python FormatTrainingData_Complex.py --infileidx 0 --outdir test/ --folderidx 0 --infile /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/71/93/merged_dlreco_b1379c2b-cd47-4140-a448-3a27e8f28fb8.root
    print("Let's Get Started.")
    infile ="/home/jmills/workdir/TrackWalker/inputfiles/nue_intrinsics_run3b/nue_intrinsics_run3b_merged_dlreco.root"

    iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.set_verbosity(5)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()
    iocv.read_entry(0)
    ev_wire = iocv.get_data(larcv.kProductImage2D,"wire")
    img_v = ev_wire.Image2DArray()
    img_y = img_v[2]
    img_np = larcv.as_ndarray(img_y)
    # assert 1==2




    iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.set_verbosity(5)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()
    iocv.read_entry(0)
    ev_chstatus = iocv.get_data(larcv.kProductChStatus,"wire")
    chstatus_map = ev_chstatus.ChStatusMap()
    chstatus_0_v = chstatus_map[0].as_vector()
    chstatus_1_v = chstatus_map[1].as_vector()
    chstatus_2_v = chstatus_map[2].as_vector()
    n_four_0 = 0
    n_four_1 = 0
    n_four_2 = 0
    print(chstatus_0_v.size(), "u")
    print(chstatus_1_v.size(), "v")
    print(chstatus_2_v.size(), "y")

    for i in range(chstatus_0_v.size()):
        if chstatus_0_v.at(i) == 4:
            n_four_0 += 1
    print("U")
    print(n_four_0, chstatus_0_v.size() , n_four_0*1.0/chstatus_0_v.size())
    for i in range(chstatus_1_v.size()):
        if chstatus_1_v.at(i) == 4:
            n_four_1 += 1
    print("V")
    print(n_four_1, chstatus_1_v.size() , n_four_1*1.0/chstatus_1_v.size())
    for i in range(chstatus_2_v.size()):
        if chstatus_2_v.at(i) == 4:
            n_four_2 += 1
    print("Y")
    print(n_four_2, chstatus_2_v.size() , n_four_2*1.0/chstatus_2_v.size())

    print("All")
    print(n_four_0+n_four_1+n_four_2, chstatus_0_v.size()+chstatus_1_v.size()+chstatus_2_v.size() , ((n_four_0+n_four_1+n_four_2)*1.0)/(chstatus_0_v.size()+chstatus_1_v.size()+chstatus_2_v.size()))
    return 0

if __name__ == '__main__':
    main()
