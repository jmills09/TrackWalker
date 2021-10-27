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



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_steps(seq,long=False):
    full_np = np.stack(seq,axis=0)
    if not long:
        return torch.tensor(full_np, dtype=torch.float)
    else:
        return torch.tensor(full_np, dtype=torch.long)

PARAMS = {}

PARAMS['USE_CONV_IM'] = True
PARAMS['LARMATCH_CKPT'] = '/home/jmills/workdir/TrackWalker/larmatch_ckpt/checkpoint.1974000th.tar'
# PARAMS['LARMATCH_CKPT'] = '/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/grid_deploy_scripts/larmatch_kps_weights/checkpoint.1974000th.tar'
PARAMS['MASK_WC'] = False

PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =1024
PARAMS['PADDING'] =10
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) # N_Features
if PARAMS['USE_CONV_IM']:
    PARAMS['EMBEDDING_DIM'] = PARAMS['EMBEDDING_DIM']*16 # 16 Features per pixel in larmatch
PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)+1 # Bonus Class is for the end of track class
PARAMS['TRACKEND_CLASS'] = (PARAMS['PADDING']*2+1)**2
PARAMS['CENTERPOINT_ISEND'] = True
if PARAMS['CENTERPOINT_ISEND']:
     PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) #No bonus end of track class
     PARAMS['TRACKEND_CLASS'] = (PARAMS['NUM_CLASSES']-1)/2
# PARAMS['INFILE'] ="/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
PARAMS['INFILE'] ="/cluster/tufts/wongjiradlab/jmills09/TrackerNet_InputData/midlevel_files/small_00002_merged_dlreco_497be540-00f9-49a8-9f80-7846143c4fce.root"
PARAMS['TRACK_IDX'] =0
PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
# TENSORDIR'] ="runs/Pad20_Hidden1024_500Entries"
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
PARAMS['NDIMENSIONS'] = 2 #Not configured to have 3 yet.
PARAMS['LEARNING_RATE'] =0.0001 # 0.01 is good for the classifier mode,
PARAMS['DO_TENSORLOG'] = True
PARAMS['TENSORDIR']  = None # Default runs/DATE_TIME
PARAMS['SAVE_MODEL'] = False #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_EPOCHS'] =10000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 5
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 100
# PARAMS['DEVICE'] = 'cuda:0'
PARAMS['DEVICE'] = 'cpu'
PARAMS['LOAD_SIZE']  = 50
PARAMS['TRAIN_EPOCH_SIZE'] = 500
PARAMS['VAL_EPOCH_SIZE'] = int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--infile', dest='infile', required=False,
        help='infile')

    parser.add_argument(
        '--infileidx', dest='infileidx', required=False,
        help='infilinfileidxeidx')

    parser.add_argument(
        '--folderidx', dest='folderidx', required=False,
        help='folderidx')

    parser.add_argument(
        '--outdir', dest='outdir', required=False,
        help='outdir')

    return parser.parse_args()


def main():
    # test with:
    # python FormatTrainingData_Complex.py --infileidx 0 --outdir test/ --folderidx 0 --infile /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/71/93/merged_dlreco_b1379c2b-cd47-4140-a448-3a27e8f28fb8.root
    print("Let's Get Started.")

    args = parse_args()
    print('Called with args:')

    args.infile       = "inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
    args.outdir       = "sizetest/"
    args.folderidx    = str(0)
    args.infileidx    = "2d"#str()
    print(args)
    PARAMS['INFILE'] =args.infile

    nbins = PARAMS['PADDING']*2+1

    # entry_per_file = 9999
    entry_per_file = 9999
    Loader = FancyLoader(PARAMS)
    # iter = int(tot_entries/entry_per_file)+1
    iter = 1
    for i in range(0,iter):
        # start_file = i*entry_per_file
        # end_file   = i*entry_per_file+entry_per_file
        # if end_file > tot_entries:
        #     end_file = tot_entries-1
        start_entry = 0
        end_entry = 2
        max_tracks = 2
        larmatch_feature_image_v, wire_image_v, step_idx_image_v, run_v, \
            subrun_v, eventid_v, entry_v, mctrack_idx_v, mctrack_length_v, \
            mctrack_pdg_v, mctrack_energy_v, charge_in_wire_v, charge_in_truth_v \
            = Loader.load_fancy(start_entry, end_entry)
        # larmatch_feature_image_v, wire_image_v, step_idx_image_v, run_v, subrun_v, eventid_v, entry_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, charge_in_wire_v, charge_in_truth_v = Loader.load_fancy()
        # from MiscFunctions import save_im
        # save_im(wire_image_v[0],     savename="test/wireimg",canv_x=4000,canv_y=1000)
        # save_im(step_idx_image_v[0], savename="test/stepimg",canv_x=4000,canv_y=1000)

        outfile_str = "Wire_" if PARAMS['USE_CONV_IM'] == False else "LArMatch_"
        outfile_str = outfile_str + "ComplexTrackIdx_"
        outfile_str = args.outdir+args.folderidx+"/Reformat_" + outfile_str
        if not os.path.exists(args.outdir+args.folderidx):
            os.makedirs(args.outdir+args.folderidx)
        outfile = ROOT.TFile(outfile_str+str(args.infileidx).zfill(3)+".root","recreate")
        tree    = ROOT.TTree("TrackWalkerInput_ComplexTrackIdx","TrackWalker Reformmated Input Tree Special Track Index Image")

        wire_image_np = larcv.NumpyArrayFloat()
        tree.Branch("wire_image_np",wire_image_np)

        larmatchfeat_image_np = larcv.NumpyArrayFloat()
        tree.Branch("larmatchfeat_image_np",larmatchfeat_image_np)

        # stepidx_image_np = larcv.NumpyArrayFloat()
        # tree.Branch("stepidx_image_np",stepidx_image_np)
        #
        # entry = array('f',[0]) # single float
        # tree.Branch("original_entry",entry,"original_entry/F")
        # mctrack_idx = array('f',[0]) # single float
        # tree.Branch("mctrack_idx",mctrack_idx,"mctrack_idx/F")
        # mctrack_length = array('f',[0]) # single float
        # tree.Branch("mctrack_length",mctrack_length,"mctrack_length/F")
        # mctrack_pdg = array('f',[0]) # single float
        # tree.Branch("mctrack_pdg",mctrack_pdg,"mctrack_pdg/F")
        # mctrack_energy = array('f',[0]) # single float
        # tree.Branch("mctrack_energy",mctrack_energy,"mctrack_energy/F")
        # run = array('f',[0]) # single float
        # tree.Branch("run",run,"run/F")
        # subrun = array('f',[0]) # single float
        # tree.Branch("subrun",subrun,"subrun/F")
        # event_id = array('f',[0]) # single float
        # tree.Branch("event_id",event_id,"event_id/F")
        #
        # charge_in_wire = array('f',[0]) # single float
        # tree.Branch("charge_in_wire",charge_in_wire,"charge_in_wire/F")
        # charge_in_truth = array('f',[0]) # single float
        # tree.Branch("charge_in_truth",charge_in_truth,"charge_in_truth/F")

        print("Number of Tracks:", len(wire_image_v))
        for idxx in range(len(wire_image_v)):
            if idxx == 0:
                continue
            print("Filling Tree", idxx)
            print("")
            print("        ",wire_image_v[idxx].shape, "Wire Image Shape")
            nFeats = wire_image_v[idxx].shape[0] * wire_image_v[idxx].shape[1]
            print("        ",nFeats, "Wire Image Feats")

            print("        ",larmatch_feature_image_v[idxx].shape, "LArFeat Image Shape")
            nFeats = larmatch_feature_image_v[idxx].shape[0] * larmatch_feature_image_v[idxx].shape[1] * larmatch_feature_image_v[idxx].shape[2]
            print("        ",nFeats, "LArFeat Image Feats")


            nonzeroIdx = np.nonzero(larmatch_feature_image_v[idxx])
            nFeats = nonzeroIdx[0].shape[0]
            sFeats = np.zeros((nFeats,4)) #x,y,nfeat,val
            for iii in range(nFeats):
                sFeats[iii,0] = nonzeroIdx[0][iii]
                sFeats[iii,1] = nonzeroIdx[1][iii]
                sFeats[iii,2] = nonzeroIdx[2][iii]
                sFeats[iii,3] = larmatch_feature_image_v[idxx][nonzeroIdx[0][iii],nonzeroIdx[1][iii],nonzeroIdx[2][iii]]



            print("Storing sparselar")
            print(wire_image_v[idxx].shape)
            print(sFeats.shape)
            wire_image_np.store(wire_image_v[idxx].astype(np.float32))
            larmatchfeat_image_np.store(sFeats.astype(np.float32))


            # print("Storing")
            # print(wire_image_v[idxx].shape)
            # print(larmatch_feature_image_v[idxx].shape)
            # wire_image_np.store(wire_image_v[idxx].astype(np.float32))
            # larmatchfeat_image_np.store(larmatch_feature_image_v[idxx].astype(np.float32))
            # stepidx_image_np.store(step_idx_image_v[idxx].astype(np.float32))
            #
            # entry[0]          = entry_v[idxx]
            # mctrack_idx[0]    = mctrack_idx_v[idxx]
            # mctrack_length[0] = mctrack_length_v[idxx]
            # mctrack_pdg[0]    = mctrack_pdg_v[idxx]
            # mctrack_energy[0]    = mctrack_energy_v[idxx]
            #
            # run[0]            = run_v[idxx]
            # subrun[0]         = subrun_v[idxx]
            # event_id[0]       = eventid_v[idxx]
            #
            # charge_in_wire[0]  = charge_in_wire_v[idxx]
            # charge_in_truth[0] = charge_in_truth_v[idxx]
            tree.Fill()
            idxx += 1
        tree.Write()
        outfile.Close()
    return 0

if __name__ == '__main__':
    main()
