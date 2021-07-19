import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint
from DataLoader import get_net_inputs_mc, DataLoader_MC
from ModelFunctions import LSTMTagger, run_validation_pass
import random
import ROOT
from array import array
from larcv import larcv


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
PARAMS['INFILE'] ="/home/jmills/workdir/TrackWalker/inputfiles/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtmerge_TrackWalker_traindata_198files.root"
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
PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50
PARAMS['TRAIN_EPOCH_SIZE'] = 500
PARAMS['VAL_EPOCH_SIZE'] = int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2


def main():
    print("Let's Get Started.")
    nbins = PARAMS['PADDING']*2+1

    entry_per_file = 100
    DataLoader = DataLoader_MC(PARAMS,all_train=True)
    tot_entries = DataLoader.nentries_train
    iter = int(tot_entries/entry_per_file)+1
    for i in range(0,iter):
        start_file = i*entry_per_file
        end_file   = i*entry_per_file+entry_per_file
        if end_file > tot_entries:
            end_file = tot_entries-1
        test_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, runs_v, subruns_v, event_ids  = DataLoader.load_dlreco_inputs_onestop(start_file,end_file, is_val=False)
        outfile_str = "Wire_" if PARAMS['USE_CONV_IM'] == False else "LArMatch_"
        outfile_str = outfile_str + "Pad_"+str(PARAMS['PADDING']).zfill(3)+"_"
        outfile = ROOT.TFile("inputfiles/ReformattedInput/Reformat_"+outfile_str+str(i).zfill(3)+".root","recreate")
        tree    = ROOT.TTree("TrackWalkerInput_Pad_"+str(PARAMS['PADDING']).zfill(3),"TrackWalker Reformmated Input Tree Pad "+str(PARAMS['PADDING']).zfill(3))
        stacked_step_images = larcv.NumpyArrayFloat()
        tree.Branch("stacked_step_images",stacked_step_images)

        stacked_targ_idx = larcv.NumpyArrayFloat()
        tree.Branch("stacked_targ_idx",stacked_targ_idx)

        stacked_targ_area = larcv.NumpyArrayFloat()
        tree.Branch("stacked_targ_area",stacked_targ_area)

        entry = array('f',[0]) # single float
        tree.Branch("original_entry",entry,"original_entry/F")
        mctrack_idx = array('f',[0]) # single float
        tree.Branch("mctrack_idx",mctrack_idx,"mctrack_idx/F")
        mctrack_length = array('f',[0]) # single float
        tree.Branch("mctrack_length",mctrack_length,"mctrack_length/F")
        mctrack_pdg = array('f',[0]) # single float
        tree.Branch("mctrack_pdg",mctrack_pdg,"mctrack_pdg/F")
        run = array('f',[0]) # single float
        tree.Branch("run",run,"run/F")
        subrun = array('f',[0]) # single float
        tree.Branch("subrun",subrun,"subrun/F")
        event_id = array('f',[0]) # single float
        tree.Branch("event_id",event_id,"event_id/F")

        idxx = 0
        for step_images, targ_next_step_idx, targ_area_next_step in test_data:
            stack_step_4d = np.stack(step_images)
            stack_step_3d = np.reshape(stack_step_4d,(21,21,-1))
            reshape_check = np.reshape(stack_step_3d,(-1,21,21,16))
            if False == np.array_equal(reshape_check, stack_step_4d):
                continue
            stacked_step_images.store(stack_step_3d.astype(np.float32))
            stacked_targ_idx.store(np.stack(targ_next_step_idx).astype(np.float32))
            stacked_targ_area.store(np.stack(targ_area_next_step).astype(np.float32))
            entry[0]          = entries_v[idxx]
            mctrack_idx[0]    = mctrack_idx_v[idxx]
            mctrack_length[0] = mctrack_length_v[idxx]
            mctrack_pdg[0]    = mctrack_pdg_v[idxx]
            run[0]            = runs_v[idxx]
            subrun[0]         = subruns_v[idxx]
            event_id[0]       = event_ids[idxx]

            tree.Fill()
            idxx += 1
        tree.Write()
        outfile.Close()
    return 0

if __name__ == '__main__':
    main()
