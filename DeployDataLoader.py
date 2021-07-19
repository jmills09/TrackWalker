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
