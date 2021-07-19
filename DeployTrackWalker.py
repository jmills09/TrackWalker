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
from MiscFunctions import blockPrint, enablePrint, cropped_np, make_steps_images
from MiscFunctions import save_im
from DataLoader import get_net_inputs_mc, DataLoader_MC
from ReformattedDataLoader import ReformattedDataLoader_MC
from ModelFunctions import LSTMTagger, run_validation_pass
import random
import ROOT
import socket
import os
from datetime import datetime
import time

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
PARAMS['TWOWRITERS'] = True

PARAMS['SAVE_MODEL'] = False #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_EPOCHS'] =10000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 1
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50 #Number of Entries to Load training tracks from
PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
PARAMS['VAL_SAMPLE_SIZE'] = 100

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_1_Fin.pt"
PARAMS['NUM_DEPLOY'] = 1

def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")

    output_dim = None
    output_dim = PARAMS['NUM_CLASSES']


    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))

    model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT']))

    optimizer = optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']

    DataLoader = DataLoader_MC(PARAMS,all_train=False,deploy=True)
    start_file = 4
    end_file   = int(start_file+5) #estimate of how many events to get the number of tracks we want to run on
    with torch.no_grad():
        model.eval()
        deploy_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, runs_v, subruns_v, event_ids, larmatch_im_v, wire_im_v, x_starts_v, y_starts_v = DataLoader.load_dlreco_inputs_onestop_deploy(start_file,end_file,MAX_TRACKS_PULL=PARAMS['NUM_DEPLOY'],is_val=True)
        print("Track Info:")
        print("Pdg:", mctrack_pdg_v[0])
        print("Length:", mctrack_length_v[0])
        print("Idx:", mctrack_idx_v[0])
        print("X:", x_starts_v[0])
        print("Y:", y_starts_v[0])

        deploy_idx = -1
        vis_wirecrops_v = []
        vispred_v = []
        start_time = time.time()
        for step_images, targ_next_step_idx, targ_area_next_step in deploy_data:
            deploy_idx += 1
            step_images = [step_images[0]] # Strip off any steps but first one
            vis_wirecrops_v.append(cropped_np(wire_im_v[deploy_idx], x_starts_v[deploy_idx], y_starts_v[deploy_idx], PARAMS['PADDING']))
            # model.zero_grad()
            step_images_in = prepare_sequence_steps(step_images).to(torch.device(PARAMS['DEVICE']))

            # Step 3. Run our forward pass.
            next_steps_pred_scores, endpoint_scores, hidden_n, cell_n = model(step_images_in)
            npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
            np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)

            is_endpoint_score = (np.argmax(endpoint_scores.cpu().detach().numpy(),axis=1)[0] == True)

            optimizer.step()
            np_idx_v = make_prediction_vector(PARAMS, np_pred)
            for ixx in range(np_idx_v.shape[0]):
                pred_x, pred_y = unflatten_pos(np_idx_v[ixx], PARAMS['PADDING']*2+1)
            # Get Next Step Stuff
            crop_center_x = x_starts_v[deploy_idx] - PARAMS['PADDING'] + pred_x
            crop_center_y = y_starts_v[deploy_idx] - PARAMS['PADDING'] + pred_y
            this_step = [      cropped_np(larmatch_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING'])]
            vispred_v.append(np_idx_v[0])
            n_steps = 1
            while is_endpoint_score == False:
                print("Steps:", n_steps)
                n_steps += 1
                vis_wirecrops_v.append(cropped_np(wire_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING']))

                step_images_in = prepare_sequence_steps(this_step).to(torch.device(PARAMS['DEVICE']))
                # Step 3. Run our forward pass.
                next_steps_pred_scores, endpoint_scores, hidden_n, cell_n = model(step_images_in,(hidden_n, cell_n))
                npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)

                is_endpoint_score = (np.argmax(endpoint_scores.cpu().detach().numpy(),axis=1)[0] == True)

                optimizer.step()
                np_idx_v = make_prediction_vector(PARAMS, np_pred)
                for ixx in range(np_idx_v.shape[0]):
                    pred_x, pred_y = unflatten_pos(np_idx_v[ixx], PARAMS['PADDING']*2+1)
                # Get Next Step Stuff
                crop_center_x = crop_center_x - PARAMS['PADDING'] + pred_x
                crop_center_y = crop_center_y - PARAMS['PADDING'] + pred_y
                this_step = [cropped_np(larmatch_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING'])]
                vispred_v.append(np_idx_v[0])
                if is_endpoint_score == True:
                    print("Endpoint Net Says Stop Here!")
                if n_steps == 25:
                    is_endpoint_score = True
                    print("Catch on Making too Many Steps says end here!")
            end_time = time.time()
            print(end_time - start_time, "Seconds Passed to deploy")
            make_steps_images(np.stack(vis_wirecrops_v,axis=0),'images/deploytest/Dep_Im',PARAMS['PADDING']*2+1,pred=vispred_v,targ=None)
            save_im(wire_im_v[deploy_idx],'images/deploytest/Dep_ImFull',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)


    print()
    print("End of Deploy")
    print()



    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
