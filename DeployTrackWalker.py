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
PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_0_300000.pt"
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
PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
PARAMS['TRACK_IDX'] =0
PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
# TENSORDIR'] ="runs/Pad20_Hidden1024_500Entries"
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
PARAMS['NDIMENSIONS'] = 2 #Not configured to have 3 yet.



PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50 #Number of Entries to Load training tracks from
PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
PARAMS['VAL_SAMPLE_SIZE'] = 100

PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_1_Fin.pt"
PARAMS['NUM_DEPLOY'] = 1
PARAMS['MAX_STEPS'] = 50

def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")

    output_dim = None
    output_dim = PARAMS['NUM_CLASSES']


    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))

    if (PARAMS['DEVICE'] != 'cpu'):
        model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT'], map_location={'cpu':PARAMS['DEVICE'],'cuda:0':PARAMS['DEVICE'],'cuda:1':PARAMS['DEVICE'],'cuda:2':PARAMS['DEVICE'],'cuda:3':PARAMS['DEVICE']}))
    else:
        model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT'], map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
    # model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT']))

    # optimizer = optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']

    DataLoader = DataLoader_MC(PARAMS,all_train=False, all_valid = True, deploy=True)
    start_file = 62
    end_file   = int(start_file+PARAMS['NUM_DEPLOY']*1.1+100) #estimate of how many events to get the number of tracks we want to run on
    with torch.no_grad():
        model.eval()
        deploy_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, runs_v, subruns_v, event_ids, larmatch_im_v, wire_im_v, x_starts_v, y_starts_v = DataLoader.load_dlreco_inputs_onestop_deploy(start_file,end_file,MAX_TRACKS_PULL=PARAMS['NUM_DEPLOY'],is_val=True)
        print("Total Tracks Grabbed:",len(deploy_data))
        deploy_idx = -1
        start_time = time.time()
        # for step_images, targ_next_step_idx, targ_area_next_step in deploy_data:
        for i in range(PARAMS['NUM_DEPLOY']):
            step_images = deploy_data[deploy_idx][0]
            targ_next_step_idx = deploy_data[deploy_idx][1]
            targ_area_next_step = deploy_data[deploy_idx][2]
            vis_wirecrops_v = []
            vispred_v = []
            deploy_idx += 1
            print()
            print("Deploy Idx",deploy_idx, "Max Idx:", len(deploy_data))
            print("Track Info:")
            print("Pdg:", mctrack_pdg_v[deploy_idx])
            print("Length:", mctrack_length_v[deploy_idx])
            print("Idx:", mctrack_idx_v[deploy_idx])
            print("X:", x_starts_v[deploy_idx])
            print("Y:", y_starts_v[deploy_idx])
            n_steps = 0
            this_step = [step_images[0]] #Seed Vertex into steplist
            all_steps = [step_images[0]] #seed vertex into steplist
            is_endpoint_score = False
            hidden_in = None
            cell_in   = None
            crop_center_x = x_starts_v[deploy_idx]
            crop_center_y = y_starts_v[deploy_idx]
            while is_endpoint_score == False:
                print("Running step", n_steps)
                vis_wirecrops_v.append(cropped_np(wire_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING']))
                # Last Prep Before Fwd
                # step_images_in = prepare_sequence_steps(this_step).to(torch.device(PARAMS['DEVICE']))
                # next_steps_pred_scores, endpoint_scores, hidden_out, cell_out = model(step_images_in,(hidden_in, cell_in))
                # hidden_in = hidden_out
                # cell_in   = cell_out

                step_images_in = prepare_sequence_steps(all_steps).to(torch.device(PARAMS['DEVICE']))
                next_steps_pred_scores, endpoint_scores, hidden_out, cell_out = model(step_images_in)
                hidden_in = hidden_out
                cell_in   = cell_out
                # End Fwd
                npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)
                is_endpoint_score = (np.argmax(endpoint_scores.cpu().detach().numpy(),axis=1)[-1] == True)

                # optimizer.step()
                np_idx_v = make_prediction_vector(PARAMS, np_pred)
                pred_x, pred_y = unflatten_pos(np_idx_v[-1], PARAMS['PADDING']*2+1)
                vispred_v.append(np_idx_v[-1])

                # Get Next Step Stuff
                crop_center_x = crop_center_x - PARAMS['PADDING'] + pred_x
                crop_center_y = crop_center_y - PARAMS['PADDING'] + pred_y
                this_step = [cropped_np(larmatch_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING'])]
                all_steps.append(cropped_np(larmatch_im_v[deploy_idx], crop_center_x, crop_center_y, PARAMS['PADDING']))
                if is_endpoint_score == True:
                    print("Endpoint Net Says Stop Here!")
                if n_steps == PARAMS['MAX_STEPS']:
                    is_endpoint_score = True
                    print("Catch on Making too Many Steps says end here!")
                n_steps += 1
            end_time = time.time()
            print(end_time - start_time, "Seconds Passed to deploy")
            thistrack_dir='images/Deploy_'+str(entries_v[deploy_idx])+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+"/"
            if not os.path.exists(thistrack_dir):
                os.mkdir(thistrack_dir)
            make_steps_images(np.stack(vis_wirecrops_v,axis=0),thistrack_dir+'Dep_ImStep',PARAMS['PADDING']*2+1,pred=vispred_v,targ=None)
            save_im(wire_im_v[deploy_idx],thistrack_dir+'Dep_ImFull',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
            pdf_name = 'Deploy_'+str(entries_v[deploy_idx])+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+".pdf"
            convert_cmd = "convert "+thistrack_dir+"*.png "+'images/'+pdf_name
            print(convert_cmd)
            os.system(convert_cmd)

    print()
    print("End of Deploy")
    print()



    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
