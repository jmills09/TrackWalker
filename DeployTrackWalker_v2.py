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
from MiscFunctions import save_im, save_im_trackline, removeChargeOnTrackSegment
from MiscFunctions import removeTrackWidth, removeTrackWidth_v2
from MiscFunctions import getProngDict, saveProngDict, save_im_multitracks
from MiscFunctions import removeDupTracks
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
# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Jul27_13-09-17_mayer/TrackerCheckPoint_10_Fin.pt"
PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Aug12_13-25-56_mayer/TrackerCheckPoint_10_Fin.pt"
#
PARAMS['MASK_WC'] = False

PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =1024
PARAMS['PADDING'] =10
PARAMS['APPEND_WIREIM'] = True
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) # N_Features
if PARAMS['USE_CONV_IM']:
    if PARAMS['APPEND_WIREIM']:
        PARAMS['EMBEDDING_DIM'] = PARAMS['EMBEDDING_DIM']*17 # 16 Features per pixel in larmatch plus wireim
    else:
        PARAMS['EMBEDDING_DIM'] = PARAMS['EMBEDDING_DIM']*16 # 16 Features per pixel in larmatch
PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)+1 # Bonus Class is for the end of track class
PARAMS['TRACKEND_CLASS'] = (PARAMS['PADDING']*2+1)**2
PARAMS['CENTERPOINT_ISEND'] = True
if PARAMS['CENTERPOINT_ISEND']:
     PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) #No bonus end of track class
     PARAMS['TRACKEND_CLASS'] = (PARAMS['NUM_CLASSES']-1)/2
# PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/nue_intrinsics_run3b/nue_intrinsics_run3b_merged_dlreco.root"

PARAMS['TRACK_IDX'] =0
PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
# TENSORDIR'] ="runs/Pad20_Hidden1024_500Entries"
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
PARAMS['NDIMENSIONS'] = 2 #Not configured to have 3 yet.

PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50 #Number of Entries to Load training tracks from
# PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
# PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
# PARAMS['VAL_SAMPLE_SIZE'] = 100

PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_1_Fin.pt"
PARAMS['NUM_DEPLOY'] = 341 #52
PARAMS['MAX_STEPS'] = 200
PARAMS['SHOWERMODE'] = True

def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")
    begin_time = time.time()

    output_dim = PARAMS['NUM_CLASSES']

    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))

    if (PARAMS['DEVICE'] != 'cpu'):
        model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT'], map_location={'cpu':PARAMS['DEVICE'],'cuda:0':PARAMS['DEVICE'],'cuda:1':PARAMS['DEVICE'],'cuda:2':PARAMS['DEVICE'],'cuda:3':PARAMS['DEVICE']}))
    else:
        model.load_state_dict(torch.load(PARAMS['MODEL_CHECKPOINT'], map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))

    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']

    deployMode = "MCNU_BNB"
    images_dir = ''
    if deployMode == "MCNU_NUE":
        PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/nue_intrinsics_run3b/nue_intrinsics_run3b_merged_dlreco.root"
        images_dir = 'deployIm_MCNU_NUE/'
    elif deployMode == "MCNU_BNB":
        images_dir = 'deployIm_MCNU_BNB/'
        PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
    else:
        images_dir = "deployIm_dump/"

    DataLoader = DataLoader_MC(PARAMS,all_train=False, all_valid = True, deploy=True)
    DataLoader.verbose = True
    DataLoader.currentEntry = 0

    prongDict = getProngDict()

    predictionProbFirstStep_h = ROOT.TH1D("predictionProbFirstStep_h", "predictionProbFirstStep_h",50,0.,1.)


    with torch.no_grad():
        model.eval()
        run_backwards = False

        deploy_idx = -1
        print("Trying to Load ", PARAMS['NUM_DEPLOY'], "entries.")
        for i in range(0,PARAMS['NUM_DEPLOY']):
            deploy_idx += 1
            deployDict, passFlag = DataLoader.load_deploy_versatile(mode = deployMode, prongDict=prongDict)
            if passFlag == 0:
                continue
            start_time = time.time()
            mask_im = np.ones((deployDict['featureImages'].shape[0],deployDict['featureImages'].shape[1]))
            recoTrackEnds = []
            recoTrack_tgraphs_v    = []
            for repeat in range(10):
                goodRecoProng = True
                n_steps = 0
                crop_center_x = deployDict['seedX']
                crop_center_y = deployDict['seedY']
                croppedFeatures = cropped_np(deployDict['featureImages'], crop_center_x, crop_center_y, PARAMS['PADDING'])
                trackRemoved_wire_im     = deployDict['featureImages'][:,:,-1].copy()
                trackRemoved_larmatch_im = deployDict['featureImages'][:,:,0:-1].copy()
                this_step = [croppedFeatures] #Seed Vertex into steplist
                all_steps = [croppedFeatures] #Seed vertex into steplist
                is_endpoint_score = False
                vis_wirecrops_v = []
                vis_predStarts_v = []
                endpoint_scores_v = []

                previous_center_x = -1
                previous_center_y = -1

                fullimage_tgraph = ROOT.TGraph()

                np_pred_vtx = None
                # Build Points for Track
                while is_endpoint_score == False:
                    print("Running step", n_steps)
                    fullimage_tgraph.SetPoint(n_steps,crop_center_x+0.5,crop_center_y+0.5)
                    vis_wirecrops_v.append(cropped_np(deployDict['featureImages'][-1], crop_center_x, crop_center_y, PARAMS['PADDING']))
                    # Remove Charge up Charge From Track
                    if previous_center_x != -1:
                        removeChargeOnTrackSegment(trackRemoved_wire_im, trackRemoved_larmatch_im, mask_im, int(crop_center_x), int(crop_center_y), int(previous_center_x), int(previous_center_y))
                    step_images_in = prepare_sequence_steps(all_steps).to(torch.device(PARAMS['DEVICE']))
                    next_steps_pred_scores, endpoint_scores, hidden_out, cell_out = model(step_images_in)
                    # End Fwd
                    npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                    np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)
                    endpoint_score = np.exp(endpoint_scores.cpu().detach().numpy())[-1]
                    print("Endpoint:", endpoint_score)
                    endpoint_scores_v.append(endpoint_score[1])
                    is_endpoint_score = (endpoint_score[1] >= 0.5)
                    if n_steps == 0 and repeat == 0:
                        np_pred_vtx = np_pred[0]
                        for aa in range(np_pred.shape[1]):
                            for bb in range(np_pred.shape[2]):
                                predictionProbFirstStep_h.Fill(np_pred[0,aa,bb])
                    if n_steps == 0:
                        np_prob_alter = cropped_np(mask_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                        np_pred[0] = np_pred[0]*np_prob_alter
                        np_pred_vtx = np_pred[0]
                        # save_im(np_pred[0],images_dir+'TrackStartScores_'+str(deployDict['entry'])+"_"+str(repeat))
                        # save_im(np_prob_alter,images_dir+'AlterArray_'+str(deployDict['entry'])+"_"+str(repeat))

                        np_pred_sum = np.zeros((np_pred.shape[1],np_pred.shape[2]))
                        for xx in range(np_pred_sum.shape[0]):
                            for yy in range(np_pred_sum.shape[1]):
                                xmin = xx-1 if xx-1 >= 0 else 0
                                xmax = xx+2 if xx+2 <  np_pred.shape[1] else np_pred.shape[1]
                                ymin = yy-1 if yy-1 >= 0 else 0
                                ymax = yy+2 if yy+2 <  np_pred.shape[2] else np_pred.shape[2]
                                np_pred_sum[xx,yy] = np.sum(np_pred[0,xmin:xmax,ymin:ymax])
                        # save_im(np_pred_sum,images_dir+'/TrackStartScoresSum_'+str(deployDict['entry'])+"_"+str(repeat))
                        if is_endpoint_score:
                            goodRecoProng = False
                        elif np.max(np_pred_vtx) < 0.1:
                            is_endpoint_score = True
                            goodRecoProng = False





                    np_idx_v = make_prediction_vector(PARAMS, np_pred)
                    pred_x, pred_y = unflatten_pos(np_idx_v[-1], PARAMS['PADDING']*2+1)
                    vis_predStarts_v.append(np_idx_v[-1])

                    # Get Next Step Stuff
                    current_dx = crop_center_x - previous_center_x
                    current_dy = crop_center_y - previous_center_y
                    previous_center_x = crop_center_x
                    previous_center_y = crop_center_y

                    crop_center_x = crop_center_x - PARAMS['PADDING'] + pred_x
                    crop_center_y = crop_center_y - PARAMS['PADDING'] + pred_y

                    net_feats_in = cropped_np(deployDict['featureImages'], crop_center_x, crop_center_y, PARAMS['PADDING'])

                    this_step = [net_feats_in]
                    all_steps.append(net_feats_in)

                    if (is_endpoint_score == True) or (n_steps == PARAMS['MAX_STEPS']):
                        removeTrackWidth_v2(trackRemoved_wire_im, trackRemoved_larmatch_im, mask_im, int(crop_center_x), int(crop_center_y), int(current_dx), int(current_dy))
                        print("Endpoint Net Says Stop Here!")
                        # Check against other track ends
                        # thisEnd = (crop_center_x,crop_center_y)
                        # for end in recoTrackEnds:
                        #     dist = ((thisEnd[0] - end[0])**2 + (thisEnd[1] - end[1])**2)**0.5
                        #     if dist < 8:
                        #         goodRecoProng = False

                        if goodRecoProng:
                            # recoTrackEnds.append(thisEnd)
                            recoTrack_tgraphs_v.append(fullimage_tgraph)
                        is_endpoint_score = True


                    n_steps += 1

                end_time = time.time()
                print(end_time - start_time, "Seconds Passed to deploy After Preprocessing Starts")
                if not goodRecoProng:
                    print("This track suggestion didn't have high enough start prob, not saving.")
                    continue

                # Save Ev Displays
                # thistrack_dir=images_dir+'Deploy_'+str(entries_v[deploy_idx])+"_"+str(mctrack_idx_v[deploy_idx])+"_"+str(repeat)+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+"/"
                # if not os.path.exists(thistrack_dir):
                #     os.mkdir(thistrack_dir)
                # make_steps_images(np.stack(vis_wirecrops_v,axis=0),thistrack_dir+'Dep_ImStep',PARAMS['PADDING']*2+1,pred=vis_predStarts_v,targ=None,endpoint_scores=endpoint_scores_v)
                # save_im(np_pred_vtx,thistrack_dir+'Dep_ImStartProb'+str(entries_v[deploy_idx]))
                # save_im(this_wire_im,thistrack_dir+'Dep_ImFull',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(ssnettrack_ims_v[deploy_idx],thistrack_dir+'Dep_ImFull_Track',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(ssnetshower_ims_v[deploy_idx],thistrack_dir+'Dep_ImFull_Shower',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(trackRemoved_wire_im,thistrack_dir+'Dep_ImRemoved',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im_trackline(this_wire_im,fullimage_tgraph,thistrack_dir+'Dep_ImFullTrackLine', canv_x = 4000,canv_y = 1000)
                #
                # pdf_name = 'Deploy_'+str(entries_v[deploy_idx])+"_"+str(mctrack_idx_v[deploy_idx])+"_"+str(repeat)+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+".pdf"
                # convert_cmd = "convert "+thistrack_dir+"*.png "+images_dir+pdf_name
                # print(convert_cmd)
                # os.system(convert_cmd)

                # #####
                # # Time to Debug:
                # exp_wire_image = np.expand_dims(this_wire_im,axis=2)
                # net_feats_in = np.concatenate((exp_wire_image,this_larmatch_im),axis=2)
                # for abc in range(net_feats_in.shape[2]):
                #     save_im(net_feats_in[2950:3100,750:850,abc],"larmatchfeat_im_test2/"+str(repeat)+"/Feat_"+str(abc).zfill(2), canv_x = 1500,canv_y = 1000)
                #     os.system("convert larmatchfeat_im_test2/"+str(repeat)+"/Feat* larmatchfeat_im_test2/LarMatchFeatTest_"+str(repeat)+".pdf")
                # #####
                # Set Next Image to start with this track removed

                this_wire_im     = trackRemoved_wire_im
                this_larmatch_im = trackRemoved_larmatch_im

            recoTrack_tgraphs_v = removeDupTracks(recoTrack_tgraphs_v)

            multTrackName = images_dir+"MultiTracks_"+str(deployDict['entry']).zfill(3)
            save_im_multitracks(multTrackName, deployDict['featureImages'][:,:,-1],recoTrack_tgraphs_v,deployDict['seedX'],deployDict['seedY'])

            print("Values")
            print("Reco    :   ", len(recoTrack_tgraphs_v))
            print("MC      :   ", deployDict['mcProngs'])
            print("MCThresh:   ", deployDict['mcProngs_thresh'])
            print("Reco - MC:  ",len(recoTrack_tgraphs_v) - deployDict['mcProngs'])
            print("Reco - MCTH:",len(recoTrack_tgraphs_v) - deployDict['mcProngs_thresh'])
            prongDict['mcProngs_h'].Fill(deployDict['mcProngs'])
            prongDict['mcProngs_thresh_h'].Fill(deployDict['mcProngs_thresh'])
            prongDict['recoProngs_h'].Fill(len(recoTrack_tgraphs_v))
            prongDict['reco_m_mcProngs_h'].Fill(len(recoTrack_tgraphs_v) - deployDict['mcProngs'])
            prongDict['reco_m_mcProngs_thresh_h'].Fill(len(recoTrack_tgraphs_v) - deployDict['mcProngs_thresh'])
    dir = ''
    if deployMode == "MCNU_NUE":
        dir = 'prongPlots_nue/'
    else:
        dir = 'prongPlots_bnb/'
    saveProngDict(prongDict,dir,deployMode)
    tmpcan = ROOT.TCanvas('canv','canv',1200,1000)
    if deployMode == "MCNU_NUE":
        predictionProbFirstStep_h.SetTitle("First Step Confidence Distribution NUE")
        predictionProbFirstStep_h.Draw()
        tmpcan.SaveAs("NUE_FIRSTSTEP_PROB.png")
    else:
        predictionProbFirstStep_h.SetTitle("First Step Confidence Distribution BNB")
        predictionProbFirstStep_h.Draw()
        tmpcan.SaveAs("BNB_FIRSTSTEP_PROB.png")
    tmpcan.SetLogy()
    if deployMode == "MCNU_NUE":
        predictionProbFirstStep_h.SetTitle("First Step Confidence Distribution NUE")
        predictionProbFirstStep_h.Draw()
        tmpcan.SaveAs("NUE_FIRSTSTEP_PROB_log.png")
    else:
        predictionProbFirstStep_h.SetTitle("First Step Confidence Distribution BNB")
        predictionProbFirstStep_h.Draw()
        tmpcan.SaveAs("BNB_FIRSTSTEP_PROB_log.png")
    print()
    print("End of Deploy")
    print()

    final_time = time.time()
    full_time = final_time - begin_time
    print(full_time, "Seconds spent on script")
    print(full_time/PARAMS['NUM_DEPLOY'], "Seconds per event")

    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
