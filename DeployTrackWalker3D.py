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
from MiscFunctions import getProngDict, saveProngDict, save_im_multitracks3D
from MiscFunctions import removeDupTracks
from DataLoader3D_Deploy import DataLoader3D_Deploy, getprojectedpixel
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
# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Sep30_09-08-40_mayer/TrackerCheckPoint_10_Fin.pt" #Overtrained
PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Oct04_13-37-26_mayer/TrackerCheckPoint_0_20000.pt"
#
PARAMS['MASK_WC'] = False

PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =1024
PARAMS['PADDING'] =10
PARAMS['APPEND_WIREIM'] = True
PARAMS['NDIMENSIONS'] = 3 #Not configured to have 3 yet.
PARAMS['VOXCUBESIDE'] = 3
PARAMS['NFEATS'] = 17
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)*PARAMS['NFEATS']*3 # N_Features*3planes
PARAMS['CENTERPOINT_ISEND'] = True
PARAMS['NUM_CLASSES'] = PARAMS['VOXCUBESIDE']**3
PARAMS['TRACKEND_CLASS'] = (PARAMS['NUM_CLASSES']-1)/2
PARAMS['TRACKEND_CLASS'] = (PARAMS['PADDING']*2+1)**2
PARAMS['CENTERPOINT_ISEND'] = True
PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
# PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/FailedDLReco.root"
# PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/nue_intrinsics_run3b/nue_intrinsics_run3b_merged_dlreco.root"

PARAMS['TRACK_IDX'] =0
PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
# TENSORDIR'] ="runs/Pad20_Hidden1024_500Entries"
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point

PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50 #Number of Entries to Load training tracks from
# PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
# PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
# PARAMS['VAL_SAMPLE_SIZE'] = 100

PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_1_Fin.pt"
PARAMS['START_ENTRY'] = 100
PARAMS['END_ENTRY'] = 341 #52
PARAMS['MAX_STEPS'] = 500
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
        images_dir = 'images3d_NUE/'
    elif deployMode == "MCNU_BNB":
        images_dir = 'images3d_BNB/'
        PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
    else:
        images_dir = "images3d_DUMP/"

    DataLoader = DataLoader3D_Deploy(PARAMS,all_train=False, all_valid = True,deploy=True)
    DataLoader.verbose = True
    DataLoader.currentEntry = 0

    prongDict = getProngDict()

    predictionProbFirstStep_h = ROOT.TH1D("predictionProbFirstStep_h", "predictionProbFirstStep_h",50,0.,1.)


    with torch.no_grad():
        model.eval()
        run_backwards = False

        deploy_idx = -1
        print("Trying to Load ", PARAMS['START_ENTRY'], " to ", PARAMS['END_ENTRY'], "entries.")
        for i in range(PARAMS['START_ENTRY'],PARAMS['END_ENTRY']):
            deploy_idx += 1
            deployDict, passFlag = DataLoader.load_deploy_versatile(mode = deployMode, prongDict=prongDict)
            if passFlag == 0:
                continue
            start_time = time.time()
            mask_im_v = [np.ones((deployDict['featureImages_v'][p].shape[0], deployDict['featureImages_v'][p].shape[1] )) for p in range(3)]
            recoTrackEnds3d = []
            recoTrack_tgraphs_vv    = [[],[],[]]
            vertexImgCoords = []
            for repeat in range(1):
                goodRecoProng = True
                n_steps = 0
                current3dPos = deployDict['seed3dPos']
                currentVoxIdx = deployDict['seedVoxelIdx']
                centerImgCoords = getprojectedpixel(deployDict['meta'], deployDict['seed3dPos'][0],deployDict['seed3dPos'][1],deployDict['seed3dPos'][2],True)
                vertexImgCoords = centerImgCoords
                print(vertexImgCoords, "Vertex")
                croppedFeatures = np.stack([cropped_np(deployDict['featureImages_v'][p], centerImgCoords[p+1], centerImgCoords[0], PARAMS['PADDING']) for p in range(3)],axis=0)
                trackRemoved_wire_im_v     = [deployDict['featureImages_v'][p][:,:,-1].copy() for p in range(3)]
                trackRemoved_larmatch_im_v = [deployDict['featureImages_v'][p][:,:,0:-1].copy() for p in range(3)]
                this_step = [croppedFeatures] #Seed Vertex into steplist
                all_steps = [croppedFeatures] #Seed vertex into steplist
                is_endpoint_score = False
                vis_wirecrops_vv = [[],[],[]]
                vis_predStarts_vv = [[],[],[]]
                endpoint_scores_v = []

                previous_imgCoords = [-1,-1,-1,-1]

                fullimage_tgraph_v = [ROOT.TGraph() for p in range(3)]

                np_pred_vtx = None
                # Build Points for Track
                while is_endpoint_score == False:
                    # print("Running step", n_steps)
                    for p in range(3):
                        fullimage_tgraph_v[p].SetPoint(n_steps,centerImgCoords[p+1]+0.5,centerImgCoords[0]+0.5)
                        vis_wirecrops_vv[p].append(cropped_np(deployDict['wireImages_v'][p], centerImgCoords[p+1], centerImgCoords[0], PARAMS['PADDING']))
                    # Remove Charge up Charge From Track
                    if previous_imgCoords[0] != -1:
                        for p in range(3):
                            removeChargeOnTrackSegment(trackRemoved_wire_im_v[p], trackRemoved_larmatch_im_v[p], mask_im_v[p], int(centerImgCoords[p+1]), int(centerImgCoords[0]), int(previous_imgCoords[p+1]), int(previous_imgCoords[0]))
                    step_images_in = prepare_sequence_steps(all_steps).to(torch.device(PARAMS['DEVICE']))
                    next_steps_pred_scores, endpoint_scores, hidden_out, cell_out = model(step_images_in)
                    # End Fwd
                    npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                    np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['VOXCUBESIDE'],PARAMS['VOXCUBESIDE'],PARAMS['VOXCUBESIDE'])
                    endpoint_score = np.exp(endpoint_scores.cpu().detach().numpy())[-1]
                    # print("Endpoint:", endpoint_score)
                    endpoint_scores_v.append(endpoint_score[1])
                    is_endpoint_score = (endpoint_score[1] >= 0.5)
                    # print("Pred First Step heatmap not configged")
                    # if n_steps == 0 and repeat == 0:
                    #     np_pred_vtx = np_pred[0]
                    #     for aa in range(np_pred.shape[1]):
                    #         for bb in range(np_pred.shape[2]):
                    #             predictionProbFirstStep_h.Fill(np_pred[0,aa,bb])
                    if n_steps == 0:
                        print("NOT ALTERING FIRST STEP FOR FUTURE TRACKS, not configed for 3d")
                        # np_prob_alter = [cropped_np(mask_im_v[p], centerImgCoords[p+1], centerImgCoords[0], PARAMS['PADDING']) for p in range(3)]
                        # np_pred[0] = np_pred[0]*np_prob_alter
                        # np_pred_vtx = np_pred[0]
                        # # save_im(np_pred[0],images_dir+'TrackStartScores_'+str(deployDict['entry'])+"_"+str(repeat))
                        # # save_im(np_prob_alter,images_dir+'AlterArray_'+str(deployDict['entry'])+"_"+str(repeat))
                        #
                        # np_pred_sum = np.zeros((np_pred.shape[1],np_pred.shape[2]))
                        # for xx in range(np_pred_sum.shape[0]):
                        #     for yy in range(np_pred_sum.shape[1]):
                        #         xmin = xx-1 if xx-1 >= 0 else 0
                        #         xmax = xx+2 if xx+2 <  np_pred.shape[1] else np_pred.shape[1]
                        #         ymin = yy-1 if yy-1 >= 0 else 0
                        #         ymax = yy+2 if yy+2 <  np_pred.shape[2] else np_pred.shape[2]
                        #         np_pred_sum[xx,yy] = np.sum(np_pred[0,xmin:xmax,ymin:ymax])
                        # # save_im(np_pred_sum,images_dir+'/TrackStartScoresSum_'+str(deployDict['entry'])+"_"+str(repeat))
                        # if is_endpoint_score:
                        #     goodRecoProng = False
                        # elif np.max(np_pred_vtx) < 0.1:
                        #     is_endpoint_score = True
                        #     goodRecoProng = False




                    np_idx_v = make_prediction_vector(PARAMS, np_pred)
                    pred_x, pred_y, pred_z = unflatten_pos(np_idx_v[-1], PARAMS['VOXCUBESIDE'])
                    vis_predStarts_vv.append(np_idx_v[-1])

                    # Get Next Step Stuff
                    # current_dx = crop_center_x - previous_center_x
                    # current_dy = crop_center_y - previous_center_y
                    previous_imgCoords = centerImgCoords
                    changeVoxIdx = [pred_x - 1, pred_y - 1, pred_z - 1]
                    currentVoxIdx = [currentVoxIdx[p]+changeVoxIdx[p] for p in range(3)]
                    current3DPos = DataLoader.voxelator.get3dCoord(currentVoxIdx)
                    centerImgCoords = getprojectedpixel(deployDict['meta'], current3DPos[0], current3DPos[1],current3DPos[2],True)
                    net_feats_in = np.stack([cropped_np(deployDict['featureImages_v'][p], centerImgCoords[p+1], centerImgCoords[0], PARAMS['PADDING']) for p in range(3)], axis=0)

                    this_step = [net_feats_in]
                    all_steps.append(net_feats_in)

                    if (is_endpoint_score == True) or (n_steps == PARAMS['MAX_STEPS']):
                        print("Track Removed Image Not Configged")
                        # removeTrackWidth_v2(trackRemoved_wire_im_v, trackRemoved_larmatch_im_v, mask_im, int(crop_center_x), int(crop_center_y), int(current_dx), int(current_dy))
                        print("Endpoint Net Says Stop Here!")
                        # Check against other track ends
                        # thisEnd = (crop_center_x,crop_center_y)
                        # for end in recoTrackEnds3d:
                        #     dist = ((thisEnd[0] - end[0])**2 + (thisEnd[1] - end[1])**2)**0.5
                        #     if dist < 8:
                        #         goodRecoProng = False

                        if goodRecoProng:
                            # recoTrackEnds3d.append(thisEnd)
                            for p in range(3):
                                recoTrack_tgraphs_vv[p].append(fullimage_tgraph_v[p])
                        is_endpoint_score = True


                    n_steps += 1

                end_time = time.time()
                print(end_time - start_time, "Seconds Passed to deploy After Preprocessing Starts")
                if not goodRecoProng:
                    print("This track suggestion didn't have high enough start prob, not saving.")
                    continue

                # Save Ev Displays
                # thistrack_dir=images_dir+'Deploy_'+str(entries_v[deploy_idx])+"_"+str(mctrack_idx_v[deploy_idx])+"_"+str(repeat)+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+"/"

                thistrack_dir=images_dir
                # if not os.path.exists(thistrack_dir):
                #     os.mkdir(thistrack_dir)
                # make_steps_images(np.stack(vis_wirecrops_vv,axis=0),thistrack_dir+'Dep_ImStep',PARAMS['PADDING']*2+1,pred=vis_predStarts_vv,targ=None,endpoint_scores=endpoint_scores_v)
                # save_im(np_pred_vtx,thistrack_dir+'Dep_ImStartProb'+str(entries_v[deploy_idx]))
                # save_im(this_wire_im_v,thistrack_dir+'Dep_ImFull',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(ssnettrack_ims_v[deploy_idx],thistrack_dir+'Dep_ImFull_Track',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(ssnetshower_ims_v[deploy_idx],thistrack_dir+'Dep_ImFull_Shower',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                # save_im(trackRemoved_wire_im_v,thistrack_dir+'Dep_ImRemoved',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
                for p in range(3):
                    save_im_trackline(deployDict['wireImages_v'][p].copy(),fullimage_tgraph_v[p],thistrack_dir+str(i).zfill(3)+'TrackLine_'+"_"+str(p), canv_x = 4000,canv_y = 1000)
                #
                # pdf_name = 'Deploy_'+str(entries_v[deploy_idx])+"_"+str(mctrack_idx_v[deploy_idx])+"_"+str(repeat)+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+".pdf"
                # convert_cmd = "convert "+thistrack_dir+"*.png "+images_dir+pdf_name
                # print(convert_cmd)
                # os.system(convert_cmd)

                # #####
                # # Time to Debug:
                # exp_wire_image = np.expand_dims(this_wire_im_v,axis=2)
                # net_feats_in = np.concatenate((exp_wire_image,this_larmatch_im_v),axis=2)
                # for abc in range(net_feats_in.shape[2]):
                #     save_im(net_feats_in[2950:3100,750:850,abc],"larmatchfeat_im_test2/"+str(repeat)+"/Feat_"+str(abc).zfill(2), canv_x = 1500,canv_y = 1000)
                #     os.system("convert larmatchfeat_im_test2/"+str(repeat)+"/Feat* larmatchfeat_im_test2/LarMatchFeatTest_"+str(repeat)+".pdf")
                # #####
                # Set Next Image to start with this track removed

                # this_wire_im_v     = trackRemoved_wire_im_v
                # this_larmatch_im_v = trackRemoved_larmatch_im_v
            print("Removing Multiple tracks not configged")
            # recoTrack_tgraphs_vv = removeDupTracks(recoTrack_tgraphs_vv)

            multTrackName = images_dir+str(deployDict['entry']).zfill(3)+"MultiTracks_"
            save_im_multitracks3D(multTrackName, deployDict['wireImages_v'],recoTrack_tgraphs_vv,vertexImgCoords)

            print("Values")
            print("Reco    :   ", len(recoTrack_tgraphs_vv))
            print("MC      :   ", deployDict['mcProngs'])
            print("MCThresh:   ", deployDict['mcProngs_thresh'])
            print("Reco - MC:  ",len(recoTrack_tgraphs_vv) - deployDict['mcProngs'])
            print("Reco - MCTH:",len(recoTrack_tgraphs_vv) - deployDict['mcProngs_thresh'])
            prongDict['mcProngs_h'].Fill(deployDict['mcProngs'])
            prongDict['mcProngs_thresh_h'].Fill(deployDict['mcProngs_thresh'])
            prongDict['recoProngs_h'].Fill(len(recoTrack_tgraphs_vv))
            prongDict['reco_m_mcProngs_h'].Fill(len(recoTrack_tgraphs_vv) - deployDict['mcProngs'])
            prongDict['reco_m_mcProngs_thresh_h'].Fill(len(recoTrack_tgraphs_vv) - deployDict['mcProngs_thresh'])
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
    print(full_time/(PARAMS['START_ENTRY']-PARAMS['END_ENTRY']), "Seconds per event")

    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
