import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from larlite import larlite
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint, cropped_np, make_steps_images
from MiscFunctions import save_im, save_im_trackline, removeChargeOnTrackSegment
from MiscFunctions import removeTrackWidth, removeTrackWidth_v2
from MiscFunctions import getProngDict, saveProngDict, save_im_multitracks3D
from MiscFunctions import removeDupTracks
from DataLoader3D_Deploy import DataLoader3D_Deploy, getprojectedpixel
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

def build_track(xs,ys,zs):

    thisTrack = larlite.track()
    npts = len(xs)
    print("Building Track")
    for pt in range(npts):
        # if pt == 0:
            # print(xs[pt],ys[pt],zs[pt],0)
        # else:
        #     dist = ((xs[pt] - xs[pt-1])**2 + (ys[pt] - ys[pt-1])**2 + (zs[pt] - zs[pt-1])**2)**0.5
            # print(xs[pt],ys[pt],zs[pt],dist)
        thisPT = ROOT.TVector3(xs[pt],ys[pt],zs[pt])
        thisTrack.add_vertex(thisPT)
        thisDir = 0
        if pt != npts-1:
            thisDir = ROOT.TVector3(xs[pt+1]-xs[pt],ys[pt+1]-ys[pt],zs[pt+1]-xs[pt])
        else:
            thisDir = ROOT.TVector3(xs[npts-1]-xs[npts-2],ys[npts-1]-ys[npts-2],zs[npts-1]-zs[npts-2])
        thisTrack.add_direction(thisDir)
    return thisTrack


PARAMS = {}

PARAMS['USE_CONV_IM'] = True
PARAMS['LARVOXEL_CKPT'] = '/home/jmills/workdir/TrackWalker/larvoxel_ckpt/lv.multidecoder.weights.10600th.tar'
PARAMS['LARVOX_CFG'] = '/home/jmills/workdir/ubdl_gen2/larflow/larmatchnet/config_voxelmultidecoder.yaml'

# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Sep30_09-08-40_mayer/TrackerCheckPoint_10_Fin.pt" #Overtrained
PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Oct29_20-21-29_mayer/TrackerCheckPoint_3_Fin.pt"
PARAMS['MODEL_CHECKPOINT'] = "runs/new_runs/Nov05_16-02-38_mayer/TrackerCheckPoint_11_Fin.pt"
#
PARAMS['MASK_WC'] = False

PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =128
PARAMS['PADDING'] =10
PARAMS['NDIMENSIONS'] = 3 #Not configured to have 3 yet.
PARAMS['VOXCUBESIDE'] = 3
PARAMS['NFEATS'] = 32
PARAMS['EMBEDDING_DIM'] = (PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)*PARAMS['NFEATS'] # N_Features
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
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =False # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point

PARAMS['DEVICE'] = 'cuda:0'

PARAMS['LOAD_SIZE']  = 50 #Number of Entries to Load training tracks from
# PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
# PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
# PARAMS['VAL_SAMPLE_SIZE'] = 100

PARAMS['AREA_TARGET'] = False   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

PARAMS['CONVERT_OUT_TO_DIST'] = 6.0


# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/model_checkpoints/TrackerCheckPoint_1_Fin.pt"
PARAMS['START_ENTRY'] = 1
PARAMS['END_ENTRY'] = 5 #52
PARAMS['MAX_STEPS'] = 100
# PARAMS['SHOWERMODE'] = False

PARAMS['OUTROOTFILEDIR'] = "ROOTDeployFiles/"
PARAMS['OUTROOTNAME'] ="" # If empty, save formatted based on input file name

def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")
    begin_time = time.time()

    output_dim = PARAMS['NDIMENSIONS'] # Shift X, Shift Y

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
    DataLoader.currentEntry = PARAMS['START_ENTRY']

    prongDict = getProngDict()

    predictionProbFirstStep_h = ROOT.TH1D("predictionProbFirstStep_h", "predictionProbFirstStep_h",50,0.,1.)

    output_storage_manager = larlite.storage_manager(larlite.storage_manager.kWRITE)
    outputFileName = PARAMS["OUTROOTNAME"]
    if outputFileName == "":
        outputFileName = "TrackerNetOut_"+os.path.split(PARAMS['INFILE'])[1]
    output_storage_manager.set_out_filename(PARAMS['OUTROOTFILEDIR']+outputFileName)
    output_storage_manager.open()

    with torch.no_grad():
        model.eval()
        run_backwards = False

        deploy_idx = -1
        print("Trying to Load ", PARAMS['START_ENTRY'], " to ", PARAMS['END_ENTRY'], "entries.")
        for i in range(PARAMS['START_ENTRY'],PARAMS['END_ENTRY']):
            deploy_idx += 1
            deployDict, passFlag = DataLoader.load_deploy_versatile(mode = deployMode, prongDict=prongDict)

            output_storage_manager.set_id(deployDict['run'],deployDict['subrun'],deployDict['event'])
            ev_track_out = output_storage_manager.get_data(larlite.data.kTrack, "RecoTrackNet")


            if passFlag == 0:
                output_storage_manager.next_event()
                continue
            start_time = time.time()
            # mask_im_v = [rewrite   np.ones((deployDict['larvoxfeats'][p].shape[0], deployDict['larvoxfeats'][p].shape[1] )) for p in range(3)]
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

                croppedFeatures = DataLoader.get_cropped_feats(deployDict['larvoxfeats'],deployDict['seedVoxelIdx'],PARAMS)

                trackRemoved_cropFeat     = deployDict['larvoxfeats'].copy()
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
                xs = [current3dPos[0]]
                ys = [current3dPos[1]]
                zs = [current3dPos[2]]
                while is_endpoint_score == False:
                    print(len(endpoint_scores_v))
                    # print("Running step", n_steps)
                    for p in range(3):
                        fullimage_tgraph_v[p].SetPoint(n_steps,centerImgCoords[p+1]+0.5,centerImgCoords[0]+0.5)
                        vis_wirecrops_vv[p].append(cropped_np(deployDict['wireImages_v'][p], centerImgCoords[p+1], centerImgCoords[0], PARAMS['PADDING']))
                    # Remove Charge up Charge From Track
                    # if previous_imgCoords[0] != -1:
                    #     for p in range(3):
                    #         removeChargeOnTrackSegment(trackRemoved_cropFeat[p], trackRemoved_cropFeat[p], mask_im_v[p], int(centerImgCoords[p+1]), int(centerImgCoords[0]), int(previous_imgCoords[p+1]), int(previous_imgCoords[0]))
                    step3d_crops_in_t = prepare_sequence_steps(all_steps).to(torch.device(PARAMS['DEVICE']))
                    xyzShifts_pred_t, endpoint_scores, hidden_out, cell_out = model(step3d_crops_in_t)
                    #Multiply by conversion factor
                    xyzShifts_pred_np = np.rint(xyzShifts_pred_t.cpu().detach().numpy()*PARAMS['CONVERT_OUT_TO_DIST'])

                    # End Fwd
                    endpoint_score = np.exp(endpoint_scores.cpu().detach().numpy())[-1]
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


                    # Get Next Step Stuff
                    # current_dx = crop_center_x - previous_center_x
                    # current_dy = crop_center_y - previous_center_y
                    previous_imgCoords = centerImgCoords
                    # print(currentVoxIdx, " Idx")
                    changeVoxIdx = [xyzShifts_pred_np[-1,0], xyzShifts_pred_np[-1,1], xyzShifts_pred_np[-1,2]]
                    currentVoxIdx = [currentVoxIdx[p]+changeVoxIdx[p] for p in range(3)]

                    current3DPos = DataLoader.voxelator.get3dCoord(currentVoxIdx)
                    # print("COORDS",current3DPos[0], current3DPos[1],current3DPos[2])
                    xs.append(current3DPos[0])
                    ys.append(current3DPos[1])
                    zs.append(current3DPos[2])

                    centerImgCoords = getprojectedpixel(deployDict['meta'], current3DPos[0], current3DPos[1],current3DPos[2],True)
                    net_feats_in = DataLoader.get_cropped_feats(deployDict['larvoxfeats'],currentVoxIdx,PARAMS)

                    this_step = [net_feats_in]
                    all_steps.append(net_feats_in)

                    if (is_endpoint_score == True) or (n_steps == PARAMS['MAX_STEPS']):
                        print("Track Removed Image Not Configged")
                        # removeTrackWidth_v2(trackRemoved_cropFeat, trackRemoved_cropFeat, mask_im, int(crop_center_x), int(crop_center_y), int(current_dx), int(current_dy))
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

                # Save track to output root file:
                thisTrack = build_track(xs,ys,zs)
                ev_track_out.push_back(thisTrack)

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
                # save_im(trackRemoved_cropFeat,thistrack_dir+'Dep_ImRemoved',x_starts_v[deploy_idx],y_starts_v[deploy_idx],canv_x = 4000,canv_y = 1000)
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

                # this_wire_im_v     = trackRemoved_cropFeat
                # this_larmatch_im_v = trackRemoved_cropFeat
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
            print("SAVING NEXT EVENT")
            output_storage_manager.next_event()
    output_storage_manager.close()
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
    print(full_time/(PARAMS['END_ENTRY']-PARAMS['START_ENTRY']), "Seconds per event")

    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
