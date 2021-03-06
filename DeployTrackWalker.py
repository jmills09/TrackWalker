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
PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Jul27_13-09-17_mayer/TrackerCheckPoint_10_Fin.pt"
# PARAMS['MODEL_CHECKPOINT'] = "/home/jmills/workdir/TrackWalker/runs/new_runs/Aug12_13-25-56_mayer/TrackerCheckPoint_10_Fin.pt"

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
PARAMS['NUM_DEPLOY'] = 52
PARAMS['MAX_STEPS'] = 200
PARAMS['SHOWERMODE'] = True

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
    DataLoader.verbose = True
    start_file = 0
    end_file   = 9999999 # int(start_file+PARAMS['NUM_DEPLOY']*1.1+100) #estimate of how many events to get the number of tracks we want to run on

    mcProngs_h = ROOT.TH1D("mcProngs_h", "mcProngs_h",10,0.,10.)
    mcProngs_h.SetTitle("mcProngs")
    mcProngs_h.SetXTitle("nProngs")
    mcProngs_h.SetYTitle("Events in Sample")
    mcProngs_thresh_h = ROOT.TH1D("mcProngs_thresh_h", "mcProngs_thresh_h",10,0.,10.)
    mcProngs_thresh_h.SetTitle("mcProngs")
    mcProngs_thresh_h.SetXTitle("nProngs")
    mcProngs_thresh_h.SetYTitle("Events in Sample")

    recoProngs_h = ROOT.TH1D("recoProngs_h", "recoProngs_h",10,0.,10.)
    recoProngs_h.SetTitle("recoProngs")
    recoProngs_h.SetXTitle("nProngs")
    recoProngs_h.SetYTitle("Events in Sample")
    reco_m_mcProngs_h = ROOT.TH1D("reco_m_mcProngs_h", "reco_m_mcProngs_h",20,-10.,10.)
    reco_m_mcProngs_h.SetTitle("Reco - MC Prongs")
    reco_m_mcProngs_h.SetXTitle("Reco - MC Prongs")
    reco_m_mcProngs_h.SetYTitle("Events in Sample")
    reco_m_mcProngs_thresh_h = ROOT.TH1D("reco_m_mcProngs_thresh_h", "reco_m_mcProngs_thresh_h",20,-10.,10.)
    reco_m_mcProngs_thresh_h.SetTitle("Reco - MC Prongs_thresh")
    reco_m_mcProngs_thresh_h.SetXTitle("Reco - MC Prongs_thresh")
    reco_m_mcProngs_thresh_h.SetYTitle("Events in Sample")



    with torch.no_grad():
        model.eval()
        run_backwards = False
        # deploy_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids, larmatch_im_v, wire_im_v, x_starts_v, y_starts_v = DataLoader.load_dlreco_inputs_onestop_deploy(start_file,end_file,MAX_TRACKS_PULL=PARAMS['NUM_DEPLOY'],run_backwards=run_backwards,is_val=True, showermode = PARAMS['SHOWERMODE'])
        deploy_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids, larmatch_im_v, wire_im_v, x_starts_v, y_starts_v, ssnettrack_ims_v, ssnetshower_ims_v, n_mcProngs, n_mcProngs_thresh = DataLoader.load_dlreco_inputs_onestop_deploy_neutrinovtx(start_file,end_file,MAX_TRACKS_PULL=PARAMS['NUM_DEPLOY'],run_backwards=run_backwards,is_val=True, showermode = PARAMS['SHOWERMODE'])

        print("Total Tracks Grabbed:",len(deploy_data))
        deploy_idx = -1
        # for step_images, targ_next_step_idx, targ_area_next_step in deploy_data:
        for i in range(PARAMS['NUM_DEPLOY']):
            # if i == 0:
            #     continue
            deploy_idx += 1

            start_time = time.time()
            step_images = deploy_data[deploy_idx][0]
            targ_next_step_idx = deploy_data[deploy_idx][1]
            targ_area_next_step = deploy_data[deploy_idx][2]
            # Not yet used.
            stepped_wire_images = deploy_data[deploy_idx][3]
            charge_in_wire_v    = deploy_data[deploy_idx][4]
            charge_in_truth_v   = deploy_data[deploy_idx][5]

            if PARAMS['APPEND_WIREIM']:
                exp_wire_images = np.expand_dims(stepped_wire_images,axis=3)
                step_images = np.concatenate((step_images,exp_wire_images),axis=3)

            print()
            print("Deploy Idx",deploy_idx, "Max Idx:", len(deploy_data))
            print("Track Info:")
            print("Pdg:", mctrack_pdg_v[deploy_idx])
            print("Length:", mctrack_length_v[deploy_idx])
            print("Idx:", mctrack_idx_v[deploy_idx])
            print("X:", x_starts_v[deploy_idx])
            print("Y:", y_starts_v[deploy_idx])
            this_wire_im = wire_im_v[deploy_idx]
            this_larmatch_im = larmatch_im_v[deploy_idx]
            mask_im = np.ones(this_wire_im.shape)
            nRecoTracks = 0
            for repeat in range(10):
                save_trackbranch = True
                trackRemoved_wire_im = this_wire_im.copy()
                trackRemoved_larmatch_im = this_larmatch_im.copy()
                n_steps = 0
                this_step = [step_images[0]] #Seed Vertex into steplist
                all_steps = [step_images[0]] #Seed vertex into steplist
                is_endpoint_score = False
                vis_wirecrops_v = []
                vispred_v = []
                endpoint_scores_v = []
                hidden_in = None
                cell_in   = None

                crop_center_x = x_starts_v[deploy_idx]
                crop_center_y = y_starts_v[deploy_idx]
                if repeat != 0:
                    crop_center_x = x_starts_v[deploy_idx]
                    crop_center_y = y_starts_v[deploy_idx]

                    larmatch_im_crop = cropped_np(this_larmatch_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                    net_feats_in = larmatch_im_crop
                    if PARAMS['APPEND_WIREIM']:
                        wire_feat_cropped_np = cropped_np(this_wire_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                        exp_wire_images = np.expand_dims(wire_feat_cropped_np,axis=2)
                        net_feats_in = np.concatenate((net_feats_in,exp_wire_images),axis=2)
                    this_step = [net_feats_in]
                    all_steps = [net_feats_in]



                previous_center_x = -1
                previous_center_y = -1
                fullimage_tgraph = ROOT.TGraph()
                images_dir = ''
                if PARAMS['SHOWERMODE']:
                    images_dir = 'images_versatile/'
                else:
                    if run_backwards:
                        images_dir = 'images_multi_reverse/'
                    else:
                        images_dir = 'images_multi/'
                np_pred_vtx = None
                while is_endpoint_score == False:
                    print("Running step", n_steps)
                    # if n_steps > 0:
                        # print("Breaking Arbitrarily")
                    #     break
                    vis_wire_cropped_np = cropped_np(this_wire_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                    fullimage_tgraph.SetPoint(n_steps,crop_center_x+0.5,crop_center_y+0.5)

                    # Remove Charge up Charge From Track
                    if previous_center_x != -1:
                        removeChargeOnTrackSegment(trackRemoved_wire_im, trackRemoved_larmatch_im, mask_im, int(crop_center_x), int(crop_center_y), int(previous_center_x), int(previous_center_y))

                    vis_wirecrops_v.append(vis_wire_cropped_np)
                    step_images_in = prepare_sequence_steps(all_steps).to(torch.device(PARAMS['DEVICE']))
                    next_steps_pred_scores, endpoint_scores, hidden_out, cell_out = model(step_images_in)
                    hidden_in = hidden_out
                    cell_in   = cell_out
                    # End Fwd
                    npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                    np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)
                    endpoint_score = np.exp(endpoint_scores.cpu().detach().numpy())[-1]
                    print(endpoint_score)
                    endpoint_scores_v.append(endpoint_score[1])
                    is_endpoint_score = (endpoint_score[1] >= 0.5)
                    # optimizer.step()
                    print(np_pred.shape)
                    if n_steps == 0 and repeat ==0:
                        # save_im(np_pred[0],images_dir+'TrackStartScores'+str(entries_v[deploy_idx]))
                        np_pred_vtx = np_pred[0]
                    if n_steps == 0:
                        np_prob_alter = cropped_np(mask_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                        np_pred[0] = np_pred[0]*np_prob_alter
                        np_pred_vtx = np_pred[0]
                        # save_im(np_pred[0],images_dir+'TrackStartScores_'+str(entries_v[deploy_idx])+"_"+str(repeat))
                        np_pred_sum = np.zeros((np_pred.shape[1],np_pred.shape[2]))
                        for xx in range(np_pred_sum.shape[0]):
                            for yy in range(np_pred_sum.shape[1]):
                                xmin = xx-1 if xx-1 >= 0 else 0
                                xmax = xx+2 if xx+2 <  np_pred.shape[1] else np_pred.shape[1]
                                ymin = yy-1 if yy-1 >= 0 else 0
                                ymax = yy+2 if yy+2 <  np_pred.shape[2] else np_pred.shape[2]
                                np_pred_sum[xx,yy] = np.sum(np_pred[0,xmin:xmax,ymin:ymax])
                        # save_im(np_pred_sum,images_dir+'/TrackStartScoresSum_'+str(entries_v[deploy_idx])+"_"+str(repeat))
                        if np.max(np_pred_vtx) < 0.1:
                            is_endpoint_score = True
                            save_trackbranch = False
                        else:
                            nRecoTracks += 1

                    np_idx_v = make_prediction_vector(PARAMS, np_pred)
                    pred_x, pred_y = unflatten_pos(np_idx_v[-1], PARAMS['PADDING']*2+1)
                    vispred_v.append(np_idx_v[-1])

                    # Get Next Step Stuff
                    current_dx = crop_center_x - previous_center_x
                    current_dy = crop_center_y - previous_center_y
                    previous_center_x = crop_center_x
                    previous_center_y = crop_center_y

                    crop_center_x = crop_center_x - PARAMS['PADDING'] + pred_x
                    crop_center_y = crop_center_y - PARAMS['PADDING'] + pred_y
                    larmatch_im_crop = cropped_np(this_larmatch_im, crop_center_x, crop_center_y, PARAMS['PADDING'])

                    # for c in range(larmatch_im_crop.shape[2]):
                    #     print("\n\nFeature ", c,"\n")
                    #     for i in range(vis_wire_cropped_np.shape[0]):
                    #         ydim = vis_wire_cropped_np.shape[1]
                    #         for j in range(vis_wire_cropped_np.shape[1]):
                    #             print(str(vis_wire_cropped_np[i][ydim-1-j]).zfill(3)," ",end="")
                    #         print()

                    net_feats_in = larmatch_im_crop


                    if PARAMS['APPEND_WIREIM']:
                        wire_feat_cropped_np = cropped_np(this_wire_im, crop_center_x, crop_center_y, PARAMS['PADDING'])
                        exp_wire_images = np.expand_dims(wire_feat_cropped_np,axis=2)
                        net_feats_in = np.concatenate((net_feats_in,exp_wire_images),axis=2)


                    this_step = [net_feats_in]
                    all_steps.append(net_feats_in)
                    if is_endpoint_score == True:
                        removeTrackWidth_v2(trackRemoved_wire_im, trackRemoved_larmatch_im, mask_im, int(crop_center_x), int(crop_center_y), int(current_dx), int(current_dy))
                        print("Endpoint Net Says Stop Here!")
                    if n_steps == PARAMS['MAX_STEPS']:
                        is_endpoint_score = True
                        removeTrackWidth_v2(trackRemoved_wire_im, trackRemoved_larmatch_im, mask_im, int(crop_center_x), int(crop_center_y), int(current_dx), int(current_dy))
                        print("Catch on Making too Many Steps says end here!")
                    n_steps += 1





                end_time = time.time()
                print(end_time - start_time, "Seconds Passed to deploy After Preprocessing Starts")
                if not save_trackbranch:
                    print("\n\nThis track suggestion didn't have high enough start prob, not saving.\n\n")
                    continue



                # Save Ev Displays
                # thistrack_dir=images_dir+'Deploy_'+str(entries_v[deploy_idx])+"_"+str(mctrack_idx_v[deploy_idx])+"_"+str(repeat)+"_"+str(runs_v[deploy_idx])+"_"+str(subruns_v[deploy_idx])+"_"+str(event_ids[deploy_idx])+"/"
                # if not os.path.exists(thistrack_dir):
                #     os.mkdir(thistrack_dir)
                # make_steps_images(np.stack(vis_wirecrops_v,axis=0),thistrack_dir+'Dep_ImStep',PARAMS['PADDING']*2+1,pred=vispred_v,targ=None,endpoint_scores=endpoint_scores_v)
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
            mcProngs_h.Fill(n_mcProngs[deploy_idx])
            mcProngs_thresh_h.Fill(n_mcProngs_thresh[deploy_idx])
            recoProngs_h.Fill(nRecoTracks)
            reco_m_mcProngs_h.Fill(nRecoTracks - n_mcProngs[deploy_idx])
            reco_m_mcProngs_thresh_h.Fill(nRecoTracks - n_mcProngs_thresh[deploy_idx])
    dir = 'prongPlots/'
    tmpcan = ROOT.TCanvas('canv','canv',1200,1000)
    mcProngs_h.Draw()
    tmpcan.SaveAs(dir+"nue_mcProngs.png")
    mcProngs_thresh_h.Draw()
    tmpcan.SaveAs(dir+"nue_mcProngs_thresh.png")
    recoProngs_h.Draw()
    tmpcan.SaveAs(dir+"nue_recoProngs.png")
    reco_m_mcProngs_h.Draw()
    tmpcan.SaveAs(dir+"nue_reco_minus_mcProngs.png")
    reco_m_mcProngs_thresh_h.Draw()
    tmpcan.SaveAs(dir+"nue_reco_minus_mcProngs_thresh.png")

    print()
    print("End of Deploy")
    print()



    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
