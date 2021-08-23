import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint, get_writers
from DataLoader import get_net_inputs_mc, DataLoader_MC
from ReformattedDataLoader import ReformattedDataLoader_MC
from ModelFunctions import LSTMTagger, GRUTagger, run_validation_pass
import random
import ROOT
import os


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

PARAMS['RAND_FLIP_INPUT'] = False
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
# PARAMS['INFILE'] ="/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
# PARAMS['INFILE'] ="/home/jmills/workdir/TrackWalker/inputfiles/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtmerge_TrackWalker_traindata_198files.root"
# PARAMS['INFILE'] = "/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/Reformat_LArMatch_Pad_010.root"
PARAMS['INFILE_TRAIN'] = "/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/hadd_train_pass.root"
PARAMS['INFILE_VAL']   = "/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/hadd_val_pass.root"

# PARAMS['TRACK_IDX'] =0
# PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
PARAMS['NDIMENSIONS'] = 2 #Not configured to have 3 yet.

PARAMS['DO_TENSORLOG'] = True
PARAMS['TENSORDIR']  = None # Default runs/DATE_TIME Deprecated
PARAMS['TWOWRITERS'] = True

PARAMS['SAVE_MODEL'] = True #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_TRACKS'] = 250000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 5
PARAMS['STOP_AFTER_NTRACKS'] = 999999999999999999999
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 1000
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 1000
PARAMS['DEVICE'] = 'cuda:2'

PARAMS['LOAD_SIZE']  = 100 #Number of Entries to Load training tracks from
PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
PARAMS['VAL_SAMPLE_SIZE'] = 500

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

PARAMS['LEARNING_RATES'] = [(0, 0.001), (10000, 0.0001), (100000, 1e-05), (1000000, 1e-06)] #(Step, New LR), place steps in order.
PARAMS['NEXTSTEP_LOSS_WEIGHT'] = 20.0
PARAMS['ENDSTEP_LOSS_WEIGHT']  = 1.0
PARAMS['LOAD_PREVIOUS_CHECKPOINT'] = ''



def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    nbins = PARAMS['PADDING']*2+1
    pred_h = ROOT.TH2D("Prediction Steps Heatmap","Prediction Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)
    targ_h = ROOT.TH2D("Target Steps Heatmap","Target Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)

    ReformattedDataLoader_Train = ReformattedDataLoader_MC(PARAMS,all_train=True)
    ReformattedDataLoader_Val   = ReformattedDataLoader_MC(PARAMS,all_valid=True)

    PARAMS['TRAIN_EPOCH_SIZE'] = ReformattedDataLoader_Train.nentries_train #500 # Number of Training Tracks to use (load )
    PARAMS['VAL_EPOCH_SIZE']   = ReformattedDataLoader_Val.nentries_val #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])

    writer_train = None
    writer_val   = None
    writer_dir   = None
    if PARAMS['DO_TENSORLOG']:
        writer_train, writer_val, writer_dir = get_writers(PARAMS)

    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")

    output_dim = None
    loss_function_next_step = None
    loss_function_endpoint = None
    if PARAMS['AREA_TARGET']:
        output_dim = PARAMS['NUM_CLASSES']
        loss_function_next_step = nn.MSELoss(reduction='none')
        loss_function_endpoint = nn.NLLLoss(reduction='none')
    elif PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
        output_dim = PARAMS['NUM_CLASSES'] # nPixels in crop + 1 for 'end of track'
        loss_function_next_step = nn.NLLLoss(reduction='none')
        loss_function_endpoint = nn.NLLLoss(reduction='none')
    else:
        output_dim = PARAMS['NDIMENSIONS'] # Shift X, Shift Y
        loss_function_next_step = nn.MSELoss(reduction='sum')
        loss_function_endpoint = nn.NLLLoss(reduction='none')


    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))
    # model = GRUTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))
    if PARAMS['LOAD_PREVIOUS_CHECKPOINT'] != '':
        if (PARAMS['DEVICE'] != 'cpu'):
            model.load_state_dict(torch.load(PARAMS['LOAD_PREVIOUS_CHECKPOINT'], map_location={'cpu':PARAMS['DEVICE'],'cuda:0':PARAMS['DEVICE'],'cuda:1':PARAMS['DEVICE'],'cuda:2':PARAMS['DEVICE'],'cuda:3':PARAMS['DEVICE']}))
        else:
            model.load_state_dict(torch.load(PARAMS['LOAD_PREVIOUS_CHECKPOINT'], map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))

    CURRENT_LR_IDX = 0
    NEXT_LR_TUPLE  = PARAMS['LEARNING_RATES'][CURRENT_LR_IDX]
    PARAMS['CURRENT_LR'] = NEXT_LR_TUPLE[1]

    optimizer = optim.Adam(model.parameters(), lr=NEXT_LR_TUPLE[1],weight_decay=1e-04)
    if len(PARAMS['LEARNING_RATES']) > CURRENT_LR_IDX+1:
        CURRENT_LR_IDX += 1
        NEXT_LR_TUPLE  = PARAMS['LEARNING_RATES'][CURRENT_LR_IDX]

    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']


    step_counter = 0
    for epoch in range(PARAMS['EPOCHS']):  # again, normally you would NOT do 300 epochs, it is toy data
        print("\n-----------------------------------\nEpoch:",epoch,"\n")
        train_idx = -1

        if not PARAMS['TWOWRITERS']:
            # To Log Stats every N epoch
            log_stats_dict_epoch_train = make_log_stat_dict('epoch_train_')
            log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
            # To Log Stats Every N Tracks Looked At
            log_stats_dict_step_train = make_log_stat_dict('step_train_')
            log_stats_dict_step_val = make_log_stat_dict('step_val_')
        else:
            log_stats_dict_epoch_train = make_log_stat_dict('epoch_')
            log_stats_dict_epoch_val = make_log_stat_dict('epoch_')
            # To Log Stats Every N Tracks Looked At
            log_stats_dict_step_train = make_log_stat_dict('step_')
            log_stats_dict_step_val = make_log_stat_dict('step_')
        number_train_loaded_so_far = 0
        n_to_load = PARAMS['LOAD_SIZE']
        while number_train_loaded_so_far < PARAMS['TRAIN_EPOCH_SIZE']:
            if (PARAMS['TRAIN_EPOCH_SIZE'] - number_train_loaded_so_far) < n_to_load:
                n_to_load = PARAMS['TRAIN_EPOCH_SIZE'] - number_train_loaded_so_far
            print()
            training_data = ReformattedDataLoader_Train.get_train_data(n_to_load)
            number_train_loaded_so_far += len(training_data)
            print(number_train_loaded_so_far, "Tracks loaded total this epoch.")

            for step_images, targ_next_step_idx, targ_area_next_step in training_data:
                model.train()
                # Change LR
                if step_counter == NEXT_LR_TUPLE[0]:
                    print("\n\nUpdating Learning Rate:")
                    print("Old LR:",PARAMS['CURRENT_LR'])
                    print("New LR:",NEXT_LR_TUPLE[1],"\n\n")
                    for pgroup in optimizer.param_groups:
                        pgroup['lr'] = NEXT_LR_TUPLE[1]
                    PARAMS['CURRENT_LR'] = NEXT_LR_TUPLE[1]
                    if len(PARAMS['LEARNING_RATES']) > CURRENT_LR_IDX+1:
                        CURRENT_LR_IDX += 1
                        NEXT_LR_TUPLE  = PARAMS['LEARNING_RATES'][CURRENT_LR_IDX]

                step_counter += 1
                train_idx += 1
                # Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()
                step_images_in = prepare_sequence_steps(step_images).to(torch.device(PARAMS['DEVICE']))
                n_steps = step_images_in.shape[0]
                np_targ_endpt = np.zeros((n_steps))
                np_targ_endpt[n_steps-1] = 1
                endpoint_targ_t = torch.tensor(np_targ_endpt).to(torch.device(PARAMS['DEVICE']),dtype=torch.long)
                targets_next_step_area  = None
                targets_onept = prepare_sequence_steps(targ_next_step_idx,long=is_long)
                if PARAMS['AREA_TARGET']:
                    targets_next_step_area = prepare_sequence_steps(targ_area_next_step,long=is_long).to(torch.device(PARAMS['DEVICE']))
                else:
                    targets_next_step_area = targets_onept.to(torch.device(PARAMS['DEVICE']))
                # Step 3. Run our forward pass.
                next_steps_pred_scores, endpoint_scores, hidden_n, cell_n = model(step_images_in) # _ is hidden state, no need to hold onto

                np_pred = None
                np_targ = None
                np_pred_endpt = None
                if PARAMS['AREA_TARGET']:
                    npts = next_steps_pred_scores.cpu().detach().numpy().shape[0]
                    np_pred = next_steps_pred_scores.cpu().detach().numpy().reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)
                    np_targ = targets_onept.cpu().detach().numpy()
                    np_pred_endpt = np.argmax(endpoint_scores.cpu().detach().numpy(),axis=1)
                elif PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    np_pred = np.argmax(next_steps_pred_scores.cpu().detach().numpy(),axis=1)
                    np_targ = targets_onept.cpu().detach().numpy()
                else:
                    np_pred = np.rint(next_steps_pred_scores.cpu().detach().numpy()) # Rounded to integers
                    np_targ = targets_onept.cpu().detach().numpy()

                # loss_weights = torch.tensor(get_loss_weights_v2(targets_next_step_area.cpu().detach().numpy(),np_pred,PARAMS),dtype=torch.float).to(torch.device(PARAMS['DEVICE']))

                loss_next_steps = loss_function_next_step(next_steps_pred_scores, targets_next_step_area)
                vals_per_step = loss_next_steps.shape[1]
                loss_next_steps_per_step =torch.mean(torch.div(torch.sum(loss_next_steps, dim=1),vals_per_step))*PARAMS['NEXTSTEP_LOSS_WEIGHT']
                loss_endpoint   = torch.mean(loss_function_endpoint(endpoint_scores, endpoint_targ_t))*PARAMS['ENDSTEP_LOSS_WEIGHT']
                loss_total = loss_next_steps_per_step + loss_endpoint
                loss_total.backward()
                optimizer.step()
                np_idx_v = make_prediction_vector(PARAMS, np_pred)
                for ixx in range(np_idx_v.shape[0]):
                    pred_x, pred_y = unflatten_pos(np_idx_v[ixx], PARAMS['PADDING']*2+1)
                    pred_h.Fill(pred_x,pred_y)
                for ixx in range(np_targ.shape[0]):
                    targ_x, targ_y = unflatten_pos(np_targ[ixx], PARAMS['PADDING']*2+1)
                    targ_h.Fill(targ_x,targ_y)
                if PARAMS['SAVE_MODEL'] and step_counter%PARAMS['CHECKPOINT_EVERY_N_TRACKS'] == 0:
                    print("CANT SAVE NEED TO SPECIFY SUBFOLDER")
                    torch.save(model.state_dict(), writer_dir+"TrackerCheckPoint_"+str(epoch)+"_"+str(step_counter)+".pt")
                if PARAMS['DO_TENSORLOG']:
                    calc_logger_stats(log_stats_dict_epoch_train, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['TRAIN_EPOCH_SIZE'], np_pred_endpt, np_targ_endpt, is_train=True,is_epoch=True)
                    if PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']!=-1:
                        calc_logger_stats(log_stats_dict_step_train, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'], np_pred_endpt, np_targ_endpt, is_train=True,is_epoch=False)
                    if step_counter%PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']== 0:
                        print("Logging Train Step",step_counter)
                        if not PARAMS['TWOWRITERS']:
                            writer_train.add_scalar('Step/train_loss_total', log_stats_dict_step_train['step_train_loss_average'], step_counter)
                            writer_train.add_scalar('Step/train_loss_endpointnet', log_stats_dict_step_train['step_train_loss_endptnet'], step_counter)
                            writer_train.add_scalar('Step/train_loss_stepnet', log_stats_dict_step_train['step_train_loss_stepnet'], step_counter)
                            writer_train.add_scalar('Step/train_acc_endpoint', log_stats_dict_step_train['step_train_acc_endpoint'], step_counter)
                            writer_train.add_scalar('Step/train_acc_exact', log_stats_dict_step_train['step_train_acc_exact'], step_counter)
                            writer_train.add_scalar('Step/train_acc_2dist', log_stats_dict_step_train['step_train_acc_2dist'], step_counter)
                            writer_train.add_scalar('Step/train_acc_5dist', log_stats_dict_step_train['step_train_acc_5dist'], step_counter)
                            writer_train.add_scalar('Step/train_acc_10dist', log_stats_dict_step_train['step_train_acc_10dist'], step_counter)
                            writer_train.add_scalar('Step/train_num_correct_exact', log_stats_dict_step_train['step_train_num_correct_exact'], step_counter)
                            writer_train.add_scalar("Step/train_average_off_distance", log_stats_dict_step_train['step_train_average_distance_off'],step_counter)
                            writer_train.add_scalar("Step/train_frac_misIDas_endpoint", log_stats_dict_step_train['step_train_frac_misIDas_endpoint'],step_counter)
                            writer_train.add_scalar("Step/train_lr", PARAMS['CURRENT_LR'],step_counter)
                            log_stats_dict_step_train = make_log_stat_dict('step_train')
                        else:
                            writer_train.add_scalar('Step/loss_total', log_stats_dict_step_train['step_loss_average'], step_counter)
                            writer_train.add_scalar('Step/loss_endpointnet', log_stats_dict_step_train['step_loss_endptnet'], step_counter)
                            writer_train.add_scalar('Step/loss_stepnet', log_stats_dict_step_train['step_loss_stepnet'], step_counter)
                            writer_train.add_scalar('Step/acc_endpoint', log_stats_dict_step_train['step_acc_endpoint'], step_counter)
                            writer_train.add_scalar('Step/acc_exact', log_stats_dict_step_train['step_acc_exact'], step_counter)
                            writer_train.add_scalar('Step/acc_2dist', log_stats_dict_step_train['step_acc_2dist'], step_counter)
                            writer_train.add_scalar('Step/acc_5dist', log_stats_dict_step_train['step_acc_5dist'], step_counter)
                            writer_train.add_scalar('Step/acc_10dist', log_stats_dict_step_train['step_acc_10dist'], step_counter)
                            writer_train.add_scalar('Step/num_correct_exact', log_stats_dict_step_train['step_num_correct_exact'], step_counter)
                            writer_train.add_scalar("Step/average_off_distance", log_stats_dict_step_train['step_average_distance_off'],step_counter)
                            writer_train.add_scalar("Step/frac_misIDas_endpoint", log_stats_dict_step_train['step_frac_misIDas_endpoint'],step_counter)
                            writer_train.add_scalar("Step/lr", PARAMS['CURRENT_LR'],step_counter)
                            log_stats_dict_step_train = make_log_stat_dict('step_')


                    if PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] !=-1 and step_counter%PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] == 0:
                        print("Logging Val Step",step_counter)
                        if not PARAMS['TWOWRITERS']:
                            log_stats_dict_step_val = make_log_stat_dict('step_val_')
                        else:
                            log_stats_dict_step_val = make_log_stat_dict('step_')
                        run_validation_pass(PARAMS, model, ReformattedDataLoader_Val, loss_function_next_step, loss_function_endpoint, writer_val, log_stats_dict_step_val, step_counter, is_epoch=False)
                if step_counter%PARAMS["STOP_AFTER_NTRACKS"] == 0:
                    break

        ####### DO VALIDATION PASS
        if PARAMS['DO_TENSORLOG'] and epoch%PARAMS['VALIDATION_EPOCH_LOGINTERVAL']==0:
            print("Logging Val Epoch", epoch)
            if not PARAMS['TWOWRITERS']:
                log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
            else:
                log_stats_dict_epoch_val = make_log_stat_dict('epoch_')
            run_validation_pass(PARAMS, model, ReformattedDataLoader_Val, loss_function_next_step, loss_function_endpoint, writer_val, log_stats_dict_epoch_val, epoch, is_epoch=True)


        # if epoch%50 ==0:
        #     print("Training Epoch Averaged")
        #     print("Exact Accuracy Endpoints:")
        #     print(log_stats_dict_epoch_train['epoch_train_acc_endpoint'])
        #     print("Fraction misID as Endpoints:")
        #     print(log_stats_dict_epoch_train['epoch_train_frac_misIDas_endpoint'])
        #     print("Exact Accuracy Trackpoints:")
        #     print(log_stats_dict_epoch_train['epoch_train_acc_exact'])
        #     print("Within 2 Accuracy:")
        #     print(log_stats_dict_epoch_train['epoch_train_acc_2dist'])
        #     print("Within 5 Accuracy:")
        #     print(log_stats_dict_epoch_train['epoch_train_acc_5dist'])
        #     print("Within 10 Accuracy:")
        #     print(log_stats_dict_epoch_train['epoch_train_acc_10dist'])
        #     print("Average Distance Off:")
        #     print(log_stats_dict_epoch_train['epoch_train_average_distance_off'])
        #     print("Loss:")
        #     print(log_stats_dict_epoch_train['epoch_train_loss_average'])
        #     print("/////////////////////////////")
        #     print()
        if PARAMS['DO_TENSORLOG'] and epoch%PARAMS['TRAIN_EPOCH_LOGINTERVAL']==0:
            print("Logging Train Epoch", epoch)
            if not PARAMS['TWOWRITERS']:
                writer_train.add_scalar('Epoch/train_loss_total', log_stats_dict_epoch_train['epoch_train_loss_average'], epoch)
                writer_train.add_scalar('Epoch/train_loss_endpointnet', log_stats_dict_epoch_train['epoch_train_loss_endptnet'], epoch)
                writer_train.add_scalar('Epoch/train_loss_stepnet', log_stats_dict_epoch_train['epoch_train_loss_stepnet'], epoch)
                writer_train.add_scalar('Epoch/train_acc_endpoint', log_stats_dict_epoch_train['epoch_train_acc_endpoint'], epoch)
                writer_train.add_scalar('Epoch/train_acc_exact', log_stats_dict_epoch_train['epoch_train_acc_exact'], epoch)
                writer_train.add_scalar('Epoch/train_acc_2dist', log_stats_dict_epoch_train['epoch_train_acc_2dist'], epoch)
                writer_train.add_scalar('Epoch/train_acc_5dist', log_stats_dict_epoch_train['epoch_train_acc_5dist'], epoch)
                writer_train.add_scalar('Epoch/train_acc_10dist', log_stats_dict_epoch_train['epoch_train_acc_10dist'], epoch)
                writer_train.add_scalar('Epoch/train_num_correct_exact', log_stats_dict_epoch_train['epoch_train_num_correct_exact'], epoch)
                writer_train.add_scalar("Epoch/train_average_off_distance", log_stats_dict_epoch_train['epoch_train_average_distance_off'],epoch)
                writer_train.add_scalar("Epoch/train_frac_misIDas_endpoint", log_stats_dict_epoch_train['epoch_train_frac_misIDas_endpoint'],epoch)
                writer_train.add_scalar("Epoch/train_lr", PARAMS['CURRENT_LR'],epoch)

            else:
                writer_train.add_scalar('Epoch/loss_total', log_stats_dict_epoch_train['epoch_loss_average'], epoch)
                writer_train.add_scalar('Epoch/loss_endpointnet', log_stats_dict_epoch_train['epoch_loss_endptnet'], epoch)
                writer_train.add_scalar('Epoch/loss_stepnet', log_stats_dict_epoch_train['epoch_loss_stepnet'], epoch)
                writer_train.add_scalar('Epoch/acc_endpoint', log_stats_dict_epoch_train['epoch_acc_endpoint'], epoch)
                writer_train.add_scalar('Epoch/acc_exact', log_stats_dict_epoch_train['epoch_acc_exact'], epoch)
                writer_train.add_scalar('Epoch/acc_2dist', log_stats_dict_epoch_train['epoch_acc_2dist'], epoch)
                writer_train.add_scalar('Epoch/acc_5dist', log_stats_dict_epoch_train['epoch_acc_5dist'], epoch)
                writer_train.add_scalar('Epoch/acc_10dist', log_stats_dict_epoch_train['epoch_acc_10dist'], epoch)
                writer_train.add_scalar('Epoch/num_correct_exact', log_stats_dict_epoch_train['epoch_num_correct_exact'], epoch)
                writer_train.add_scalar("Epoch/average_off_distance", log_stats_dict_epoch_train['epoch_average_distance_off'],epoch)
                writer_train.add_scalar("Epoch/frac_misIDas_endpoint", log_stats_dict_epoch_train['epoch_frac_misIDas_endpoint'],epoch)
                writer_train.add_scalar("Epoch/lr", PARAMS['CURRENT_LR'],epoch)

        if step_counter%PARAMS["STOP_AFTER_NTRACKS"] == 0:
            break


    print()
    print("End of Training")
    print()
    if PARAMS['DO_TENSORLOG']:
        writer_train.close()
        writer_val.close()
    # See what the scores are after training
    if PARAMS['SAVE_MODEL']:
        print("CANT SAVE NEED TO SPECIFY SUBFOLDER")
        torch.save(model.state_dict(), writer_dir+"TrackerCheckPoint_"+str(PARAMS['EPOCHS'])+"_Fin.pt")
    # with torch.no_grad():
    #     train_idx = -1
    #     for step_images, next_steps in training_data:
    #         train_idx += 1
    #         print()
    #         print("Event:", event_ids[train_idx])
    #         print("Track Idx:",train_idx)
    #         step_images_in = prepare_sequence_steps(step_images).to(torch.device(PARAMS['DEVICE']))
    #         targets = prepare_sequence_steps(next_steps,long=is_long).to(torch.device(PARAMS['DEVICE']))
    #         np_targ = targets.cpu().detach().numpy()
    #         next_steps_pred_scores, endpoint_scores = model(step_images_in)
    #
    #         np_pred = None
    #         if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #             np_pred = np.argmax(next_steps_pred_scores.cpu().detach().numpy(),axis=1)
    #         else:
    #             np_pred = np.rint(next_steps_pred_scores.cpu().detach().numpy()) # Rounded to integers
    #
    #         num_correct_exact = 0
    #         for ix in range(np_pred.shape[0]):
    #             if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #                 if np_pred[ix] == np_targ[ix]:
    #                     num_correct_exact = num_correct_exact + 1
    #             else:
    #                 if np.array_equal(np_pred[ix], np_targ[ix]):
    #                     num_correct_exact += 1
    #         print("Accuracy",float(num_correct_exact)/float(np_pred.shape[0]))
    #         print("Points:",float(np_pred.shape[0]))
    #
    #         np_targ = targets.cpu().detach().numpy()
    #         if not PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #             print("Predictions Raw")
    #             print(next_steps_pred_scores.cpu().detach().numpy())
    #
    #         # make_steps_images(step_images_in.cpu().detach().numpy(),"images/PredStep_Final_"+str(train_idx).zfill(2)+"_",PADDING*2+1,pred=np_pred,targ=np_targ)
    canv = ROOT.TCanvas('canv','canv',1000,800)
    ROOT.gStyle.SetOptStat(0)
    # pred_h.SetMaximum(500.0)
    pred_h.DrawNormalized("COLZ")
    # canv.SaveAs('pred_h.png')
    # targ_h.SetMaximum(500.0)
    targ_h.DrawNormalized("COLZ")
    # canv.SaveAs('targ_h.png')
    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
