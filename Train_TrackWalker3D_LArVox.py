import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats_distshifter
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint, get_writers
# from DataLoader import get_net_inputs_mc, DataLoader_MC
# from ReformattedDataLoader import ReformattedDataLoader_MC
# from ComplexReformattedDataLoader import ComplexReformattedDataLoader_MC
from DataLoader3D import DataLoader3D
from ModelFunctions import LSTMTagger, GRUTagger, run_validation_pass
import random
import ROOT
import os


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_steps(seq,long=False):
    # full_np = np.stack(seq,axis=0)
    if not long:
        return torch.tensor(seq, dtype=torch.float)
    else:
        return torch.tensor(seq, dtype=torch.long)

PARAMS = {}

PARAMS['MASK_WC'] = False
PARAMS['RAND_FLIP_INPUT'] = False
PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =256
PARAMS['PADDING'] = 10
PARAMS['VOXCUBESIDE'] = 3
PARAMS['APPEND_WIREIM'] = True
PARAMS['NDIMENSIONS'] = 3 #Not configured to have 3 yet.
PARAMS['NFEATS'] = 32
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)*PARAMS['NFEATS'] # N_Features
PARAMS['CENTERPOINT_ISEND'] = True
PARAMS['NUM_CLASSES'] = PARAMS['VOXCUBESIDE']**3
PARAMS['TRACKEND_CLASS'] = (PARAMS['NUM_CLASSES']-1)/2

PARAMS['INFILE_TRAIN'] = "TEST3DReformat/0/Reformat_LArVox_ComplexTrackIdx_000_FULL.root"
PARAMS['INFILE_VAL']   = "TEST3DReformat/0/Reformat_LArVox_ComplexTrackIdx_000_FULL.root"

PARAMS['ALWAYS_EDGE'] = True # True points are always placed at the edge of the Padded Box
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =False # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point

PARAMS['DO_TENSORLOG'] = True
PARAMS['TENSORDIR']  = None # Default runs/DATE_TIME Deprecated
PARAMS['TWOWRITERS'] = True

PARAMS['SAVE_MODEL'] = True #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_TRACKS'] = 20000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 200
PARAMS['STOP_AFTER_NTRACKS'] = 999999999999999999999
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1 # Log Val check every every X train epochs
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1 # Log Train Epoch Nums every X epochs
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 1 # Log Validation check every X train tracks
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 1 # Log Train Values every X train tracks
PARAMS['TRAIN_EPOCH_SIZE'] = -1 #500 # Number of Training Tracks to use (load )
PARAMS['VAL_EPOCH_SIZE'] = -1 #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
PARAMS['VAL_SAMPLE_SIZE'] = PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] #Number of val tracks to do every val check

PARAMS['DEVICE'] = 'cuda:3'
PARAMS['LOAD_SIZE']  = 100 #Number of Entries to Load training tracks from at once

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = False   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2
PARAMS['DO_CROPSHIFT'] = False
PARAMS['CROPSHIFT_MAXAMT'] = 1
PARAMS['TARG_STEP_DIST']   = 3

PARAMS['LEARNING_RATES']   = [(000000, 0.00001), (37300, 0.000001)] #(Step, New LR), place steps in order.
PARAMS['WEIGHT_DECAY']     = 1e-04
# PARAMS['LEARNING_RATES'] = [(0, 0.001), (10000, 0.0001), (100000, 1e-05), (1000000, 1e-06)] #(Step, New LR), place steps in order.
PARAMS['NEXTSTEP_LOSS_WEIGHT'] = 1.0
PARAMS['ENDSTEP_LOSS_WEIGHT']  = 1.0
PARAMS['LOAD_PREVIOUS_CHECKPOINT'] = '' #Set Starting Entry manually right below this
PARAMS['STARTING_ENTRY'] = -1



def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    nbins = PARAMS['PADDING']*2+1
    pred_h = ROOT.TH2D("Prediction Steps Heatmap","Prediction Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)
    targ_h = ROOT.TH2D("Target Steps Heatmap","Target Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)

    ComplexReformattedDataLoader_Train = DataLoader3D(PARAMS,all_train=True,LArVoxMode = True)
    ComplexReformattedDataLoader_Val   = DataLoader3D(PARAMS,all_valid=True,LArVoxMode = True)

    PARAMS['TRAIN_EPOCH_SIZE'] = ComplexReformattedDataLoader_Train.nentries_train #500 # Number of Training Tracks to use (load )
    PARAMS['VAL_EPOCH_SIZE']   = ComplexReformattedDataLoader_Val.nentries_val #int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])

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
        loss_function_next_step = nn.MSELoss(reduction='none')
        loss_function_endpoint = nn.NLLLoss(reduction='none')


    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))
    # model = GRUTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))
    step_counter = 0
    startEpoch   = 0
    startEpochEntryOffset = 0
    if PARAMS['LOAD_PREVIOUS_CHECKPOINT'] != '':
        step_counter = PARAMS['STARTING_ENTRY']
        trainEntry = step_counter%ComplexReformattedDataLoader_Train.nentries_train
        valEntry   = step_counter%ComplexReformattedDataLoader_Val.nentries_val
        startEpoch = int(step_counter/ComplexReformattedDataLoader_Train.nentries_train)
        startEpochEntryOffset = step_counter%ComplexReformattedDataLoader_Train.nentries_train
        ComplexReformattedDataLoader_Train.set_current_entry(trainEntry)
        ComplexReformattedDataLoader_Val.set_current_entry(valEntry)
        if (PARAMS['DEVICE'] != 'cpu'):
            model.load_state_dict(torch.load(PARAMS['LOAD_PREVIOUS_CHECKPOINT'], map_location={'cpu':PARAMS['DEVICE'],'cuda:0':PARAMS['DEVICE'],'cuda:1':PARAMS['DEVICE'],'cuda:2':PARAMS['DEVICE'],'cuda:3':PARAMS['DEVICE']}))
        else:
            model.load_state_dict(torch.load(PARAMS['LOAD_PREVIOUS_CHECKPOINT'], map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))


    CURRENT_LR_IDX = 0
    NEXT_LR_TUPLE  = PARAMS['LEARNING_RATES'][CURRENT_LR_IDX]
    PARAMS['CURRENT_LR'] = NEXT_LR_TUPLE[1]

    optimizer = optim.Adam(model.parameters(), lr=NEXT_LR_TUPLE[1],weight_decay=PARAMS['WEIGHT_DECAY'])
    if len(PARAMS['LEARNING_RATES']) > CURRENT_LR_IDX+1:
        CURRENT_LR_IDX += 1
        NEXT_LR_TUPLE  = PARAMS['LEARNING_RATES'][CURRENT_LR_IDX]

    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']

    # Make our trusty loggers
    if not PARAMS['TWOWRITERS']:
        # To Log Stats every N epoch
        log_stats_dict_epoch_train = make_log_stat_dict('epoch_train_',PARAMS)
        log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_',PARAMS)
        # To Log Stats Every N Tracks Looked At
        log_stats_dict_step_train = make_log_stat_dict('step_train_',PARAMS)
        log_stats_dict_step_val = make_log_stat_dict('step_val_',PARAMS)
    else:
        log_stats_dict_epoch_train = make_log_stat_dict('epoch_',PARAMS)
        log_stats_dict_epoch_val = make_log_stat_dict('epoch_',PARAMS)
        # To Log Stats Every N Tracks Looked At
        log_stats_dict_step_train = make_log_stat_dict('step_',PARAMS)
        log_stats_dict_step_val = make_log_stat_dict('step_',PARAMS)
    firstEpoch = True

    for epoch in range(startEpoch, PARAMS['EPOCHS']):  # again, normally you would NOT do 300 epochs, it is toy data
        print("\n-----------------------------------\nEpoch:",epoch,"\n")
        train_idx = -1
        number_train_loaded_so_far = 0
        if firstEpoch:
            number_train_loaded_so_far = startEpochEntryOffset
            firstEpoch = False
        n_to_load = PARAMS['LOAD_SIZE']
        while number_train_loaded_so_far < PARAMS['TRAIN_EPOCH_SIZE']:
            if (PARAMS['TRAIN_EPOCH_SIZE'] - number_train_loaded_so_far) < n_to_load:
                n_to_load = PARAMS['TRAIN_EPOCH_SIZE'] - number_train_loaded_so_far
            print()
            training_data = ComplexReformattedDataLoader_Train.get_train_data(n_to_load)
            number_train_loaded_so_far += len(training_data)
            print(number_train_loaded_so_far, "Tracks loaded total this epoch.")

            for step3d_crops_in_np, xyzShifts_targ_np in training_data:
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
                step3d_crops_in_t = torch.tensor(step3d_crops_in_np,dtype=torch.float).to(torch.device(PARAMS['DEVICE']))
                # Run Forward Pass
                xyzShifts_pred_t, endpoint_pred_scores_t, hidden_n, cell_n = model(step3d_crops_in_t)
                # Create Endpoint Target Truth
                endpoint_targ_np = np.zeros((step3d_crops_in_t.shape[0]))
                endpoint_targ_np[step3d_crops_in_t.shape[0]-1] = 1
                endpoint_targ_t = torch.tensor(endpoint_targ_np,dtype=torch.long).to(torch.device(PARAMS['DEVICE']))
                # Get Endpoint Predictions in np
                endpoint_pred_scores_np = np.argmax(endpoint_pred_scores_t.cpu().detach().numpy(),axis=1)
                # Get Pred XYZ Shift in np, Rounded to ints for OffDist Calc
                xyzShifts_pred_np = np.rint(xyzShifts_pred_t.cpu().detach().numpy())
                xyzShifts_targ_t = torch.tensor(xyzShifts_targ_np).to(torch.device(PARAMS['DEVICE']))

                # Calc loss and backward pass
                loss_next_steps = loss_function_next_step(xyzShifts_pred_t, xyzShifts_targ_t)
                vals_per_step = loss_next_steps.shape[1]
                loss_next_steps_per_step =torch.mean(torch.div(torch.sum(loss_next_steps, dim=1),vals_per_step))*PARAMS['NEXTSTEP_LOSS_WEIGHT']
                loss_endpoint   = torch.mean(loss_function_endpoint(endpoint_pred_scores_t, endpoint_targ_t))*PARAMS['ENDSTEP_LOSS_WEIGHT']
                loss_total = loss_next_steps_per_step + loss_endpoint
                loss_total.backward()
                optimizer.step()


                if PARAMS['SAVE_MODEL'] and step_counter%PARAMS['CHECKPOINT_EVERY_N_TRACKS'] == 0:
                    # print("CANT SAVE NEED TO SPECIFY SUBFOLDER")
                    torch.save(model.state_dict(), writer_dir+"TrackerCheckPoint_"+str(epoch)+"_"+str(step_counter)+".pt")

                if PARAMS['DO_TENSORLOG']:
                    calc_logger_stats_distshifter(log_stats_dict_epoch_train, PARAMS, xyzShifts_pred_np, xyzShifts_targ_np, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['TRAIN_EPOCH_SIZE'], endpoint_pred_scores_np, endpoint_targ_np, is_train=True,is_epoch=True)
                    if PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']!=-1:
                        calc_logger_stats_distshifter(log_stats_dict_step_train, PARAMS, xyzShifts_pred_np, xyzShifts_targ_np, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'], endpoint_pred_scores_np, endpoint_targ_np, is_train=True,is_epoch=False)
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
                            if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                                writer_train.add_scalar("Step/train_average_off_distanceX", log_stats_dict_step_train['step_train_offDistX'],step_counter)
                                writer_train.add_scalar("Step/train_average_off_distanceY", log_stats_dict_step_train['step_train_offDistY'],step_counter)
                                writer_train.add_scalar("Step/train_average_off_distanceZ", log_stats_dict_step_train['step_train_offDistZ'],step_counter)
                                writer_train.add_scalar("Step/train_predStepDist", log_stats_dict_step_train['step_train_predStepDist'],step_counter)
                                writer_train.add_scalar("Step/train_trueStepDist", log_stats_dict_step_train['step_train_trueStepDist'],step_counter)

                            writer_train.add_scalar("Step/train_frac_misIDas_endpoint", log_stats_dict_step_train['step_train_frac_misIDas_endpoint'],step_counter)
                            writer_train.add_scalar("Step/train_lr", PARAMS['CURRENT_LR'],step_counter)
                            log_stats_dict_step_train = make_log_stat_dict('step_train',PARAMS)
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
                            if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                                writer_train.add_scalar("Step/average_off_distanceX", log_stats_dict_step_train['step_offDistX'],step_counter)
                                writer_train.add_scalar("Step/average_off_distanceY", log_stats_dict_step_train['step_offDistY'],step_counter)
                                writer_train.add_scalar("Step/average_off_distanceZ", log_stats_dict_step_train['step_offDistZ'],step_counter)
                                writer_train.add_scalar("Step/predStepDist", log_stats_dict_step_train['step_predStepDist'],step_counter)
                                writer_train.add_scalar("Step/trueStepDist", log_stats_dict_step_train['step_trueStepDist'],step_counter)
                            writer_train.add_scalar("Step/frac_misIDas_endpoint", log_stats_dict_step_train['step_frac_misIDas_endpoint'],step_counter)
                            writer_train.add_scalar("Step/lr", PARAMS['CURRENT_LR'],step_counter)
                            log_stats_dict_step_train = make_log_stat_dict('step_',PARAMS)


                    if PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] !=-1 and step_counter%PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] == 0:
                        print("Logging Val Step",step_counter)
                        if not PARAMS['TWOWRITERS']:
                            log_stats_dict_step_val = make_log_stat_dict('step_val_',PARAMS)
                        else:
                            log_stats_dict_step_val = make_log_stat_dict('step_',PARAMS)
                        run_validation_pass(PARAMS, model, ComplexReformattedDataLoader_Val, loss_function_next_step, loss_function_endpoint, writer_val, log_stats_dict_step_val, step_counter, is_epoch=False)
                if step_counter%PARAMS["STOP_AFTER_NTRACKS"] == 0:
                    break

        ####### DO VALIDATION PASS
        if PARAMS['DO_TENSORLOG'] and epoch%PARAMS['VALIDATION_EPOCH_LOGINTERVAL']==0:
            print("Logging Val Epoch", epoch)
            if not PARAMS['TWOWRITERS']:
                log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_',PARAMS)
            else:
                log_stats_dict_epoch_val = make_log_stat_dict('epoch_',PARAMS)
            run_validation_pass(PARAMS, model, ComplexReformattedDataLoader_Val, loss_function_next_step, loss_function_endpoint, writer_val, log_stats_dict_epoch_val, epoch, is_epoch=True)


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
                if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                    writer_train.add_scalar("Epoch/train_average_off_distanceX", log_stats_dict_epoch_train['epoch_train_offDistX'],epoch)
                    writer_train.add_scalar("Epoch/train_average_off_distanceY", log_stats_dict_epoch_train['epoch_train_offDistY'],epoch)
                    writer_train.add_scalar("Epoch/train_average_off_distanceZ", log_stats_dict_epoch_train['epoch_train_offDistZ'],epoch)
                    writer_train.add_scalar("Epoch/train_predStepDist", log_stats_dict_epoch_train['epoch_train_predStepDist'],epoch)
                    writer_train.add_scalar("Epoch/train_trueStepDist", log_stats_dict_epoch_train['epoch_train_trueStepDist'],epoch)
                writer_train.add_scalar("Epoch/train_frac_misIDas_endpoint", log_stats_dict_epoch_train['epoch_train_frac_misIDas_endpoint'],epoch)
                writer_train.add_scalar("Epoch/train_lr", PARAMS['CURRENT_LR'],epoch)
                log_stats_dict_epoch_train = make_log_stat_dict('epoch_train_',PARAMS)
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
                if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                    writer_train.add_scalar("Epoch/average_off_distanceX", log_stats_dict_epoch_train['epoch_offDistX'],epoch)
                    writer_train.add_scalar("Epoch/average_off_distanceY", log_stats_dict_epoch_train['epoch_offDistY'],epoch)
                    writer_train.add_scalar("Epoch/average_off_distanceZ", log_stats_dict_epoch_train['epoch_offDistZ'],epoch)
                    writer_train.add_scalar("Epoch/predStepDist", log_stats_dict_epoch_train['epoch_predStepDist'],epoch)
                    writer_train.add_scalar("Epoch/trueStepDist", log_stats_dict_epoch_train['epoch_trueStepDist'],epoch)
                writer_train.add_scalar("Epoch/frac_misIDas_endpoint", log_stats_dict_epoch_train['epoch_frac_misIDas_endpoint'],epoch)
                writer_train.add_scalar("Epoch/lr", PARAMS['CURRENT_LR'],epoch)
                log_stats_dict_epoch_train = make_log_stat_dict('epoch_',PARAMS)

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
    #     for step3d_crops_in_np, next_steps in training_data:
    #         train_idx += 1
    #         print()
    #         print("Event:", event_ids[train_idx])
    #         print("Track Idx:",train_idx)
    #         step3d_crops_in_t = prepare_sequence_steps(step3d_crops_in_np).to(torch.device(PARAMS['DEVICE']))
    #         targets = prepare_sequence_steps(next_steps,long=is_long).to(torch.device(PARAMS['DEVICE']))
    #         xyzShifts_targ_np = targets.cpu().detach().numpy()
    #         xyzShifts_pred_t, endpoint_scores = model(step3d_crops_in_t)
    #
    #         xyzShifts_pred_np = None
    #         if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #             xyzShifts_pred_np = np.argmax(xyzShifts_pred_t.cpu().detach().numpy(),axis=1)
    #         else:
    #             xyzShifts_pred_np = np.rint(xyzShifts_pred_t.cpu().detach().numpy()) # Rounded to integers
    #
    #         num_correct_exact = 0
    #         for ix in range(xyzShifts_pred_np.shape[0]):
    #             if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #                 if xyzShifts_pred_np[ix] == xyzShifts_targ_np[ix]:
    #                     num_correct_exact = num_correct_exact + 1
    #             else:
    #                 if np.array_equal(xyzShifts_pred_np[ix], xyzShifts_targ_np[ix]):
    #                     num_correct_exact += 1
    #         print("Accuracy",float(num_correct_exact)/float(xyzShifts_pred_np.shape[0]))
    #         print("Points:",float(xyzShifts_pred_np.shape[0]))
    #
    #         xyzShifts_targ_np = targets.cpu().detach().numpy()
    #         if not PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
    #             print("Predictions Raw")
    #             print(xyzShifts_pred_t.cpu().detach().numpy())
    #
    #         # make_steps_images(step3d_crops_in_t.cpu().detach().numpy(),"images/PredStep_Final_"+str(train_idx).zfill(2)+"_",PADDING*2+1,pred=xyzShifts_pred_np,targ=xyzShifts_targ_np)
    # canv = ROOT.TCanvas('canv','canv',1000,800)
    # ROOT.gStyle.SetOptStat(0)
    # pred_h.SetMaximum(500.0)
    # pred_h.DrawNormalized("COLZ")
    # canv.SaveAs('pred_h.png')
    # targ_h.SetMaximum(500.0)
    # targ_h.DrawNormalized("COLZ")
    # canv.SaveAs('targ_h.png')
    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
