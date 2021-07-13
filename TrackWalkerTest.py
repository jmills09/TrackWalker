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
from DataLoader import get_net_inputs_mc
from ModelFunctions import LSTMTagger, run_validation_pass
import random
import ROOT


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

PARAMS['USE_CONV_IM'] = False
PARAMS['LARMATCH_CKPT'] = '/home/jmills/workdir/TrackWalker/larmatch_ckpt/checkpoint.1974000th.tar'
PARAMS['MASK_WC'] = False

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
PARAMS['EPOCHS'] = 100
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['DEVICE'] = 'cuda:3'

PARAMS['TRAIN_EPOCH_SIZE'] = 100
PARAMS['VAL_EPOCH_SIZE'] = int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])
PARAMS['LOAD_SIZE']  = 10

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2


def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    nbins = PARAMS['PADDING']*2+1
    pred_h = ROOT.TH2D("Prediction Steps Heatmap","Prediction Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)
    targ_h = ROOT.TH2D("Target Steps Heatmap","Target Steps Heatmap",nbins,-0.5,nbins+0.5,nbins,-0.5,nbins+0.5)

    START_ENTRY = 0
    END_ENTRY   = 1000
    START_ENTRY_VAL = 6000
    END_ENTRY_VAL   = 6200 # val is train right now
    training_data, full_image, steps_x, steps_y, event_ids, rse_pdg_dict =  get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY)
    # print("Number of Training Examples:", len(training_data))
    validation_data, val_full_image, val_steps_x, val_steps_y, val_event_ids, val_rse_pdg_dict =  get_net_inputs_mc(PARAMS, START_ENTRY_VAL, END_ENTRY_VAL)
    print("Number of Validation Examples:", len(validation_data))
    # This will rip only a few tracks from the loaded train and val sets
    # training_data=training_data[0:1]
    # validation_data=validation_data[0:1]


    if PARAMS['SHUFFLE_DATASET']:
        all_data = training_data + validation_data
        random.shuffle(all_data)
        pivot = int(0.8*len(all_data))
        training_data=all_data[0:pivot]
        validation_data=all_data[pivot:-1]
    if PARAMS['VAL_IS_TRAIN']:
        validation_data = training_data


    writer = []
    if PARAMS['DO_TENSORLOG']:
        if PARAMS['TENSORDIR'] == None:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(log_dir=PARAMS['TENSORDIR'])
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////")
    print("Initializing Model")
    # print("Length Training   Set:", len(training_data))
    # print("Length Validation Set:", len(validation_data))
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
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])
    is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']


    step_counter = 0
    for epoch in range(PARAMS['EPOCHS']):  # again, normally you would NOT do 300 epochs, it is toy data
        print("\n-----------------------------------\nEpoch:",epoch,"\n")
        train_idx = -1
        # To Log Stats every N epoch
        log_stats_dict_epoch_train = make_log_stat_dict('epoch_train_')
        log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
        # To Log Stats Every N Tracks Looked At
        log_stats_dict_step_train = make_log_stat_dict('step_train_')
        log_stats_dict_step_val = make_log_stat_dict('step_val_')
        number_train_loaded_so_far = 0
        START_ENTRY = 0
        END_ENTRY   = PARAMS['LOAD_SIZE']
        # while number_train_loaded_so_far < PARAMS['TRAIN_EPOCH_SIZE']:
        #     training_data, full_image, steps_x, steps_y, event_ids, rse_pdg_dict =  get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY, PARAMS['TRAIN_EPOCH_SIZE']-number_train_loaded_so_far)
        #     number_train_loaded_so_far += len(training_data)
        #     START_ENTRY = END_ENTRY
        #     END_ENTRY = START_ENTRY+PARAMS['LOAD_SIZE']
        #     print('Loaded From Events',START_ENTRY, 'to', END_ENTRY)
        #     print(len(training_data), "Tracks from this eventlist")
        #     print(number_train_loaded_so_far, "Tracks loaded total this epoch.")
        for step_images, targ_next_step_idx, targ_area_next_step in training_data:
            model.train()
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
            targets_loss  = None
            targets_onept = prepare_sequence_steps(targ_next_step_idx,long=is_long)
            if PARAMS['AREA_TARGET']:
                targets_loss = prepare_sequence_steps(targ_area_next_step,long=is_long).to(torch.device(PARAMS['DEVICE']))
            else:
                targets_loss = targets_onept.to(torch.device(PARAMS['DEVICE']))
            # Step 3. Run our forward pass.

            next_steps_pred_scores, endpoint_scores = model(step_images_in)
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

            # loss_weights = torch.tensor(get_loss_weights_v2(targets_loss.cpu().detach().numpy(),np_pred,PARAMS),dtype=torch.float).to(torch.device(PARAMS['DEVICE']))
            loss_next_steps = loss_function_next_step(next_steps_pred_scores, targets_loss)
            vals_per_step = loss_next_steps.shape[1]
            loss_next_steps_per_step = torch.mean(torch.div(torch.sum(loss_next_steps, dim=1),vals_per_step))
            loss_endpoint   = torch.mean(loss_function_endpoint(endpoint_scores, endpoint_targ_t))*2
            loss_weighted = loss_next_steps_per_step + loss_endpoint#*loss_weights
            loss_total = loss_weighted
            loss_total.backward()
            optimizer.step()
            np_idx_v = make_prediction_vector(PARAMS, np_pred)
            for ixx in range(np_idx_v.shape[0]):
                pred_x, pred_y = unflatten_pos(np_idx_v[ixx], PARAMS['PADDING']*2+1)
                pred_h.Fill(pred_x,pred_y)
            for ixx in range(np_targ.shape[0]):
                targ_x, targ_y = unflatten_pos(np_targ[ixx], PARAMS['PADDING']*2+1)
                targ_h.Fill(targ_x,targ_y)
            if PARAMS['DO_TENSORLOG']:
                calc_logger_stats(log_stats_dict_epoch_train, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['LOAD_SIZE'], np_pred_endpt, np_targ_endpt, is_train=True,is_epoch=True)
                if PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']!=-1:
                    calc_logger_stats(log_stats_dict_step_train, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'], np_pred_endpt, np_targ_endpt, is_train=True,is_epoch=False)
                if step_counter%PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']== 0:
                    print("Logging Train Step",step_counter)
                    writer.add_scalar('Step/train_loss_total', log_stats_dict_step_train['step_train_loss_average'], step_counter)
                    writer.add_scalar('Step/train_loss_endpointnet', log_stats_dict_step_train['step_train_loss_endptnet'], step_counter)
                    writer.add_scalar('Step/train_loss_stepnet', log_stats_dict_step_train['step_train_loss_stepnet'], step_counter)
                    writer.add_scalar('Step/train_acc_endpoint', log_stats_dict_step_train['step_train_acc_endpoint'], step_counter)
                    writer.add_scalar('Step/train_acc_exact', log_stats_dict_step_train['step_train_acc_exact'], step_counter)
                    writer.add_scalar('Step/train_acc_2dist', log_stats_dict_step_train['step_train_acc_2dist'], step_counter)
                    writer.add_scalar('Step/train_acc_5dist', log_stats_dict_step_train['step_train_acc_5dist'], step_counter)
                    writer.add_scalar('Step/train_acc_10dist', log_stats_dict_step_train['step_train_acc_10dist'], step_counter)
                    writer.add_scalar('Step/train_num_correct_exact', log_stats_dict_step_train['step_train_num_correct_exact'], step_counter)
                    writer.add_scalar("Step/train_average_off_distance", log_stats_dict_step_train['step_train_average_distance_off'],step_counter)
                    writer.add_scalar("Step/train_frac_misIDas_endpoint", log_stats_dict_step_train['step_train_frac_misIDas_endpoint'],step_counter)
                    log_stats_dict_step_train = make_log_stat_dict('step_train_')


                if PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] !=-1 and step_counter%PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] == 0:
                    print("Logging Val Step",step_counter)
                    log_stats_dict_step_val = make_log_stat_dict('step_val_')
                    run_validation_pass(PARAMS, model, validation_data, loss_function_next_step, loss_function_endpoint, writer, log_stats_dict_step_val, step_counter, is_epoch=False)


        ####### DO VALIDATION PASS
        if PARAMS['DO_TENSORLOG'] and epoch%PARAMS['VALIDATION_EPOCH_LOGINTERVAL']==0:
            print("Logging Val Epoch", epoch)
            log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
            run_validation_pass(PARAMS, model, validation_data, loss_function_next_step, loss_function_endpoint, writer, log_stats_dict_epoch_val, epoch, is_epoch=True)


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
            writer.add_scalar('Epoch/train_loss_total', log_stats_dict_epoch_train['epoch_train_loss_average'], epoch)
            writer.add_scalar('Epoch/train_loss_endpointnet', log_stats_dict_epoch_train['epoch_train_loss_endptnet'], epoch)
            writer.add_scalar('Epoch/train_loss_stepnet', log_stats_dict_epoch_train['epoch_train_loss_stepnet'], epoch)
            writer.add_scalar('Epoch/train_acc_endpoint', log_stats_dict_epoch_train['epoch_train_acc_endpoint'], epoch)
            writer.add_scalar('Epoch/train_acc_exact', log_stats_dict_epoch_train['epoch_train_acc_exact'], epoch)
            writer.add_scalar('Epoch/train_acc_2dist', log_stats_dict_epoch_train['epoch_train_acc_2dist'], epoch)
            writer.add_scalar('Epoch/train_acc_5dist', log_stats_dict_epoch_train['epoch_train_acc_5dist'], epoch)
            writer.add_scalar('Epoch/train_acc_10dist', log_stats_dict_epoch_train['epoch_train_acc_10dist'], epoch)
            writer.add_scalar('Epoch/train_num_correct_exact', log_stats_dict_epoch_train['epoch_train_num_correct_exact'], epoch)
            writer.add_scalar("Epoch/train_average_off_distance", log_stats_dict_epoch_train['epoch_train_average_distance_off'],epoch)
            writer.add_scalar("Epoch/train_frac_misIDas_endpoint", log_stats_dict_epoch_train['epoch_train_frac_misIDas_endpoint'],epoch)
        if PARAMS['SAVE_MODEL']:
            print("CANT SAVE NEED TO SPECIFY SUBFOLDER")
        # if PARAMS['SAVE_MODEL'] and epoch%PARAMS['CHECKPOINT_EVERY_N_EPOCHS'] == 0:
            # torch.save(model.state_dict(), "model_checkpoints/TrackerCheckPoint_"+str(epoch)+".pt")
    print()
    print("End of Training")
    print()
    if PARAMS['DO_TENSORLOG']:
        writer.close()
    # See what the scores are after training
    if PARAMS['SAVE_MODEL']:
        print("CANT SAVE NEED TO SPECIFY SUBFOLDER")
        torch.save(model.state_dict(), "model_checkpoints/TrackerCheckPoint_"+str(PARAMS['EPOCHS'])+"_Fin.pt")
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
    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
