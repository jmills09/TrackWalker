import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats, make_log_stat_dict, reravel_array
from DataLoader import get_net_inputs_mc
from ModelFunctions import LSTMTagger, run_validation_pass
import random

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
PARAMS['HIDDEN_DIM'] =1024
PARAMS['PADDING'] =2
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) # N_Features
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
PARAMS['TRAINING_SIZE'] = -1 # Deprecated
PARAMS['DO_TENSORLOG'] = False
PARAMS['TENSORDIR']  = None # Default runs/DATE_TIME
PARAMS['SAVE_MODEL'] = False #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_EPOCHS'] =10000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 5
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 1
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 1
PARAMS['DEVICE'] = 'cuda:3'
PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
# PARAMS['AREA_TARGET'] = True   # Change network to be predicting 

PARAMS['USE_CONV_IM'] = False
PARAMS['LARMATCH_CKPT'] = '/home/jmills/workdir/TrackWalker/larmatch_ckpt/checkpoint.1974000th.tar'
PARAMS['MASK_WC'] = False

def main():
    print("Let's Get Started.")
    torch.manual_seed(1)
    START_ENTRY = 0
    END_ENTRY   = 1
    START_ENTRY_VAL = 0
    END_ENTRY_VAL   =1
    training_data, full_image, steps_x, steps_y, event_ids, rse_pdg_dict =  get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY)
    print("Number of Training Examples:", len(training_data))
    validation_data, val_full_image, val_steps_x, val_steps_y, val_event_ids, val_rse_pdg_dict =  get_net_inputs_mc(PARAMS, START_ENTRY_VAL, END_ENTRY_VAL)
    print("Number of Validation Examples:", len(validation_data))
    # This will rip only a few tracks from the loaded train and val sets
    training_data=training_data[0:1]
    validation_data=validation_data[0:1]
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
    print("Length Training   Set:", len(training_data))
    print("Length Validation Set:", len(validation_data))

    output_dim = None
    loss_function = None
    if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
        output_dim = PARAMS['NUM_CLASSES'] # nPixels in crop + 1 for 'end of track'
        loss_function = nn.NLLLoss(reduction='none')
    else:
        output_dim = PARAMS['NDIMENSIONS'] # Shift X, Shift Y
        loss_function = nn.MSELoss(reduction='sum')

    model = LSTMTagger(PARAMS['EMBEDDING_DIM'], PARAMS['HIDDEN_DIM'], output_dim).to(torch.device(PARAMS['DEVICE']))
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['LEARNING_RATE'])


    step_counter = 0
    for epoch in range(PARAMS['EPOCHS']):  # again, normally you would NOT do 300 epochs, it is toy data
        print("Epoch:",epoch)
        train_idx = -1
        # To Log Stats every N epoch
        log_stats_dict_epoch_train = make_log_stat_dict('epoch_train_')
        log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
        # To Log Stats Every N Tracks Looked At
        log_stats_dict_step_train = make_log_stat_dict('step_train_')
        log_stats_dict_step_val = make_log_stat_dict('step_val_')

        for step_images, next_steps in training_data:
            model.train()
            step_counter += 1
            train_idx += 1
            # Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            step_images_in = prepare_sequence_steps(step_images).to(torch.device(PARAMS['DEVICE']))
            is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']
            targets = prepare_sequence_steps(next_steps,long=is_long).to(torch.device(PARAMS['DEVICE']))
            np_targ = targets.cpu().detach().numpy()

            # Step 3. Run our forward pass.
            next_steps_pred_scores, endpoint_scores = model(step_images_in)
            np_pred = None
            if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                np_pred = np.argmax(next_steps_pred_scores.cpu().detach().numpy(),axis=1)
            else:
                np_pred = np.rint(next_steps_pred_scores.cpu().detach().numpy()) # Rounded to integers



            loss_weights = torch.tensor(get_loss_weights_v2(targets.cpu().detach().numpy(),np_pred,PARAMS),dtype=torch.float).to(torch.device(PARAMS['DEVICE']))
            loss = loss_function(next_steps_pred_scores, targets)
            loss_weighted = loss*loss_weights
            loss_total = torch.mean(loss_weighted)
            loss_total.backward()
            optimizer.step()
            print("Info:")
            print(reravel_array(next_steps_pred_scores.cpu().detach().numpy()[0],PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1))
            print(targets.cpu().detach().numpy()[0])
            print(loss.cpu().detach().numpy()[0])
            print(loss_weights.cpu().detach().numpy()[0])
            print(loss_weighted.cpu().detach().numpy()[0])
            print()
            if PARAMS['DO_TENSORLOG']:
                calc_logger_stats(log_stats_dict_epoch_train, PARAMS, np_pred, np_targ, loss_total, len(training_data), is_train=True,is_epoch=True)
                if PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']!=-1:
                    calc_logger_stats(log_stats_dict_step_train, PARAMS, np_pred, np_targ, loss_total, PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'], is_train=True,is_epoch=False)
                if step_counter%PARAMS['TRAIN_TRACKIDX_LOGINTERVAL']== 0:
                    writer.add_scalar('Step/train_loss', log_stats_dict_step_train['step_train_loss_average'], step_counter)
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
                    log_stats_dict_step_val = make_log_stat_dict('step_val_')
                    run_validation_pass(PARAMS, model, validation_data, loss_function, writer, log_stats_dict_step_val, step_counter, is_epoch=False)

        ####### DO VALIDATION PASS
        if PARAMS['DO_TENSORLOG'] and epoch%PARAMS['VALIDATION_EPOCH_LOGINTERVAL']==0:
            print("Logging Val Epoch", epoch)
            log_stats_dict_epoch_val = make_log_stat_dict('epoch_val_')
            run_validation_pass(PARAMS, model, validation_data, loss_function, writer, log_stats_dict_epoch_val, epoch, is_epoch=True)


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
            writer.add_scalar('Epoch/train_loss', log_stats_dict_epoch_train['epoch_train_loss_average'], epoch)
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
