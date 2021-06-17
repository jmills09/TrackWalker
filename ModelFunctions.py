import ROOT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MiscFunctions import calc_logger_stats, get_loss_weights_v2, reravel_array


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden2endpoint = nn.Linear(hidden_dim, 2) # head to classify step as endpoint

    def forward(self, sequence):
        nsteps = sequence.shape[0]
        sequence_reshaped = sequence.view((nsteps,1,-1))

        lstm_out, _ = self.lstm(sequence_reshaped)
        tag_space = self.hidden2tag(lstm_out.view(sequence.shape[0], -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = (1+torch.tanh(tag_space))/2

        endpoint_space = self.hidden2endpoint(lstm_out.view(sequence.shape[0],-1))
        endpoint_scores = F.log_softmax(endpoint_space, dim=1)
        return tag_scores, endpoint_scores



def Custom_NLLLoss(logs, targets):
    # This is not used.
    out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        out[i] = logs[i][targets[i]]
    return -out.sum()/len(out)

def prepare_sequence_steps(seq,long=False):
    full_np = np.stack(seq,axis=0)
    if not long:
        return torch.tensor(full_np, dtype=torch.float)
    else:
        return torch.tensor(full_np, dtype=torch.long)

def run_validation_pass(PARAMS, model, validation_data, loss_function_next_step, loss_function_endpoint, writer, log_stats_dict, log_idx, is_epoch=True):
    with torch.no_grad():
        model.eval()
        for step_images, targ_next_step_idx, targ_area_next_step in validation_data:
            model.zero_grad()
            step_images_in = prepare_sequence_steps(step_images).to(torch.device(PARAMS['DEVICE']))
            n_steps = step_images_in.shape[0]
            np_targ_endpt = np.zeros((n_steps))
            np_targ_endpt[n_steps-1] = 1
            endpoint_targ_t = torch.tensor(np_targ_endpt).to(torch.device(PARAMS['DEVICE']),dtype=torch.long)
            is_long = PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] and not PARAMS['AREA_TARGET']
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
                np_pred_scores = next_steps_pred_scores.cpu().detach().numpy()
                npts = np_pred_scores.shape[0]
                np_pred = np_pred_scores.reshape(npts,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1)
                np_targ = targets_onept.cpu().detach().numpy()
                np_pred_endpt = np.argmax(endpoint_scores.cpu().detach().numpy(),axis=1)
            elif PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                np_pred = np.argmax(next_steps_pred_scores.cpu().detach().numpy(),axis=1)
                np_targ = targets_onept.cpu().detach().numpy()
            else:
                np_pred = np.rint(next_steps_pred_scores.cpu().detach().numpy()) # Rounded to integers
                np_targ = targets_onept.cpu().detach().numpy()
            # loss_weights = torch.tensor(get_loss_weights_v2(targets.cpu().detach().numpy(),np_pred,PARAMS),dtype=torch.float).to(torch.device(PARAMS['DEVICE']))
            loss_next_steps = loss_function_next_step(next_steps_pred_scores, targets_loss)
            vals_per_step = loss_next_steps.shape[1]
            loss_next_steps_per_step = torch.mean(torch.div(torch.sum(loss_next_steps, dim=1),vals_per_step))
            loss_endpoint   = torch.mean(loss_function_endpoint(endpoint_scores, endpoint_targ_t))
            loss_weighted = loss_next_steps_per_step + loss_endpoint#*loss_weights
            loss_total = loss_weighted
            calc_logger_stats(log_stats_dict, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, len(validation_data), np_pred_endpt, np_targ_endpt, is_train=False, is_epoch=is_epoch)
        prestring = "step_"
        folder = "Step/"
        if is_epoch:
            prestring = "epoch_"
            folder = "Epoch/"
        writer.add_scalar(folder+'validation_loss_total', log_stats_dict[prestring+'val_loss_average'], log_idx)
        writer.add_scalar(folder+'validation_loss_endpointnet', log_stats_dict[prestring+'val_loss_endptnet'], log_idx)
        writer.add_scalar(folder+'validation_loss_stepnet', log_stats_dict[prestring+'val_loss_stepnet'], log_idx)
        writer.add_scalar(folder+'validation_acc_endpoint', log_stats_dict[prestring+'val_acc_endpoint'], log_idx)
        writer.add_scalar(folder+'validation_acc_exact', log_stats_dict[prestring+'val_acc_exact'], log_idx)
        writer.add_scalar(folder+'validation_acc_2dist', log_stats_dict[prestring+'val_acc_2dist'], log_idx)
        writer.add_scalar(folder+'validation_acc_5dist', log_stats_dict[prestring+'val_acc_5dist'], log_idx)
        writer.add_scalar(folder+'validation_acc_10dist', log_stats_dict[prestring+'val_acc_10dist'], log_idx)
        writer.add_scalar(folder+'validation_num_correct_exact', log_stats_dict[prestring+'val_num_correct_exact'], log_idx)
        writer.add_scalar(folder+"validation_average_off_distance", log_stats_dict[prestring+'val_average_distance_off'],log_idx)
        writer.add_scalar(folder+"validation_frac_misIDas_endpoint", log_stats_dict[prestring+'val_frac_misIDas_endpoint'],log_idx)
