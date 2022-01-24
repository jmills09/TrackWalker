import ROOT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MiscFunctions import calc_logger_stats, calc_logger_stats_distshifter, get_loss_weights_v2, reravel_array


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden2endpoint = nn.Linear(hidden_dim, 2) # head to classify step as endpoint

    def forward(self, sequence, hidden_input=None, cell_input=None):
        nsteps = sequence.shape[0]
        sequence_reshaped = sequence.view((nsteps,1,-1))
        lstm_out = None
        hidden_n = None
        cell_n   = None
        if hidden_input != None and cell_input != None:
            lstm_out, (hidden_n, cell_n) = self.lstm(sequence_reshaped,(hidden_input,cell_input))
        else:
            lstm_out, (hidden_n, cell_n) = self.lstm(sequence_reshaped)

        tag_space = self.hidden2tag(lstm_out.view(sequence.shape[0], -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # tag_scores = (1+torch.tanh(tag_space))/2
        tag_scores = torch.tanh(tag_space)
        endpoint_space = self.hidden2endpoint(lstm_out.view(sequence.shape[0],-1))
        endpoint_scores = F.log_softmax(endpoint_space, dim=1)
        # print("      Reg Softmax",F.softmax(endpoint_space, dim=1).detach().cpu().numpy())
        return tag_scores, endpoint_scores, hidden_n.detach(), cell_n.detach()


class GRUTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(GRUTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden2endpoint = nn.Linear(hidden_dim, 2) # head to classify step as endpoint

    def forward(self, sequence, hidden_input=None, cell_input=None):
        nsteps = sequence.shape[0]
        sequence_reshaped = sequence.view((nsteps,1,-1))
        lstm_out = None
        hidden_n = None
        cell_n   = None

        gru_out, hidden_n = self.gru(sequence_reshaped)

        tag_space = self.hidden2tag(gru_out.view(sequence.shape[0], -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = (1+torch.tanh(tag_space))/2
        endpoint_space = self.hidden2endpoint(gru_out.view(sequence.shape[0],-1))
        endpoint_scores = F.log_softmax(endpoint_space, dim=1)
        _ = None
        return tag_scores, endpoint_scores, hidden_n.detach(), _



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

def run_validation_pass(PARAMS, model, reformatloader, loss_function_next_step, loss_function_endpoint, writer, log_stats_dict, log_idx, is_epoch=True):
    with torch.no_grad():
        model.eval()
        start_entry = reformatloader.current_val_entry
        end_entry   = reformatloader.current_val_entry + PARAMS['VAL_SAMPLE_SIZE']
        if end_entry > reformatloader.nentries_val:
            end_entry = reformatloader.nentries_val
            start_entry = 0
            end_entry   = 0 + PARAMS['VAL_SAMPLE_SIZE']
        for i in range(PARAMS['VAL_SAMPLE_SIZE']):
            validation_data = reformatloader.get_val_data(1)
            for step3d_crops_in_np, xyzShifts_targ_np in validation_data:
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
                xyzShifts_pred_np = xyzShifts_pred_t.cpu().detach().numpy()
                xyzShifts_targ_t = torch.tensor(xyzShifts_targ_np).to(torch.device(PARAMS['DEVICE']))

                # Calc loss and backward pass
                loss_next_steps = loss_function_next_step(xyzShifts_pred_t, xyzShifts_targ_t)
                vals_per_step = loss_next_steps.shape[1]
                loss_next_steps_per_step =torch.mean(torch.div(torch.sum(loss_next_steps, dim=1),vals_per_step))*PARAMS['NEXTSTEP_LOSS_WEIGHT']
                loss_endpoint   = torch.mean(loss_function_endpoint(endpoint_pred_scores_t, endpoint_targ_t))*PARAMS['ENDSTEP_LOSS_WEIGHT']
                loss_total = loss_next_steps_per_step + loss_endpoint
                if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    calc_logger_stats(log_stats_dict, PARAMS, np_pred, np_targ, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['VAL_SAMPLE_SIZE'], np_pred_endpt, np_targ_endpt, is_train=False, is_epoch=is_epoch)
                else:
                    calc_logger_stats_distshifter(log_stats_dict, PARAMS, xyzShifts_pred_np, xyzShifts_targ_np, loss_total, loss_endpoint, loss_next_steps_per_step, PARAMS['VAL_SAMPLE_SIZE'], endpoint_pred_scores_np, endpoint_targ_np, is_train=False, is_epoch=is_epoch)

        print("Loaded Val Entries ", start_entry, "to", reformatloader.current_val_entry)
        prestring = "step_"
        folder = "Step/"
        if is_epoch:
            prestring = "epoch_"
            folder = "Epoch/"
        if not PARAMS['TWOWRITERS']:
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
            if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                writer.add_scalar(folder+"validation_average_off_distanceX", log_stats_dict[prestring+'val_offDistX'],log_idx)
                writer.add_scalar(folder+"validation_average_off_distanceY", log_stats_dict[prestring+'val_offDistY'],log_idx)
                writer.add_scalar(folder+"validation_average_off_distanceZ", log_stats_dict[prestring+'val_offDistZ'],log_idx)
                writer.add_scalar(folder+"validation_predStepDist", log_stats_dict[prestring+'val_predStepDist'],log_idx)
                writer.add_scalar(folder+"validation_trueStepDist", log_stats_dict[prestring+'val_trueStepDist'],log_idx)

            writer.add_scalar(folder+"validation_frac_misIDas_endpoint", log_stats_dict[prestring+'val_frac_misIDas_endpoint'],log_idx)
            writer.add_scalar(folder+"validation_lr", PARAMS['CURRENT_LR'],log_idx)
        else:
            writer.add_scalar(folder+'loss_total', log_stats_dict[prestring+'loss_average'], log_idx)
            writer.add_scalar(folder+'loss_endpointnet', log_stats_dict[prestring+'loss_endptnet'], log_idx)
            writer.add_scalar(folder+'loss_stepnet', log_stats_dict[prestring+'loss_stepnet'], log_idx)
            writer.add_scalar(folder+'acc_endpoint', log_stats_dict[prestring+'acc_endpoint'], log_idx)
            writer.add_scalar(folder+'acc_exact', log_stats_dict[prestring+'acc_exact'], log_idx)
            writer.add_scalar(folder+'acc_2dist', log_stats_dict[prestring+'acc_2dist'], log_idx)
            writer.add_scalar(folder+'acc_5dist', log_stats_dict[prestring+'acc_5dist'], log_idx)
            writer.add_scalar(folder+'acc_10dist', log_stats_dict[prestring+'acc_10dist'], log_idx)
            writer.add_scalar(folder+'num_correct_exact', log_stats_dict[prestring+'num_correct_exact'], log_idx)
            writer.add_scalar(folder+"average_off_distance", log_stats_dict[prestring+'average_distance_off'],log_idx)
            if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
                writer.add_scalar(folder+"average_off_distanceX", log_stats_dict[prestring+'offDistX'],log_idx)
                writer.add_scalar(folder+"average_off_distanceY", log_stats_dict[prestring+'offDistY'],log_idx)
                writer.add_scalar(folder+"average_off_distanceZ", log_stats_dict[prestring+'offDistZ'],log_idx)
                writer.add_scalar(folder+"predStepDist", log_stats_dict[prestring+'predStepDist'],log_idx)
                writer.add_scalar(folder+"trueStepDist", log_stats_dict[prestring+'trueStepDist'],log_idx)
            writer.add_scalar(folder+"frac_misIDas_endpoint", log_stats_dict[prestring+'frac_misIDas_endpoint'],log_idx)
            writer.add_scalar(folder+"lr", PARAMS['CURRENT_LR'],log_idx)
