import ROOT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MiscFunctions import calc_logger_stats, get_loss_weights_v2


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden2endpoint = nn.Linear(hidden_dim, 2) # head to classify step as endpoint

    def forward(self, sentence):
        sentence_reshaped = sentence.view((sentence.shape[0],1,-1))
        lstm_out, _ = self.lstm(sentence_reshaped)
        tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        endpoint_space = self.hidden2endpoint(lstm_out.view(sentence.shape[0],-1))
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

def run_validation_pass(PARAMS, model, validation_data, loss_function, writer, log_stats_dict, log_idx, is_epoch=True):
    with torch.no_grad():
        model.eval()
        for step_images, next_steps in validation_data:
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
            calc_logger_stats(log_stats_dict, PARAMS, np_pred, np_targ, loss_total, len(validation_data), is_train=False, is_epoch=is_epoch)
        prestring = "step_"
        folder = "Step/"
        if is_epoch:
            prestring = "epoch_"
            folder = "Epoch/"
        writer.add_scalar(folder+'validation_loss', log_stats_dict[prestring+'val_loss_average'], log_idx)
        writer.add_scalar(folder+'validation_acc_endpoint', log_stats_dict[prestring+'val_acc_endpoint'], log_idx)
        writer.add_scalar(folder+'validation_acc_exact', log_stats_dict[prestring+'val_acc_exact'], log_idx)
        writer.add_scalar(folder+'validation_acc_2dist', log_stats_dict[prestring+'val_acc_2dist'], log_idx)
        writer.add_scalar(folder+'validation_acc_5dist', log_stats_dict[prestring+'val_acc_5dist'], log_idx)
        writer.add_scalar(folder+'validation_acc_10dist', log_stats_dict[prestring+'val_acc_10dist'], log_idx)
        writer.add_scalar(folder+'validation_num_correct_exact', log_stats_dict[prestring+'val_num_correct_exact'], log_idx)
        writer.add_scalar(folder+"validation_average_off_distance", log_stats_dict[prestring+'val_average_distance_off'],log_idx)
        writer.add_scalar(folder+"validation_frac_misIDas_endpoint", log_stats_dict[prestring+'val_frac_misIDas_endpoint'],log_idx)
