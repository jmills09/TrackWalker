import ROOT
import numpy as np
import os, sys

def make_prediction_vector(PARAMS, np_pred):
    # Take in a np array np_pred of (nsteps,xdim,ydim) and convert it to a vector
    # of (nsteps,1dpix predictions)
    np_preds_vec = np.zeros((np_pred.shape[0]))
    for ix in range(np_pred.shape[0]):
        np_flat = np_pred[ix].flatten()
        np_preds_vec[ix] = np.argmax(np_flat,axis=0)
    return np_preds_vec

def make_log_stat_dict(namestring=''):
    if namestring != '':
        if namestring[-1] != "_":
            namestring = namestring + "_"
    log_dict = {}
    log_dict[namestring+'loss_average']= 0
    log_dict[namestring+'loss_stepnet']= 0
    log_dict[namestring+'loss_endptnet']= 0

    log_dict[namestring+'acc_endpoint']= 0
    log_dict[namestring+'num_correct_exact']= 0
    log_dict[namestring+'frac_misIDas_endpoint']= 0
    log_dict[namestring+'acc_exact']= 0
    log_dict[namestring+'acc_2dist']= 0
    log_dict[namestring+'acc_5dist']= 0
    log_dict[namestring+'acc_10dist']= 0
    log_dict[namestring+'average_distance_off']= 0
    return log_dict

def calc_logger_stats(log_stats_dict, PARAMS, np_pred, np_targ,
            loss_total, loss_endptnet, loss_stepnet,
            batch_size, endpt_pred, endpt_targ,
            is_train=True, is_epoch=True):

    input_image_dimension = PARAMS['PADDING']*2+1
    num_correct_endpoint = 0
    num_mislabeledas_endpoint = 0
    num_correct_exact = 0
    num_correct_2dist = 0
    num_correct_5dist = 0
    num_correct_10dist = 0
    prestring = ""
    if is_epoch:
        prestring = "epoch_"
    else:
        prestring = "step_"
    if is_train:
        prestring = prestring + "train_"
    else:
        prestring = prestring + "val_"

    dists = []
    np_pred_flat_idx_vec = make_prediction_vector(PARAMS, np_pred)
    for ix in range(np_pred_flat_idx_vec.shape[0]):
        this_dist = get_pred_targ_dist(np_pred_flat_idx_vec[ix],np_targ[ix], input_image_dimension)
        dists.append(this_dist)
        if ix != np_pred_flat_idx_vec.shape[0]-1:
            if np_pred_flat_idx_vec[ix] == np_targ[ix]:
                num_correct_exact = num_correct_exact + 1
            if this_dist <= 2.0:
                num_correct_2dist += 1
            if this_dist <= 5.0:
                num_correct_5dist += 1
            if this_dist <= 10.0:
                num_correct_10dist += 1

    # Calc Endpoint Network Metrics, Is last Pt Correct? How many steps get called end?
    for ix in range(endpt_pred.shape[0]):
        if endpt_pred[ix] == 1 and endpt_targ[ix] == 1:
            num_correct_endpoint += 1
        elif endpt_pred[ix] == 1 and endpt_targ[ix] != 1:
            num_mislabeledas_endpoint += 1


    log_stats_dict[prestring+'loss_average'] += loss_total.cpu().detach().numpy()/batch_size
    log_stats_dict[prestring+'loss_endptnet'] += loss_endptnet.cpu().detach().numpy()/batch_size
    log_stats_dict[prestring+'loss_stepnet'] += loss_stepnet.cpu().detach().numpy()/batch_size

    log_stats_dict[prestring+'acc_endpoint'] += float(num_correct_endpoint)/batch_size
    log_stats_dict[prestring+'num_correct_exact'] += float(num_correct_exact)/batch_size
    log_stats_dict[prestring+'frac_misIDas_endpoint'] += num_mislabeledas_endpoint/float(endpt_pred.shape[0]-1)/batch_size

    if len(dists) != 0:
        log_stats_dict[prestring+'acc_exact'] += float(num_correct_exact)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_2dist'] += float(num_correct_2dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_5dist'] += float(num_correct_5dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_10dist'] += float(num_correct_10dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'average_distance_off'] += np.mean(dists)/batch_size
    return log_stats_dict







def get_loss_weights_v2(targets, np_pred, PARAMS):
    dim = PARAMS['PADDING']*2+1
    loss_weights = np.ones((targets.shape[0]))
    # These special weights are only used if PARAMS['CENTERPOINT_ISEND'] is false
    misID_mid_as_end_weight =  1 #float(dim+dim)/2 #(dim+dim) / 2
    misID_end_as_mid_weight =  0.00001 #float(dim+dim)/2 #(dim+dim) / 2
    for idx in range(targets.shape[0]):
        target = targets[idx]
        pred   = np_pred[idx]
        if target == pred:
            loss_weights[idx] = 1.0
        elif target != dim*dim and pred != dim*dim:
            targ_x, targ_y = unflatten_pos(target, dim)
            pred_x, pred_y = unflatten_pos(pred,   dim)
            loss_weights[idx] = (abs(targ_x - pred_x)**2 + abs(targ_y - pred_y)**2)#**0.5
        elif target == dim*dim and pred != dim*dim:
            loss_weights[idx] = misID_end_as_mid_weight
        elif target != dim*dim and pred == dim*dim:
            loss_weights[idx] = misID_mid_as_end_weight
    return loss_weights

def get_pred_targ_dist(pred, targ, dim):
    if targ == pred:
        return 0
    else:
        targ_x, targ_y = unflatten_pos(targ, dim)
        pred_x, pred_y = unflatten_pos(pred, dim)
        dist = ((targ_x - pred_x)**2 + (targ_y - pred_y)**2)**0.5
        return dist


def save_im(np_arr, savename="file",pred_next_x=None,pred_next_y=None,true_next_x=None,true_next_y=None,canv_x=-1,canv_y=-1):
    ROOT.gStyle.SetOptStat(0)
    x_len = np_arr.shape[0]
    y_len = np_arr.shape[1]
    hist = ROOT.TH2F(savename,savename,x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0)
    for x in range(x_len):
        for y in range(y_len):
            hist.SetBinContent(x+1,y+1,np_arr[x,y])
    xscale = 1000
    yscale =  800
    if canv_x != -1:
        xscale = canv_x
    if canv_y != -1:
        yscale = canv_y
    # yscale = int(4000.0*np_arr.shape[1]/np_arr.shape[0]-200)
    canv = ROOT.TCanvas('canv','canv',xscale,yscale)

    hist.SetMaximum(50.0)
    hist.Draw("COLZ")
    hist_vert = ROOT.TH2F("vert","vert",x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0)
    vert_x = (x_len)/2
    vert_y = (y_len)/2
    # hist_vert.Fill(vert_x,vert_y)
    # hist_vert.SetMarkerStyle(29)
    # hist_vert.SetMarkerColor(1)
    # hist_vert.SetMarkerSize(2)
    # hist_vert.Draw("SAME")

    # graph_vert.SetMarkerLineWidth(5)
    if pred_next_x != None:
        graph_next = ROOT.TGraph()
        graph_next.SetPoint(0,pred_next_x+0.5,pred_next_y+0.5)
        graph_next.SetMarkerStyle(22)
        graph_next.SetMarkerColor(2)
        graph_next.SetMarkerSize(2)
        # graph_next.SetMarkerLineWidth(5)
        graph_next.Draw("PSAME")
    if true_next_x != None:
        graph_targ = ROOT.TGraph()
        graph_targ.SetPoint(0,true_next_x+0.5,true_next_y+0.5)
        graph_targ.SetMarkerStyle(23)
        graph_targ.SetMarkerColor(4)
        graph_targ.SetMarkerSize(2)
        # graph_targ.SetMarkerLineWidth(5)
        graph_targ.Draw("PSAME")
    graph_vert = ROOT.TGraph()
    graph_vert.SetPoint(0,int(vert_x)+0.5,int(vert_y)+0.5)
    graph_vert.SetMarkerStyle(8)
    graph_vert.SetMarkerColor(1)
    graph_vert.SetMarkerSize(1)
    graph_vert.Draw("PSAME")

    canv.SaveAs(savename+'.png')
    return 0

def make_steps_images(np_images,string_pattern,dim,pred=None,targ=None):
    for im_ix in range(np_images.shape[0]):
    # for im_ix in range(2):
        y = np_images[im_ix]
        y = reravel_array(y,dim,dim)
        pred_next_pos = pred[im_ix]
        pred_next_x, pred_next_y = unflatten_pos(pred_next_pos,dim)

        true_next_pos = targ[im_ix]
        true_next_x, true_next_y = unflatten_pos(true_next_pos,dim)
        if pred_next_pos == dim**2: # if predicted track end
            pred_next_x = y.shape[0]/2
            pred_next_y = y.shape[1]/2
        if true_next_pos == dim**2: #If true track end
            true_next_x = y.shape[0]/2-0.5
            true_next_y = y.shape[1]/2-0.5
        save_im(y,string_pattern+str(im_ix).zfill(3),pred_next_x,pred_next_y,true_next_x,true_next_y)
    return 0
    # save_im(cropped_step_image,'step'+str(idx))


def momentum_walk(np_arr,position,momentum):

    return np_arr,new_position

def dumb_diag_test(steps = None):
    np_arr = np.zeros((50,50))
    if steps == None:
        steps = [0,1,2,3,5,6,8,10,11,14,16,17,19,21,22,23,26,28,30,32,33,34,35,39,41,42,44,49]
        # np_arr[0,1] = np.random.rand()*10
        # np_arr[49,48] = np.random.rand()*10
    pos = 0
    for step in steps:
        # pos = pos + step
        np_arr[step,step] = 10 #np.random.rand()*10
    return np_arr

def unravel_array(np_arr):
    unraveled = np_arr.flatten()
    return unraveled

def reravel_array(np_arr,x_dim,y_dim):
    raveled = np_arr.reshape(x_dim,y_dim)
    return raveled

def flatten_pos(x,y,square_dim):
    new_pos = x*square_dim+y
    return new_pos

def unflatten_pos(pos,square_dim):
    x = int(pos/square_dim)
    y = pos%square_dim
    return x,y

def cropped_np(np_arr,x_center,y_center,padding):
    if len(np_arr.shape) == 2:
        pad_widths = padding
    else:
        pad_widths = [(padding,padding) for p in range(len(np_arr.shape)-1)]
        pad_widths.append((0,0))
    pad_arr = np.pad(np_arr,pad_widths)
    x_st = x_center
    x_end = x_center+padding+padding+1
    y_st = y_center
    y_end = y_center+padding+padding+1
    new_arr = pad_arr[int(x_st):int(x_end),int(y_st):int(y_end)]
    return new_arr

def paste_target(np_arr,x_targ,y_targ,buffer):
    low_x = int(x_targ-buffer)
    low_y = int(y_targ-buffer)
    high_x = int(x_targ+buffer+1)
    high_y = int(y_targ+buffer+1)
    if low_x < 0:
        low_x = 0
    if low_y < 0:
        low_y = 0
    if high_x > np_arr.shape[0]:
        high_x = np_arr.shape[0]
    if high_y > np_arr.shape[1]:
        high_y = np_arr.shape[1]
    np_arr[low_x:high_x,low_y:high_y] = 1
    return np_arr

def calc_logger_stats_prearea_predict(log_stats_dict, PARAMS, np_pred, np_targ, loss_total, batch_size,
            is_train=True, is_epoch=True):

    input_image_dimension = PARAMS['PADDING']*2+1
    num_correct_endpoint = 0
    num_mislabeledas_endpoint = 0
    num_correct_exact = 0
    num_correct_2dist = 0
    num_correct_5dist = 0
    num_correct_10dist = 0
    prestring = ""
    if is_epoch:
        prestring = "epoch_"
    else:
        prestring = "step_"
    if is_train:
        prestring = prestring + "train_"
    else:
        prestring = prestring + "val_"

    dists = []
    np_pred_flat_idx_vec = make_prediction_vector(PARAMS, np_pred)
    for ix in range(np_pred_flat_idx_vec.shape[0]-1):
        if np_pred_flat_idx_vec[ix] == PARAMS['TRACKEND_CLASS']:
            num_mislabeledas_endpoint += 1
            continue
        this_dist = get_pred_targ_dist(np_pred_flat_idx_vec[ix],np_targ[ix], input_image_dimension)
        dists.append(this_dist)
        if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
            if ix != np_pred_flat_idx_vec.shape[0]-1:
                if np_pred_flat_idx_vec[ix] == np_targ[ix]:
                    num_correct_exact = num_correct_exact + 1
                if this_dist <= 2.0:
                    num_correct_2dist += 1
                if this_dist <= 5.0:
                    num_correct_5dist += 1
                if this_dist <= 10.0:
                    num_correct_10dist += 1
        else:
            if np.array_equal(np_pred_flat_idx_vec[ix], np_targ[ix]):
                num_correct_exact += 1
    # HANDLE CASE FOR last ix (Track End)
    if np_pred_flat_idx_vec[np_pred_flat_idx_vec.shape[0]-1] == np_targ[np_pred_flat_idx_vec.shape[0]-1]:
        num_correct_endpoint += 1

    log_stats_dict[prestring+'loss_average'] += loss_total.cpu().detach().numpy()/batch_size
    log_stats_dict[prestring+'acc_endpoint'] += float(num_correct_endpoint)/batch_size
    log_stats_dict[prestring+'num_correct_exact'] += float(num_correct_exact)/batch_size
    log_stats_dict[prestring+'frac_misIDas_endpoint'] += num_mislabeledas_endpoint/float(np_pred_flat_idx_vec.shape[0]-1)/batch_size

    if len(dists) != 0:
        log_stats_dict[prestring+'acc_exact'] += float(num_correct_exact)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_2dist'] += float(num_correct_2dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_5dist'] += float(num_correct_5dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'acc_10dist'] += float(num_correct_10dist)/float(len(dists))/batch_size
        log_stats_dict[prestring+'average_distance_off'] += np.mean(dists)/batch_size
    assert 1==2
    # This is an old method, do not use
    return log_stats_dict

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stdout = sys.__stderr__
