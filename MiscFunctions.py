import ROOT
import numpy as np
import os, sys
import signal
import socket
from datetime import datetime
from tensorboardX import SummaryWriter

def removeTrackWidth(wire_im, larmatch_im, x, y, old_dx, old_dy, halfWidth = 5):
    # Given two arrays and a current xy, zero out pixels on perpendicular slow
    # to dx / dy
    # Calc perpendicular slope

    dx = old_dy
    dy = 0-old_dx
    if abs(dx) >= abs(dy):
        DX = halfWidth
        # if dx != 0 and dy != 0:
        #     DX = int((halfWidth*1.0)/(dy*1.0/dx*1.0))
        low  = 0  if dx > 0 else DX*-1
        high = DX if dx > 0 else 0
        ddx_list = range(low,high) if dx > 0 else reversed(range(low,high))
        for ddx in ddx_list:
            ddy = int(float(ddx)*float(dy)/float(dx))
            wire_im[x+ddx,y+ddy] = 0
            larmatch_im[x+ddx,y+ddy,:] = 0
    else:
        DY = halfWidth
        # if dy != 0 and dx != 0:
            # DY = int((halfWidth*1.0)/(dx*1.0/dy*1.0))
        low  = 0  if dy > 0 else DY*-1
        high = DY if dy > 0 else 0
        ddy_list = range(low,high) if dy > 0 else reversed(range(low,high))
        for ddy in ddy_list:
            ddx = int(float(ddy)*float(dx)/float(dy))
            wire_im[x+ddx,y+ddy] = 0
            larmatch_im[x+ddx,y+ddy,:] = 0
    # Now do it in the other direction of the perpendicular slope
    dx = 0-old_dy
    dy = old_dx

    if abs(dx) >= abs(dy):
        DX = halfWidth
        # if dx != 0 and dy != 0:
        #     DX = int((halfWidth*1.0)/(dy*1.0/dx*1.0))
        low  = 0  if dx <= 0 else DX*-1
        high = DX if dx <= 0 else 0
        ddx_list = range(low,high) if dx > 0 else reversed(range(low,high))
        for ddx in ddx_list:
            ddy = int(float(ddx)*float(dy)/float(dx))
            wire_im[x+ddx,y+ddy] = 0
            larmatch_im[x+ddx,y+ddy,:] = 0
    else:
        DY = halfWidth
        # if dy != 0 and dx != 0:
        #     DY = int((halfWidth*1.0)/(dx*1.0/dy*1.0))
        low  = 0  if dy > 0 else DY*-1
        high = DY if dy > 0 else 0
        ddy_list = range(low,high) if dy > 0 else reversed(range(low,high))
        for ddy in ddy_list:
            ddx = int(float(ddy)*float(dx)/float(dy))
            wire_im[x+ddx,y+ddy] = 0
            larmatch_im[x+ddx,y+ddy,:] = 0
    return 0

def removeTrackWidth_v2(wire_im, larmatch_im, x, y, old_dx, old_dy, halfWidth = 5):
    xmin = x-halfWidth   if x-halfWidth   >= 0 else 0
    xmax = x+halfWidth+1 if x+halfWidth+1 <= wire_im.shape[0] else wire_im.shape[0]
    ymin = y-halfWidth   if y-halfWidth   >= 0 else 0
    ymax = y+halfWidth+1 if y+halfWidth+1 <= wire_im.shape[1] else wire_im.shape[1]
    wire_im[xmin:xmax,ymin:ymax]       = 0
    larmatch_im[xmin:xmax,ymin:ymax,:] = 0
    return 0


def removeChargeOnTrackSegment(wire_im, larmatch_im, this_x, this_y, last_x, last_y):
    if this_x == last_x and this_y == last_y:
        return 0
    # Given two arrays, and a current and last step, go through and
    # zero out pixels between the last and current step
    dx = this_x - last_x
    dy = this_y - last_y
    # If moving more in x than y
    wire_im[last_x,last_y] = 0
    larmatch_im[last_x,last_y,:] = 0
    removeTrackWidth_v2(wire_im, larmatch_im, last_x, last_y, dx, dy)
    if abs(dx) >= abs(dy):
        low  = 0  if dx > 0 else dx
        high = dx if dx > 0 else 0
        ddx_list = range(low,high) if dx > 0 else reversed(range(low,high))
        for ddx in ddx_list:
            ddy = int(float(ddx)*float(dy)/float(dx))
            wire_im[last_x+ddx,last_y+ddy] = 0
            larmatch_im[last_x+ddx,last_y+ddy,:] = 0
            removeTrackWidth_v2(wire_im, larmatch_im, last_x+ddx, last_y+ddy, dx, dy)
    else:
        low  = 0  if dy > 0 else dy
        high = dy if dy > 0 else 0
        ddy_list = range(low,high) if dy > 0 else reversed(range(low,high))
        for ddy in ddy_list:
            ddx = int(float(ddy)*float(dx)/float(dy))
            wire_im[last_x+ddx,last_y+ddy] = 0
            larmatch_im[last_x+ddx,last_y+ddy,:] = 0
            removeTrackWidth_v2(wire_im, larmatch_im, last_x+ddx, last_y+ddy, dx, dy)


    return 0


def removeTrackWidth_v2(wire_im, larmatch_im, mask_im, x, y, old_dx, old_dy, halfWidth = 2):
    xmin = x-halfWidth   if x-halfWidth   >= 0 else 0
    xmax = x+halfWidth+1 if x+halfWidth+1 <= wire_im.shape[0] else wire_im.shape[0]
    ymin = y-halfWidth   if y-halfWidth   >= 0 else 0
    ymax = y+halfWidth+1 if y+halfWidth+1 <= wire_im.shape[1] else wire_im.shape[1]
    # wire_im[xmin:xmax,ymin:ymax]       = 0
    mask_im[xmin:xmax,ymin:ymax]       = 0
    # larmatch_im[xmin:xmax,ymin:ymax,:] = 0
    return 0


def removeChargeOnTrackSegment(wire_im, larmatch_im, mask_im, this_x, this_y, last_x, last_y):
    if this_x == last_x and this_y == last_y:
        return 0
    # Given two arrays, and a current and last step, go through and
    # zero out pixels between the last and current step
    dx = this_x - last_x
    dy = this_y - last_y
    # If moving more in x than y
    # wire_im[last_x,last_y] = 0
    mask_im[last_x,last_y] = 0
    # larmatch_im[last_x,last_y,:] = 0
    removeTrackWidth_v2(wire_im, larmatch_im, mask_im, last_x, last_y, dx, dy)
    if abs(dx) >= abs(dy):
        low  = 0  if dx > 0 else dx
        high = dx if dx > 0 else 0
        ddx_list = range(low,high) if dx > 0 else reversed(range(low,high))
        for ddx in ddx_list:
            ddy = int(float(ddx)*float(dy)/float(dx))
            if last_x+ddx < 0:
                ddx = 0-last_x
            elif last_x + ddx > mask_im.shape[0]-1:
                ddx = mask_im.shape[0]-1 - last_x
            if last_y+ddy < 0:
                ddy = 0-last_y
            elif last_y + ddy > mask_im.shape[1]-1:
                ddy = mask_im.shape[1]-1 - last_y
            # wire_im[last_x+ddx,last_y+ddy] = 0
            mask_im[last_x+ddx,last_y+ddy] = 0
            # larmatch_im[last_x+ddx,last_y+ddy,:] = 0
            removeTrackWidth_v2(wire_im, larmatch_im, mask_im, last_x+ddx, last_y+ddy, dx, dy)
    else:
        low  = 0  if dy > 0 else dy
        high = dy if dy > 0 else 0
        ddy_list = range(low,high) if dy > 0 else reversed(range(low,high))
        for ddy in ddy_list:
            ddx = int(float(ddy)*float(dx)/float(dy))
            # wire_im[last_x+ddx,last_y+ddy] = 0
            mask_im[last_x+ddx,last_y+ddy] = 0
            # larmatch_im[last_x+ddx,last_y+ddy,:] = 0
            removeTrackWidth_v2(wire_im, larmatch_im, mask_im, last_x+ddx, last_y+ddy, dx, dy)


    return 0




def get_writers(PARAMS):
    writer_dir = ''
    if PARAMS['TENSORDIR'] == None:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        top_log_dir = os.path.join(
            'runs/new_runs', current_time + '_' + socket.gethostname())
        sub_log_dir = socket.gethostname()
        print("DIRNAME:", top_log_dir+"/"+sub_log_dir)
        # if not os.path.exists(top_log_dir):
        #     os.mkdir(top_log_dir)
        writer_dir = top_log_dir+"/"
        writer_train = SummaryWriter(top_log_dir+"/"+sub_log_dir+"_train")
        writer_val   = SummaryWriter(top_log_dir+"/"+sub_log_dir+"_val")
        if not PARAMS['TWOWRITERS']:
            writer_val = writer_train
        with open(top_log_dir+"/"+'PARAMS_LOG.txt', 'w') as logfile:
            logfile.write("Ran with PARAMS:\n")
            logfile.write("------------------------------------------------\n\n")
            # set alarm
            wait_time=30
            signal.signal(signal.SIGALRM, interrupted)
            signal.alarm(wait_time)
            s = get_user_input(wait_time)
            # disable the alarm after success
            logfile.write(s)
            signal.alarm(0)
            for k,v in PARAMS.items():
                line = "PARAMS['"+k+"'] = "+str(v)+"\n"
                logfile.write(line)
            logfile.write('\n\n')

            logfile.close()
    else:
        print("Deprecated to force tensordir")
        assert 1==2
        writer_train = SummaryWriter(top_log_dir=PARAMS['TENSORDIR'])
    return writer_train, writer_val, writer_dir

def interrupted(signum, frame):
    # "called when read times out"
    print('Alright times up, lets do this, LEEROY JENKINS!')
    assert 1==2 # Need some way to throw an error, this better work.


def get_user_input(wait_time=30):
    try:
            print('\n--------------------------------------------\n')
            print('You have '+str(wait_time)+' seconds. Type in any special reasons')
            print('for your run now, press enter when finished:\n')

            foo = input()
            return str(foo)+'\n\n'
    except:
            # timeout
            return '\n\n'



def make_prediction_vector(PARAMS, np_pred):
    # Take in a np array np_pred of (nsteps,xdim,ydim) and convert it to a vector
    # of (nsteps,1dpix predictions)
    np_preds_vec = np.zeros((np_pred.shape[0]))
    for ix in range(np_pred.shape[0]):
        np_flat = np_pred[ix].flatten()
        np_preds_vec[ix] = np.argmax(np_flat,axis=0)
    return np_preds_vec

def make_log_stat_dict(namestring='',PARAMS=None):
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
    if PARAMS != None and PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] != True:
        log_dict[namestring+'offDistX']= 0
        log_dict[namestring+'offDistY']= 0
        log_dict[namestring+'offDistZ']= 0
        log_dict[namestring+'predStepDist']= 0
        log_dict[namestring+'trueStepDist']= 0

    return log_dict

def calc_logger_stats(log_stats_dict, PARAMS, np_pred, np_targ,
            loss_total, loss_endptnet, loss_stepnet,
            batch_size, endpt_pred, endpt_targ,
            is_train=True, is_epoch=True,no_prestring=False):

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
    if not PARAMS['TWOWRITERS']:
        if is_train:
            prestring = prestring + "train_"
        else:
            prestring = prestring + "val_"

    dists = []
    np_pred_flat_idx_vec = make_prediction_vector(PARAMS, np_pred)

    for ix in range(np_pred_flat_idx_vec.shape[0]):
        this_dist = get_pred_targ_dist(np_pred_flat_idx_vec[ix],np_targ[ix], PARAMS['VOXCUBESIDE'])
        if ix != np_pred_flat_idx_vec.shape[0]-1:
            dists.append(this_dist)
            if np_pred_flat_idx_vec[ix] == np_targ[ix]:
                num_correct_exact = num_correct_exact + 1
            if this_dist <= 2.0:
                num_correct_2dist += 1
            if this_dist <= 5.0:
                num_correct_5dist += 1
            if this_dist <= 10.0:
                num_correct_10dist += 1
            else:
                print(np_pred_flat_idx_vec[ix],np_targ[ix], PARAMS['VOXCUBESIDE'])
                print(this_dist)
                assert 1==2

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


def calc_logger_stats_distshifter(log_stats_dict, PARAMS, np_pred_unscaled, np_targ_unscaled,
            loss_total, loss_endptnet, loss_stepnet,
            batch_size, endpt_pred, endpt_targ,
            is_train=True, is_epoch=True,no_prestring=False):
    np_pred = np_pred_unscaled*PARAMS['CONVERT_OUT_TO_DIST']
    np_targ = np_targ_unscaled*PARAMS['CONVERT_OUT_TO_DIST']
    # print("Logger Stats:")
    # for i in range(np_pred.shape[0]):
    #     print(np_pred[i,:], np_targ[i,:], np_pred[i,:]-np_targ[i,:], np.sum((np_pred[i,:]-np_targ[i,:])*(np_pred[i,:]-np_targ[i,:]))**0.5)
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
    if not PARAMS['TWOWRITERS']:
        if is_train:
            prestring = prestring + "train_"
        else:
            prestring = prestring + "val_"

    dists = []
    distx = []
    disty = []
    distz = []
    stepDist = []
    trueStepDist = []
    for ix in range(np_pred.shape[0]):
        this_dist = ((np_pred[ix,0] - np_targ[ix,0])**2 + (np_pred[ix,1] - np_targ[ix,1])**2 + (np_pred[ix,2] - np_targ[ix,2])**2)**0.5
        dists.append(this_dist)
        distx.append(abs(np_pred[ix,0] - np_targ[ix,0]))
        disty.append(abs(np_pred[ix,1] - np_targ[ix,1]))
        distz.append(abs(np_pred[ix,2] - np_targ[ix,2]))
        stepDist.append(((np_pred[ix,0])**2 + (np_pred[ix,1])**2 + (np_pred[ix,2])**2)**0.5)
        trueStepDist.append(((np_targ[ix,0])**2 + (np_targ[ix,1])**2 + (np_targ[ix,2])**2)**0.5)
        if this_dist == 0:
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
        if not PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
            log_stats_dict[prestring+'offDistX'] += np.mean(distx)/batch_size
            log_stats_dict[prestring+'offDistY'] += np.mean(disty)/batch_size
            log_stats_dict[prestring+'offDistZ'] += np.mean(distz)/batch_size
            log_stats_dict[prestring+'predStepDist'] += np.mean(stepDist)/batch_size
            log_stats_dict[prestring+'trueStepDist'] += np.mean(trueStepDist)/batch_size
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
        targ_x, targ_y, targ_z = unflatten_pos(targ, dim)
        pred_x, pred_y, pred_z = unflatten_pos(pred, dim)
        dist = ((targ_x - pred_x)**2 + (targ_y - pred_y)**2 + (targ_z - pred_z)**2)**0.5
        return dist


def save_im(np_arr, savename="file",pred_next_x=None,pred_next_y=None,true_next_x=None,true_next_y=None,canv_x=-1,canv_y=-1,title=""):
    ROOT.gStyle.SetOptStat(0)
    x_len = np_arr.shape[0]
    y_len = np_arr.shape[1]
    if title=="":
        title=savename

    hist = ROOT.TH2F(title,title,x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0)
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
    # hist.SetMaximum(50.0)
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

def save_im_trackline(np_arr, trackline, savename="file",canv_x=-1,canv_y=-1,title=""):
    ROOT.gStyle.SetOptStat(0)
    x_len = np_arr.shape[0]
    y_len = np_arr.shape[1]
    if title=="":
        title=savename

    hist = ROOT.TH2F(title,title,x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0)
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
    hist.SetMaximum(100.0)
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
    # if pred_next_x != None:
    #     graph_next = ROOT.TGraph()
    #     graph_next.SetPoint(0,pred_next_x+0.5,pred_next_y+0.5)
    #     graph_next.SetMarkerStyle(22)
    #     graph_next.SetMarkerColor(2)
    #     graph_next.SetMarkerSize(2)
    #     # graph_next.SetMarkerLineWidth(5)
    #     graph_next.Draw("PSAME")
    # if true_next_x != None:
    #     graph_targ = ROOT.TGraph()
    #     graph_targ.SetPoint(0,true_next_x+0.5,true_next_y+0.5)
    #     graph_targ.SetMarkerStyle(23)
    #     graph_targ.SetMarkerColor(4)
    #     graph_targ.SetMarkerSize(2)
    #     # graph_targ.SetMarkerLineWidth(5)
    #     graph_targ.Draw("PSAME")
    # graph_vert = ROOT.TGraph()
    # graph_vert.SetPoint(0,int(vert_x)+0.5,int(vert_y)+0.5)
    # graph_vert.SetMarkerStyle(8)
    # graph_vert.SetMarkerColor(1)
    # graph_vert.SetMarkerSize(1)
    trackline.SetMarkerSize(1)
    trackline.Draw("SAMEC*")

    canv.SaveAs(savename+'.png')
    return 0

def save_im_multitracks3D(filename, feats_v, recoTrack_tgraphs_vv, vertexImgCoords):
    crop_pad = 50
    wires_v = [feats_v[p].copy() for p in range(3)]
    print(vertexImgCoords, "Cropping Final Image Around Vertex")
    croppedIm_v = [cropped_np(wires_v[p], vertexImgCoords[p+1], vertexImgCoords[0], crop_pad) for p in range(3)]
    ROOT.gStyle.SetOptStat(0)
    x_len = croppedIm_v[0].shape[0]
    y_len = croppedIm_v[0].shape[1]
    if filename=="":
        filename="Test"

    hist_v = [ROOT.TH2F(filename+"_"+str(p),filename+"_"+str(p),x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0) for p in range(3)]
    for x in range(x_len):
        for y in range(y_len):
            for p in range(3):
                hist_v[p].SetBinContent(x+1,y+1,croppedIm_v[p][x,y])
    for p in range(3):
        canv = ROOT.TCanvas('canv','canv',1200,1000)
        hist_v[p].Draw("COLZ")
        vtxTgraph = ROOT.TGraph()
        vtxTgraph.SetMarkerColor(1)
        vtxTgraph.SetMarkerStyle(24)
        vtxTgraph.SetMarkerSize(3)
        vtxTgraph.SetPoint(0,crop_pad+0.5,crop_pad+0.5)

        remadeTgraphs = []
        for idx in range(len(recoTrack_tgraphs_vv[p])):
            remadeTgraphs.append(ROOT.TGraph())

        colorWheel = [2,7,4,3,6,5,9,12,46,13,14,15,16]
        for idx in range(len(recoTrack_tgraphs_vv[p])):
            remadeTgraphs[idx].SetMarkerColor(colorWheel[idx])
            remadeTgraphs[idx].SetMarkerSize(2)
            remadeTgraphs[idx].SetMarkerStyle(24)
            remadeTgraphs[idx].SetLineWidth(2)
            remadeTgraphs[idx].SetLineColor(colorWheel[idx])

            for ptIdx in range(recoTrack_tgraphs_vv[p][idx].GetN()):
                oldX, oldY = ROOT.Double(0), ROOT.Double(0)
                recoTrack_tgraphs_vv[p][idx].GetPoint(ptIdx,oldX,oldY)
                remadeTgraphs[idx].SetPoint(ptIdx,oldX-vertexImgCoords[p+1]+crop_pad,oldY-vertexImgCoords[0]+crop_pad)
            remadeTgraphs[idx].Draw("SAMELP")
        vtxTgraph.Draw("SAMELP")
        canv.SaveAs(filename+"_"+str(p)+".png")


def save_im_multitracks(filename, wireIm_np, recoTrack_tgraphs_v, vertex_x, vertex_y):
    crop_pad = 50
    croppedIm = cropped_np(wireIm_np, vertex_x, vertex_y, crop_pad)
    ROOT.gStyle.SetOptStat(0)
    x_len = croppedIm.shape[0]
    y_len = croppedIm.shape[1]
    if filename=="":
        filename="Test"

    hist = ROOT.TH2F(filename,filename,x_len,0,(x_len)*1.0,y_len,0,(y_len)*1.0)
    for x in range(x_len):
        for y in range(y_len):
            hist.SetBinContent(x+1,y+1,croppedIm[x,y])
    canv = ROOT.TCanvas('canv','canv',1200,1000)
    hist.Draw("COLZ")
    vtxTgraph = ROOT.TGraph()
    vtxTgraph.SetMarkerColor(1)
    vtxTgraph.SetMarkerStyle(24)
    vtxTgraph.SetMarkerSize(3)
    vtxTgraph.SetPoint(0,crop_pad+0.5,crop_pad+0.5)

    remadeTgraphs = []
    for idx in range(len(recoTrack_tgraphs_v)):
        remadeTgraphs.append(ROOT.TGraph())

    colorWheel = [2,7,4,3,6,5,9,12,46,13,14,15,16]
    for idx in range(len(recoTrack_tgraphs_v)):
        remadeTgraphs[idx].SetMarkerColor(colorWheel[idx])
        remadeTgraphs[idx].SetMarkerSize(2)
        remadeTgraphs[idx].SetMarkerStyle(24)
        remadeTgraphs[idx].SetLineWidth(2)
        remadeTgraphs[idx].SetLineColor(colorWheel[idx])

        for ptIdx in range(recoTrack_tgraphs_v[idx].GetN()):
            oldX, oldY = ROOT.Double(0), ROOT.Double(0)
            recoTrack_tgraphs_v[idx].GetPoint(ptIdx,oldX,oldY)
            remadeTgraphs[idx].SetPoint(ptIdx,oldX-vertex_x+crop_pad,oldY-vertex_y+crop_pad)
        remadeTgraphs[idx].Draw("SAMELP")
    vtxTgraph.Draw("SAMELP")
    canv.SaveAs(filename+".png")


def make_steps_images(np_images,string_pattern,dim,pred=None,targ=None,endpoint_scores=None):
    for im_ix in range(np_images.shape[0]):
    # for im_ix in range(2):
        y = np_images[im_ix]
        y = reravel_array(y,dim,dim)
        pred_next_x = None
        pred_next_y = None
        true_next_x = None
        true_next_y = None
        if pred != None:
            pred_next_pos = pred[im_ix]
            pred_next_x, pred_next_y = unflatten_pos(pred_next_pos,dim)
            if pred_next_pos == dim**2: # if predicted track end
                pred_next_x = y.shape[0]/2
                pred_next_y = y.shape[1]/2
        if targ != None:
            true_next_pos = targ[im_ix]
            true_next_x, true_next_y = unflatten_pos(true_next_pos,dim)
            if true_next_pos == dim**2: #If true track end
                true_next_x = y.shape[0]/2-0.5
                true_next_y = y.shape[1]/2-0.5
        if endpoint_scores == None:
            save_im(y,string_pattern+str(im_ix).zfill(3),pred_next_x,pred_next_y,true_next_x,true_next_y)
        else:
            thistitle = string_pattern+str(im_ix).zfill(3)+"_{:.2f}".format(endpoint_scores[im_ix])
            save_im(y,string_pattern+str(im_ix).zfill(3),pred_next_x,pred_next_y,true_next_x,true_next_y,title=thistitle)

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

def unflatten_pos(pos,cube_dim):
    x = int(pos/cube_dim)
    y = pos%cube_dim
    x = int(pos/(cube_dim**2))
    y = int((pos%(cube_dim**2))/cube_dim)
    z = int(pos%cube_dim)
    return x,y,z

def cropped_np(np_arr,x_center,y_center,padding):
    # if len(np_arr.shape) == 2:
    #     pad_widths = padding
    # else:
    #     pad_widths = [(padding,padding) for p in range(len(np_arr.shape)-1)]
    #     pad_widths.append((0,0))
    # # print("Shape Pre Pad:",np_arr.shape)
    # pad_arr = np.pad(np_arr,pad_widths)
    # x_st = x_center
    # x_end = x_center+padding+padding+1
    # y_st = y_center
    # y_end = y_center+padding+padding+1
    # new_arr = pad_arr[int(x_st):int(x_end),int(y_st):int(y_end)]
    new_arr = None
    if len(np_arr.shape) == 2:
        new_arr = np.zeros((padding*2+1,padding*2+1))
    else:
        new_arr = np.zeros((padding*2+1,padding*2+1,np_arr.shape[2]))
    x_st  = x_center - padding
    x_end = x_center + padding + 1
    y_st  = y_center - padding
    y_end = y_center + padding + 1
    x_st_new  = 0
    x_end_new = padding*2+1
    y_st_new  = 0
    y_end_new = padding*2+1
    if x_st >= np_arr.shape[0] or y_st >= np_arr.shape[1]:
        return new_arr
    elif x_end < 0 or y_end < 0:
        return new_arr
    if x_st < 0:
        x_st_new = 0 - x_st
        x_st = 0
    if y_st < 0:
        y_st_new = 0 - y_st
        y_st = 0
    if x_end >= np_arr.shape[0]:
        x_end_new = padding*2+1 - (x_end - np_arr.shape[0])
        x_end = np_arr.shape[0]
    if y_end >= np_arr.shape[1]:
        y_end_new = padding*2+1 - (y_end - np_arr.shape[1])
        y_end = np_arr.shape[1]

    new_arr[int(x_st_new):int(x_end_new),int(y_st_new):int(y_end_new)] = np_arr[int(x_st):int(x_end),int(y_st):int(y_end)]
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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stdout = sys.__stderr__

def ignoreROOT():
    ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 1001;")

def getProngDict():
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
    dict = {}
    dict['mcProngs_h']=mcProngs_h
    dict['mcProngs_thresh_h']=mcProngs_thresh_h
    dict['recoProngs_h']=recoProngs_h
    dict['reco_m_mcProngs_h']=reco_m_mcProngs_h
    dict['reco_m_mcProngs_thresh_h']=reco_m_mcProngs_thresh_h
    return dict

def saveProngDict(prongDict,dir='',mode=""):
    # tmpcan = ROOT.TCanvas('canv','canv',1200,1000)
    # prongDict['mcProngs_h'].Draw()
    # tmpcan.SaveAs(dir+"nue_mcProngs.png")
    # prongDict['mcProngs_thresh_h'].Draw()
    # tmpcan.SaveAs(dir+"nue_mcProngs_thresh.png")
    # prongDict['recoProngs_h'].Draw()
    # tmpcan.SaveAs(dir+"nue_recoProngs.png")
    # prongDict['reco_m_mcProngs_h'].Draw()
    # tmpcan.SaveAs(dir+"nue_reco_minus_mcProngs.png")
    # prongDict['reco_m_mcProngs_thresh_h'].Draw()
    # tmpcan.SaveAs(dir+"nue_reco_minus_mcProngs_thresh.png")
    ROOT.gStyle.SetOptStat(1)
    preText = ""
    if mode == "MCNU_BNB":
        preText = "bnb"
    else:
        preText = "nue"
    tmpcan = ROOT.TCanvas('canv','canv',1200,1000)
    prongDict['mcProngs_h'].Draw()
    tmpcan.SaveAs(dir+preText+"_mcProngs.png")
    prongDict['mcProngs_thresh_h'].Draw()
    tmpcan.SaveAs(dir+preText+"_mcProngs_thresh.png")
    prongDict['recoProngs_h'].Draw()
    tmpcan.SaveAs(dir+preText+"_recoProngs.png")
    prongDict['reco_m_mcProngs_h'].Draw()
    tmpcan.SaveAs(dir+preText+"_reco_minus_mcProngs.png")
    prongDict['reco_m_mcProngs_thresh_h'].Draw()
    tmpcan.SaveAs(dir+preText+"_reco_minus_mcProngs_thresh.png")

def removeDupTracks(recoTracks_v):
    print()
    print("Starting Number of Tracks:", len(recoTracks_v))
    revisedTracks_v = []
    tracksToCut = []
    # Remove Based on Track End proximity
    trackEnds = []
    for idx in range(len(recoTracks_v)):
        if idx in tracksToCut:
            continue
        goodTrack = True
        endX, endY = ROOT.Double(0), ROOT.Double(0)
        recoTracks_v[idx].GetPoint(recoTracks_v[idx].GetN()-1,endX,endY)
        thisEnd = [endX,endY]
        for end in trackEnds:
            dist = ((thisEnd[0] - end[0])**2 + (thisEnd[1] - end[1])**2)**0.5
            if dist < 8:
                goodTrack = False
                tracksToCut.append(idx)
                break

        if goodTrack:
            trackEnds.append(thisEnd)

    # Remove Based on Track Direction
    trackThetas  = []
    trackOrigIdxs= []
    trackLengths = []
    diffAllowance = 0.1
    for idx in range(len(recoTracks_v)):
        if idx in tracksToCut:
            continue
        goodTrack = True
        startX, startY = ROOT.Double(0), ROOT.Double(0)
        recoTracks_v[idx].GetPoint(0,startX,startY)
        endX, endY = ROOT.Double(0), ROOT.Double(0)
        recoTracks_v[idx].GetPoint(recoTracks_v[idx].GetN()-1,endX,endY)
        dx = endX-startX
        dy = endY-startY
        thisTheta  = np.arctan2(dy,dx)
        thisLength = ((dx)**2 + (dy)**2)**0.5
        for passIdx in range(len(trackThetas)):
            theta = trackThetas[passIdx]
            diffTheta = abs(thisTheta - theta)
            if diffTheta < diffAllowance or diffTheta > 2*np.pi - diffAllowance:
                goodTrack = False
                if thisLength < trackLengths[passIdx]:
                    # Track that was stored is longer (better) keep that one
                    tracksToCut.append(idx)
                    break
                else:
                    # This track is longer, keep it, remove prev stored.
                    tracksToCut.append(trackOrigIdxs[passIdx])
                    # Swap stored info in lists
                    trackThetas[passIdx]   = thisTheta
                    trackOrigIdxs[passIdx] = idx
                    trackLengths[passIdx]  = thisLength



        if goodTrack:
            trackThetas.append(thisTheta)
            trackOrigIdxs.append(idx)
            trackLengths.append(thisLength)


    #Add good Tracks:
    for idx in range(len(recoTracks_v)):
        if idx not in tracksToCut:
            revisedTracks_v.append(recoTracks_v[idx])
    print("Ending Number of Tracks:", len(revisedTracks_v))
    print()
    return revisedTracks_v
