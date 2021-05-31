import ROOT
import numpy as np

def get_loss_weights(targets, np_pred, dim):
    np_wgts_full = np.ones((targets.shape[0],dim*dim+1))
    for idx in range(targets.shape[0]):
        target = targets[idx]
        np_wgts = np.ones((dim,dim))
        if target != dim*dim: #This is the end of track class label
            targ_x, targ_y = unflatten_pos(target,dim)
            for x in range(dim):
                for y in range(dim):
                    # np_wgts[x][y] = ((targ_x-x)**2 + (targ_y-y)**2)**0.5 #Distance Squared
                    np_wgts[x][y] = abs(targ_x - x) + abs(targ_y - y) #dx + dy
        np_wgts[targ_x][targ_y] = 1.0
        # END_TRACK_WGT = ((dim*1.0/2)**2 + (dim*1.0/2)**2)**0.5 # Like being off by halfway across the image diagonally.
        END_TRACK_WGT = dim
        np_wgts_flat = np.append(unravel_array(np_wgts), END_TRACK_WGT)
        np_wgts_full[idx] = np_wgts_flat

    loss_weights = np.zeros((targets.shape[0]))
    for i in range(loss_weights.shape[0]):
        loss_weights[i] = np_wgts_full[i][np_pred[i]]

    return loss_weights

def get_loss_weights_v2(targets, np_pred, dim):
    loss_weights = np.ones((targets.shape[0]))

    for idx in range(targets.shape[0]):
        target = targets[idx]
        pred   = np_pred[idx]
        if target == pred:
            loss_weights[idx] = 1.0
        elif target != dim*dim and pred != dim*dim:
            targ_x, targ_y = unflatten_pos(target,dim)
            pred_x, pred_y = unflatten_pos(pred,  dim)
            loss_weights[idx] = abs(targ_x - pred_x) + abs(targ_y - pred_y)
        elif target == dim*dim and pred != dim*dim:
            loss_weights[idx] = (dim+dim) / 2
        elif target != dim*dim and pred == dim*dim:
            loss_weights[idx] = (dim+dim) / 2
    return loss_weights


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
    pad_arr = np.pad(np_arr,padding)
    x_st = x_center
    x_end = x_center+padding+padding+1
    y_st = y_center
    y_end = y_center+padding+padding+1
    new_arr = pad_arr[int(x_st):int(x_end),int(y_st):int(y_end)]

    return new_arr
