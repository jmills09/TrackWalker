import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from MiscFunctions import unflatten_pos, flatten_pos, make_steps_images
from LArMatchModel import LArMatchConvNet


def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

class DataLoader3D:
    def __init__(self, PARAMS, verbose=False, all_train = False, all_valid = False):
        self.PARAMS = PARAMS
        self.verbose = verbose
        self.infile = None
        if all_train:
            self.infile = ROOT.TFile(PARAMS['INFILE_TRAIN'])
        elif all_valid:
            self.infile = ROOT.TFile(PARAMS['INFILE_VAL'])
        else:
            self.infile = ROOT.TFile(PARAMS['INFILE'])

        self.intree = self.infile.Get("TrackWalker3DInput")
        self.RAND_FLIP_INPUTS = PARAMS['RAND_FLIP_INPUT']
        self.nentries = self.intree.GetEntries()
        self.nentries_train = int(self.nentries*0.8)
        self.nentries_val   = self.nentries-self.nentries_train
        self.nentry_val_buffer = self.nentries_train
        self.current_train_entry = 0
        self.current_val_entry   = 0
        if all_train:
            self.nentries_train = self.nentries
            self.nentries_val   = 0
            self.nentry_val_buffer = self.nentries_train
        elif all_valid:
            self.nentries_val = self.nentries
            self.nentries_train   = 0
            self.nentry_val_buffer = self.nentries_train
        print()
        print("Total Events in File:        ", self.nentries)
        print("Total Events in Training:    ", self.nentries_train)
        print("Total Events in Validation:  ", self.nentries_val)
        print()
        self.PDG_to_Part = {
        2212:"PROTON",
        2112:"NEUTRON",
        211:"PIPLUS",
        -211:"PIMINUS",
        111:"PI0",
        11:"ELECTRON",
        -11:"POSITRON",
        13:"MUON",
        -13:"ANTIMUON",
        }
    def get_train_data(self, n_load):
        dim = 2*self.PARAMS['PADDING']+1
        start_entry = self.current_train_entry
        end_entry   = self.current_train_entry + n_load
        if end_entry > self.nentries_train:
            end_entry = self.nentries_train

        print("Loading Train Entries ", start_entry, "to", end=" ")
        training_data = []
        nAdded = 0
        i = start_entry-1
        while nAdded < n_load:
            i += 1
            if i >= self.nentries_train:
                i = -1
                continue

            # print("Doing Track ",i)
            self.intree.GetEntry(i)
            # print("Loaded Train",i)

            stepidx_im_np       = self.intree.stepidx_image_np.tonumpy().copy()
            # if there is only 1 or fewer points on the index map then dont include track
            if len(np.where(stepidx_im_np > 0)[0]) <= 1 or np.amax(stepidx_im_np) == 0:
                continue
            larmatchfeat_im_np  = self.intree.larmatchfeat_image_np.tonumpy().copy()
            wire_im_np          = self.intree.wire_image_np.tonumpy().copy()
            wire_im_np          = np.expand_dims(wire_im_np,axis=2)



            if self.PARAMS['APPEND_WIREIM']:
                larmatchfeat_im_np = np.concatenate((larmatchfeat_im_np,wire_im_np),axis=2)



            feature_ims_np_v, flat_next_positions, flat_area_positions = \
                make_track_crops(larmatchfeat_im_np, stepidx_im_np, self.PARAMS)
            # rerav = []
            # for ar in flat_area_positions:
            #     rerav.append(reravel_array(ar,2*self.PARAMS['PADDING']+1,2*self.PARAMS['PADDING']+1))
            # if i == 7:
            #     import os
            #     make_steps_images(np.stack(rerav,axis=0),"testcomplexformat/TrueStep",2*self.PARAMS['PADDING']+1,targ=flat_next_positions)
            #     convert_cmd = "convert "+"testcomplexformat/*.png "+'testcomplexformat/areapos.pdf'
            #     print(convert_cmd)
            #     os.system(convert_cmd)
            #     assert 1==2


############################
# To Make Images of Track Crops
############################
            # from MiscFunctions import save_im
            # import os
            # dims = stacked_wire_images.shape
            # ydim = int(1000*dims[1]/dims[0])
            # save_im(stacked_wire_images,savename='testcomplexformat/'+str(i).zfill(3)+'_0wire_im',canv_x=1000,canv_y=ydim)
            # save_im(stacked_step_idx,savename='testcomplexformat/'+str(i).zfill(3)+'_1step_idx_im',canv_x=1000,canv_y=ydim)

# #############################
            if self.RAND_FLIP_INPUTS:
                print("Random flipping of images not implemented for complex dataloading")
                assert 1==2


            # flat_next_positions = unstack(stacked_targ_idx)
            training_data.append((feature_ims_np_v,flat_next_positions,flat_area_positions))
            nAdded += 1

        self.current_train_entry = i+1
        print(self.current_train_entry)
        if self.current_train_entry == self.nentries_train:
            self.current_train_entry = 0
        return training_data

    def get_val_data(self, n_load):
        dim = 2*self.PARAMS['PADDING']+1
        start_entry = self.current_val_entry
        end_entry   = self.current_val_entry + n_load
        if end_entry > self.nentries_val:
            end_entry = self.nentries_val
            self.current_val_entry = 0
            start_entry = self.current_val_entry
            end_entry   = self.current_val_entry + n_load
        # print("Loading Val Entries ", start_entry+self.nentry_val_buffer, "to", end_entry+self.nentry_val_buffer)
        val_data = []
        nAdded = 0
        i = start_entry-1
        while nAdded < n_load:
            i += 1
            if i >= self.nentries_val:
                i = -1
                continue

            self.intree.GetEntry(i)
            # print("Loaded Val",i)

            stepidx_im_np       = self.intree.stepidx_image_np.tonumpy().copy()
            if len(np.where(stepidx_im_np > 0)[0]) <= 1 or np.amax(stepidx_im_np) == 0:
                continue
            larmatchfeat_im_np  = self.intree.larmatchfeat_image_np.tonumpy().copy()
            wire_im_np          = self.intree.wire_image_np.tonumpy().copy()
            wire_im_np          = np.expand_dims(wire_im_np,axis=2)

            if self.PARAMS['APPEND_WIREIM']:
                larmatchfeat_im_np = np.concatenate((larmatchfeat_im_np,wire_im_np),axis=2)

            feature_ims_np_v, flat_next_positions, flat_area_positions = \
                make_track_crops(larmatchfeat_im_np, stepidx_im_np, self.PARAMS)

            if self.RAND_FLIP_INPUTS:
                print("Random flipping of images not implemented for complex dataloading")
                assert 1==2
            val_data.append((feature_ims_np_v,flat_next_positions,flat_area_positions))
            nAdded += 1

        self.current_val_entry = i+1
        if self.current_val_entry == self.nentries_val:
            self.current_val_entry = 0
        return val_data

def make_track_crops(feat_im_np, stepidx_im_np, PARAMS):
    feat_steps_np_v = []
    flattened_positions_v  = []
    area_positions_v  = []
    # print("Making Steps")
    # print(feat_im_np.shape)
    # print(stepidx_im_np.shape)
    nSteps = 0
    # for y in range(stepidx_im_np.shape[1]):
    #     for x in range(stepidx_im_np.shape[0]):
    #         print(stepidx_im_np[x,stepidx_im_np.shape[1]-1-y],end=" , ")
    #     print()
    nextfullimPosition  = np.where(stepidx_im_np == np.ma.masked_equal(stepidx_im_np, 0.0, copy=False).min())
    endPosition         = np.where(stepidx_im_np == np.amax(stepidx_im_np))
    lastPosition = -1
    # print()
    # print(nextfullimPosition)
    # print()
    endIdx   = np.amax(feat_im_np)
    isFinished = False
    while lastPosition != nextfullimPosition:
        lowx  = nextfullimPosition[0][0]-PARAMS['PADDING']
        highx = nextfullimPosition[0][0]+PARAMS['PADDING']+1
        lowy  = nextfullimPosition[1][0]-PARAMS['PADDING']
        highy = nextfullimPosition[1][0]+PARAMS['PADDING']+1

        if PARAMS['DO_CROPSHIFT']:
            delta_x = np.random.randint(-2,3)
            delta_y = np.random.randint(-2,3)
            lowx  += delta_x
            highx += delta_x
            lowy  += delta_y
            highy += delta_y
        feat_crop    = feat_im_np[lowx:highx, lowy:highy,:]
        stepidx_crop = stepidx_im_np[lowx:highx, lowy:highy]
        lastPosition = nextfullimPosition
        nextfullimPosition  = np.where(stepidx_im_np == np.amax(stepidx_crop))
        nextcropimPosition  = np.where(stepidx_crop == np.amax(stepidx_crop))
        feat_steps_np_v.append(feat_crop)
        flattened_positions_v.append(nextcropimPosition[0][0]*stepidx_crop.shape[0]+nextcropimPosition[1][0])
        if PARAMS['AREA_TARGET']:
            zeros_np = np.zeros(stepidx_crop.shape)
            area_positions_v.append(unravel_array(paste_target(zeros_np,nextcropimPosition[0][0],nextcropimPosition[1][0],PARAMS['TARGET_BUFFER'])))

    return feat_steps_np_v, flattened_positions_v, area_positions_v



def flip_target_idx_xdim(in_targ_idx,PARAMS):
    out_targ_idx = -999*np.ones(in_targ_idx.shape)
    for i in range(in_targ_idx.shape[0]):
        pos_2d = unflatten_pos(in_targ_idx[i], PARAMS['PADDING']*2+1)
        new_pos2d = [2*PARAMS['PADDING'] - pos_2d[0],pos_2d[1]]
        pos_1d = flatten_pos(new_pos2d[0],new_pos2d[1], PARAMS['PADDING']*2+1)
        out_targ_idx[i] = pos_1d
    return out_targ_idx

def flip_flat_area_positions_xdim(in_flat_area_positions, PARAMS):
        out_flat_area_positions = []
        dim = PARAMS['PADDING']*2+1
        for i in range(len(in_flat_area_positions)):
            this_flat_area_position = in_flat_area_positions[i]
            area_position_2d = this_flat_area_position.reshape(dim,dim)
            area_position_2d = np.flip(area_position_2d,axis=0)
            flat_flip_area = area_position_2d.flatten()
            out_flat_area_positions.append(flat_flip_area)
        return out_flat_area_positions
