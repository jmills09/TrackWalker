import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from MiscFunctions import unflatten_pos, flatten_pos
from LArMatchModel import LArMatchConvNet

def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

class ReformattedDataLoader_MC:
    def __init__(self, PARAMS, verbose=False, all_train = False, all_valid = False):
        self.PARAMS = PARAMS
        self.verbose = verbose

        # self.infile = ROOT.TFile("/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/Reformat_LArMatch_Pad_010.root")
        # self.infile = ROOT.TFile("/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/endlevel_reformat_partial1221files.root")
        self.infile = None
        if all_train:
            self.infile = ROOT.TFile(PARAMS['INFILE_TRAIN'])
        elif all_valid:
            self.infile = ROOT.TFile(PARAMS['INFILE_VAL'])
        else:
            self.infile = ROOT.TFile(PARAMS['INFILE'])

        self.intree = None
        try:
            self.intree = self.infile.Get("TrackWalkerInput_Pad_010")
        except:
            self.intree = self.infile.Get("TrackWalkerInput") #Legacy format

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

        print("Loading Train Entries ", start_entry, "to", end_entry)
        training_data = []
        for i in range(start_entry,end_entry):
            self.intree.GetEntry(i)
            stacked_step_images = self.intree.stacked_step_images.tonumpy().copy()
            stacked_targ_idx    = self.intree.stacked_targ_idx.tonumpy().copy()
            stacked_targ_area   = self.intree.stacked_targ_area.tonumpy().copy()
            stacked_wire_images    = self.intree.stacked_wire_images.tonumpy().copy()
            flat_area_positions = unstack(stacked_targ_area)
            stacked_wire_images = np.expand_dims(stacked_wire_images,axis=3)
            stacked_step_images      = np.reshape(stacked_step_images,(-1,dim,dim,16)) #16 is nLarMatch Features


            if self.RAND_FLIP_INPUTS:
                if (np.random.randint(0,2) > 0.5): #Coin Flip
                    stacked_step_images = np.flip(stacked_step_images,axis=1)
                    stacked_targ_idx    = flip_target_idx_xdim(stacked_targ_idx,self.PARAMS)
                    flat_area_positions = flip_flat_area_positions_xdim(flat_area_positions, self.PARAMS)
                    stacked_wire_images = np.flip(stacked_wire_images,axis=1)

            stepped_images = stacked_step_images
            if self.PARAMS['APPEND_WIREIM']:
                stepped_images = np.concatenate((stepped_images,stacked_wire_images),axis=3)

            flat_next_positions = unstack(stacked_targ_idx)
            training_data.append((stepped_images,flat_next_positions,flat_area_positions))

            # from MiscFunctions import save_im
            # import os
            # for f in range(16):
            #     save_im(stepped_images[0,:,:,f],savename='larmatchfeat_im_test/larmatch_feat_'+str(f).zfill(2),canv_x=1000,canv_y=1000)
            # convert_cmd = "convert "+"larmatchfeat_im_test/*.png "+'larmatchfeat_im_test/featcheck.pdf'
            # print(convert_cmd)
            # os.system(convert_cmd)
            # assert 1==2

        self.current_train_entry = end_entry
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
        for i in range(start_entry+self.nentry_val_buffer,end_entry+self.nentry_val_buffer):
            self.intree.GetEntry(i)
            stacked_step_images = self.intree.stacked_step_images.tonumpy()
            stacked_targ_idx    = self.intree.stacked_targ_idx.tonumpy()
            stacked_targ_area   = self.intree.stacked_targ_area.tonumpy()
            stacked_wire_images    = self.intree.stacked_wire_images.tonumpy().copy()
            flat_area_positions = unstack(stacked_targ_area)
            stacked_wire_images = np.expand_dims(stacked_wire_images,axis=3)

            if self.RAND_FLIP_INPUTS:
                if (np.random.randint(0,2) > 0.5): #Coin Flip
                    stacked_step_images = np.flip(stacked_step_images,axis=0)
                    stacked_targ_idx    = flip_target_idx_xdim(stacked_targ_idx,self.PARAMS)
                    flat_area_positions = flip_flat_area_positions_xdim(flat_area_positions, self.PARAMS)
                    stacked_wire_images = np.flip(stacked_wire_images,axis=1)


            stepped_images = np.reshape(stacked_step_images,(-1,dim,dim,16)) #16 is nLarMatch Features
            if self.PARAMS['APPEND_WIREIM']:
                stepped_images = np.concatenate((stepped_images,stacked_wire_images),axis=3)
            flat_next_positions = unstack(stacked_targ_idx)
            val_data.append((stepped_images,flat_next_positions,flat_area_positions))

        self.current_val_entry = end_entry
        if self.current_val_entry == self.nentries_val:
            self.current_val_entry = 0
        return val_data


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
