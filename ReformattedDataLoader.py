import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from LArMatchModel import LArMatchConvNet

def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

class ReformattedDataLoader_MC:
    def __init__(self, PARAMS, verbose=False, all_train = False):
        self.PARAMS = PARAMS
        self.verbose = verbose

        self.infile = ROOT.TFile("/home/jmills/workdir/TrackWalker/inputfiles/ReformattedInput/Reformat_LArMatch_Pad_010.root")
        self.intree = self.infile.Get("TrackWalkerInput")

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
            stepped_images = np.reshape(stacked_step_images,(-1,21,21,16))
            flat_next_positions = unstack(stacked_targ_idx)
            flat_area_positions = unstack(stacked_targ_area)
            training_data.append((stepped_images,flat_next_positions,flat_area_positions))
        self.current_train_entry = end_entry
        if self.current_train_entry == self.nentries_train:
            self.current_train_entry = 0
        return training_data

    def get_val_data(self, n_load):
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

            stepped_images = np.reshape(stacked_step_images,(-1,21,21,16))
            flat_next_positions = unstack(stacked_targ_idx)
            flat_area_positions = unstack(stacked_targ_area)
            val_data.append((stepped_images,flat_next_positions,flat_area_positions))
        self.current_val_entry = end_entry
        if self.current_val_entry == self.nentries_val:
            self.current_val_entry = 0
        return val_data
