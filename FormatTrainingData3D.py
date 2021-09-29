import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
# from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import get_loss_weights_v2, unflatten_pos, calc_logger_stats
from MiscFunctions import make_log_stat_dict, reravel_array, make_prediction_vector
from MiscFunctions import blockPrint, enablePrint
from FancyLoader3D import FancyLoader3D
from ModelFunctions import LSTMTagger, run_validation_pass
from VoxelFunctions import Voxelator
import random
import ROOT
from array import array
from larcv import larcv
import os
import argparse




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

PARAMS['USE_CONV_IM'] = True
# PARAMS['LARMATCH_CKPT'] = '/home/jmills/workdir/TrackWalker/larmatch_ckpt/checkpoint.1974000th.tar'
PARAMS['LARMATCH_CKPT'] = '/home/jmills/workdir/TrackWalker/larmatch_ckpt/checkpoint.1974000th.tar'
PARAMS['MASK_WC'] = False

PARAMS['MIN_TRACK_LENGTH'] = 3.0
PARAMS['HIDDEN_DIM'] =1024
PARAMS['PADDING'] =10
PARAMS['EMBEDDING_DIM'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) # N_Features
if PARAMS['USE_CONV_IM']:
    PARAMS['EMBEDDING_DIM'] = PARAMS['EMBEDDING_DIM']*16 # 16 Features per pixel in larmatch
PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1)+1 # Bonus Class is for the end of track class
PARAMS['TRACKEND_CLASS'] = (PARAMS['PADDING']*2+1)**2
PARAMS['CENTERPOINT_ISEND'] = True
if PARAMS['CENTERPOINT_ISEND']:
     PARAMS['NUM_CLASSES'] =(PARAMS['PADDING']*2+1)*(PARAMS['PADDING']*2+1) #No bonus end of track class
     PARAMS['TRACKEND_CLASS'] = (PARAMS['NUM_CLASSES']-1)/2
# PARAMS['INFILE'] ="/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
PARAMS['INFILE'] ="/cluster/tufts/wongjiradlab/jmills09/TrackerNet_InputData/midlevel_files/small_00002_merged_dlreco_497be540-00f9-49a8-9f80-7846143c4fce.root"
PARAMS['TRACK_IDX'] =0
PARAMS['EVENT_IDX'] =0
PARAMS['ALWAYS_EDGE'] =True # True points are always placed at the edge of the Padded Box
# TENSORDIR'] ="runs/Pad20_Hidden1024_500Entries"
PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER'] =True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
PARAMS['NDIMENSIONS'] = 2 #Not configured to have 3 yet.
PARAMS['LEARNING_RATE'] =0.0001 # 0.01 is good for the classifier mode,
PARAMS['DO_TENSORLOG'] = True
PARAMS['TENSORDIR']  = None # Default runs/DATE_TIME
PARAMS['SAVE_MODEL'] = False #should the network save the model?
PARAMS['CHECKPOINT_EVERY_N_EPOCHS'] =10000 # if not saving then this doesn't matter
PARAMS['EPOCHS'] = 5
PARAMS['VALIDATION_EPOCH_LOGINTERVAL'] = 1
PARAMS['VALIDATION_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['TRAIN_EPOCH_LOGINTERVAL'] = 1
PARAMS['TRAIN_TRACKIDX_LOGINTERVAL'] = 100
PARAMS['DEVICE'] = 'cuda:0'
# PARAMS['DEVICE'] = 'cpu'
PARAMS['LOAD_SIZE']  = 50
PARAMS['TRAIN_EPOCH_SIZE'] = 500
PARAMS['VAL_EPOCH_SIZE'] = int(0.8*PARAMS['TRAIN_EPOCH_SIZE'])

PARAMS['SHUFFLE_DATASET'] = False
PARAMS['VAL_IS_TRAIN'] = False # This will set the validation set equal to the training set
PARAMS['AREA_TARGET'] = True   # Change network to be predicting
PARAMS['TARGET_BUFFER'] = 2

isRequired = False
def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--infile', dest='infile', required=isRequired,
        help='infile')

    parser.add_argument(
        '--infileidx', dest='infileidx', required=isRequired,
        help='infileidxeidx')

    parser.add_argument(
        '--folderidx', dest='folderidx', required=isRequired,
        help='folderidx')

    parser.add_argument(
        '--outdir', dest='outdir', required=isRequired,
        help='outdir')

    return parser.parse_args()


def main():
    # test with:
    # python FormatTrainingData_Complex.py --infileidx 0 --outdir test/ --folderidx 0 --infile /cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data/00/01/71/93/merged_dlreco_b1379c2b-cd47-4140-a448-3a27e8f28fb8.root
    print("Let's Get Started.")

    args = parse_args()


    # args.infile       = "inputfiles/VAL_BnBOverLay_DLReco_Tracker.root"
    # args.outdir       = "TEST3DReformat/"
    # args.folderidx    = str(0)
    # args.infileidx    = "TESTSPARSEALL"#str()

    print('Called with args:')
    print(args)
    PARAMS['INFILE'] =args.infile

    nbins = PARAMS['PADDING']*2+1

    # Create Outfile stuff:
    outfile_str = "Wire_" if PARAMS['USE_CONV_IM'] == False else "LArMatch_"
    outfile_str = outfile_str + "ComplexTrackIdx_"
    outfile_str = args.outdir+args.folderidx+"/Reformat_" + outfile_str
    if not os.path.exists(args.outdir+args.folderidx):
        os.makedirs(args.outdir+args.folderidx)


    tree    = ROOT.TTree("TrackWalker3DInput","TrackWalker 3D Input")

    feats_u_np = larcv.NumpyArrayFloat()
    tree.Branch("feats_u_np",feats_u_np)

    feats_v_np = larcv.NumpyArrayFloat()
    tree.Branch("feats_v_np",feats_v_np)

    feats_y_np = larcv.NumpyArrayFloat()
    tree.Branch("feats_y_np",feats_y_np)

    feat_shapes_np = larcv.NumpyArrayFloat()
    tree.Branch("feat_shapes_np",feat_shapes_np)

    voxelsteps_np = larcv.NumpyArrayFloat()
    tree.Branch("voxelsteps_np",voxelsteps_np)

    originInFullImg_np = larcv.NumpyArrayFloat()
    tree.Branch("originInFullImg_np",originInFullImg_np)

    charge_in_wires_np = larcv.NumpyArrayFloat()
    tree.Branch("charge_in_wires_np",charge_in_wires_np)

    charge_in_truths_np = larcv.NumpyArrayFloat()
    tree.Branch("charge_in_truths_np",charge_in_truths_np)

    entry = array('f',[0]) # single float
    tree.Branch("original_entry",entry,"original_entry/F")
    mctrack_idx = array('f',[0]) # single float
    tree.Branch("mctrack_idx",mctrack_idx,"mctrack_idx/F")
    mctrack_length = array('f',[0]) # single float
    tree.Branch("mctrack_length",mctrack_length,"mctrack_length/F")
    mctrack_pdg = array('f',[0]) # single float
    tree.Branch("mctrack_pdg",mctrack_pdg,"mctrack_pdg/F")
    mctrack_energy = array('f',[0]) # single float
    tree.Branch("mctrack_energy",mctrack_energy,"mctrack_energy/F")
    run = array('f',[0]) # single float
    tree.Branch("run",run,"run/F")
    subrun = array('f',[0]) # single float
    tree.Branch("subrun",subrun,"subrun/F")
    event_id = array('f',[0]) # single float
    tree.Branch("event_id",event_id,"event_id/F")

    # entry_per_file = 9999
    entry_per_file = 9999
    Loader = FancyLoader3D(PARAMS)
    # iter = int(tot_entries/entry_per_file)+1
    startEntry = 0
    Loader.currentEntry = startEntry
    endEntry = Loader.nentries_ll
    # endEntry = 2
    for i in range(startEntry,endEntry):

        loadingDict = Loader.load_fancy()

        # from MiscFunctions import save_im
        # save_im(wire_image_v[0],     savename="test/wireimg",canv_x=4000,canv_y=1000)
        # save_im(step_idx_image_v[0], savename="test/stepimg",canv_x=4000,canv_y=1000)
        # voxelator = Voxelator(PARAMS)
        # voxelator.saveDetector3D(loadingDict['voxSteps_vv'][0])

        print("Number of Tracks:", len(loadingDict["features_image_vv"]))
        for idxx in range(len(loadingDict["features_image_vv"])):
            featShapes = np.zeros((3,2))
            sparseFeats_v = []
            for p in range(3):
                nonzeroIdx = np.nonzero(loadingDict['features_image_vv'][idxx][p])
                featShapes[p,0] = loadingDict['features_image_vv'][idxx][p].shape[0]
                featShapes[p,1] = loadingDict['features_image_vv'][idxx][p].shape[1]
                nFeats = nonzeroIdx[0].shape[0]
                print(nFeats,"nFeats")
                sFeats = np.zeros((nFeats,4)) #x,y,nfeat,val
                for iii in range(nFeats):
                    sFeats[iii,0] = nonzeroIdx[0][iii]
                    sFeats[iii,1] = nonzeroIdx[1][iii]
                    sFeats[iii,2] = nonzeroIdx[2][iii]
                    sFeats[iii,3] = loadingDict['features_image_vv'][idxx][p][nonzeroIdx[0][iii],nonzeroIdx[1][iii],nonzeroIdx[2][iii]]
                print(sFeats.shape)
                sparseFeats_v.append(sFeats)


            # print("Filling Tree", idxx)
            #
            # print("\n\nChecking Sizes")
            # print("    Planes:")
            # for p in range(3):
            #     print("        ",p)
            #     print(loadingDict['features_image_vv'][idxx][p].shape, "shape cropped features")
            #     nCroppedFeats = loadingDict['features_image_vv'][idxx][p].shape[0]* loadingDict['features_image_vv'][idxx][p].shape[1]*loadingDict['features_image_vv'][idxx][p].shape[2]
            #     print(nCroppedFeats, "Number shape cropped features")
            #     nSparseFeats = np.sum(np.where(loadingDict['features_image_vv'][idxx][p]!=0,1,0))
            #     print(nSparseFeats, "Number nonzero idx")
            #     print(nSparseFeats*1.0/nCroppedFeats, "Ratio Saved")
            #
            # stepsShape = loadingDict['voxSteps_vv'][idxx].shape
            # print(stepsShape, "Step Shape")
            # print(stepsShape[0]*stepsShape[1], "Step Numbers")
            # print()
            # imcoordShape = loadingDict['minImgCoords_v'][idxx].shape
            # print(imcoordShape, "ImCoord Shape")
            # print("\n\n")

            # feats_u_np.store(loadingDict['features_image_vv'][idxx][0].astype(np.float32))
            # feats_v_np.store(loadingDict['features_image_vv'][idxx][1].astype(np.float32))
            # feats_y_np.store(loadingDict['features_image_vv'][idxx][2].astype(np.float32))

            feats_u_np.store(sparseFeats_v[0].astype(np.float32))
            feats_v_np.store(sparseFeats_v[1].astype(np.float32))
            feats_y_np.store(sparseFeats_v[2].astype(np.float32))
            feat_shapes_np.store(featShapes.astype(np.float32))

            voxelsteps_np.store(loadingDict['voxSteps_vv'][idxx].astype(np.float32))
            originInFullImg_np.store(loadingDict['minImgCoords_v'][idxx].astype(np.float32))
            charge_in_wires_np.store(loadingDict['charge_in_wires_v'][idxx].astype(np.float32))
            charge_in_truths_np.store(loadingDict['charge_in_truths_v'][idxx].astype(np.float32))
            entry[0]          = loadingDict['entry_v'][idxx]
            mctrack_idx[0]    = loadingDict['mctrack_idx_v'][idxx]
            mctrack_length[0] = loadingDict['mctrack_length_v'][idxx]
            mctrack_pdg[0]    = loadingDict['mctrack_pdg_v'][idxx]
            mctrack_energy[0] = loadingDict['mctrack_energy_v'][idxx]

            run[0]            = loadingDict['run_v'][idxx]
            subrun[0]         = loadingDict['subrun_v'][idxx]
            event_id[0]       = loadingDict['eventid_v'][idxx]

            tree.Fill()
            idxx += 1
    print("Writing File:")
    print(outfile_str+str(args.infileidx).zfill(3)+".root")
    outfile = ROOT.TFile(outfile_str+str(args.infileidx).zfill(3)+".root","recreate")
    outfile.cd()
    tree.Write()
    outfile.Close()
    return 0

if __name__ == '__main__':
    main()
