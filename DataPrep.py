import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp

HIDDEN_DIM = 1024
PADDING = 20
EMBEDDING_DIM = (PADDING*2+1)*(PADDING*2+1) # N_Features
NUM_CLASSES = (PADDING*2+1)*(PADDING*2+1)+1 # Bonus Class is for the end of track class
INFILE = "/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
ALWAYS_EDGE = True # True points are always placed at the edge of the Padded Box
# TENSORDIR = "runs/TESTRUN"
CLASSIFIER_NOT_DISTANCESHIFTER = True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
NDIMENSIONS = 2 #Not configured to have 3 yet.

def main():
    outfile = ROOT.TFile("out.root","RECREATE")
    outtree = ROOT.TTree("example","Example output tree")
    cropped_steps_np = larcv.NumpyArrayFloat()
    next_steps_np    = larcv.NumpyArrayFloat()
    cropped_steps_flat_np    = larcv.NumpyArrayFloat()
    next_steps_flat_np       = larcv.NumpyArrayFloat()
    run = array('i',[0])
    subrun = array('i',[0])
    event = array('i',[0])
    entry_idx = array('i',[0])
    track_pdg = array('i',[0])
    # Need to write these things, tutorial here: https://github.com/LArbys/ubdl/blob/master/Tutorials/saving_with_root_numpy.ipynb
    # and here: https://pep-root6.github.io/docs/analysis/python/pyroot.html


    outtree.Branch("data",data)

    input_image_dimension = PADDING*2+1
    steps_x = []
    steps_y = []
    full_image = []
    training_data = []
    event_ids = []
    step_dist_3d = []
    image_list,xs,ys, runs, subruns, events, filepaths, entries, track_pdgs = load_rootfile(INFILE, step_dist_3d)
    print()
    print()
    # save_im(image_list[EVENT_IDX],"images/EventDisp")
    for EVENT_IDX in range(len(image_list)):
        print("Doing Event:", EVENT_IDX)
        print("N MC Tracks:", len(xs[EVENT_IDX]))
        for TRACK_IDX in range(len(xs[EVENT_IDX])):
            # if TRACK_IDX != 0:
            #     continue
            print("     Doing Track:", TRACK_IDX)

            full_image = image_list[EVENT_IDX]
            steps_x = xs[EVENT_IDX][TRACK_IDX]
            steps_y = ys[EVENT_IDX][TRACK_IDX]

            print("         Original Track Points", len(steps_x))
            new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,PADDING,always_edge=ALWAYS_EDGE)
            steps_x = new_steps_x
            steps_y = new_steps_y

            print("         After Inserted Track Points", len(steps_x))
            if len(steps_x) == 0: #Don't  include tracks without points.
                continue
            # save_im(full_image,'file',canv_x = 4000, canv_y = 1000) # If you want to save the full image as an event_display

            # Many of the following categories are just a reformatting of each other
            # They are duplicated to allow for easy network mode switching
            stepped_images = [] # List of cropped images as 2D numpy array
            flat_stepped_images = [] # list of cropped images as flattened 1D np array
            next_positions = [] # list of next step positions as np(x,y)
            flat_next_positions = [] # list of next step positions in flattened single coord idx
            xy_shifts = [] # list of X,Y shifts to take the next step
            for idx in range(len(steps_x)):
                # if idx > 1:
                #     continue
                step_x = steps_x[idx]
                step_y = steps_y[idx]
                next_step_x = -1.0
                next_step_y = -1.0
                if idx != len(steps_x)-1:
                    next_step_x = steps_x[idx+1]
                    next_step_y = steps_y[idx+1]
                cropped_step_image = cropped_np(full_image, step_x, step_y, PADDING)
                required_padding_x = PADDING - step_x
                required_padding_y = PADDING - step_y
                stepped_images.append(cropped_step_image)
                flat_stepped_images.append(unravel_array(cropped_step_image))
                flat_next_positions_array = np.zeros(input_image_dimension*input_image_dimension+1)
                if idx != len(steps_x)-1:
                    target_x = required_padding_x + next_step_x
                    target_y = required_padding_y + next_step_y
                    np_step_target = np.array([target_x*1.0,target_y*1.0])
                    flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                    next_positions.append(np_step_target)
                    flat_next_positions.append(flat_np_step_target)
                    np_xy_shift = np.array([target_x*1.0-PADDING,target_y*1.0-PADDING ])
                    xy_shifts.append(np_xy_shift)
                else:
                    next_positions.append(np.array([-1.0,-1.0]))
                    flat_next_positions.append(NUM_CLASSES-1)
                    np_xy_shift = np.array([0.0,0.0])
                    xy_shifts.append(np_xy_shift)
            if CLASSIFIER_NOT_DISTANCESHIFTER:
                training_data.append((flat_stepped_images,flat_next_positions))
                event_ids.append(EVENT_IDX)
            else:
                training_data.append((flat_stepped_images,xy_shifts))
                event_ids.append(EVENT_IDX)


if __name__ == '__main__':
    main()
