import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from MiscFunctions import unflatten_pos, flatten_pos, make_steps_images
from VoxelFunctions import Voxelator


def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

class DataLoader3D:
    def __init__(self, PARAMS, verbose=False, all_train = False, all_valid = False, LArVoxMode=False):
        self.PARAMS = PARAMS
        self.LArVoxMode = LArVoxMode
        self.voxelator = Voxelator(self.PARAMS) if not self.LArVoxMode else Voxelator(self.PARAMS, "LARVOXNETMICROBOONE")
        self.verbose = verbose
        self.infile = None
        if all_train:
            self.infile = ROOT.TFile(PARAMS['INFILE_TRAIN'])
        elif all_valid:
            self.infile = ROOT.TFile(PARAMS['INFILE_VAL'])
        else:
            self.infile = ROOT.TFile(PARAMS['INFILE'])

        self.intree = self.infile.Get("TrackWalker3DInput") if not self.LArVoxMode else self.infile.Get("TrackWalker3DVoxInput")
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
            featcrops_np_v, xyzShifts_np_v, voxelsteps_final_np_v, isOK = self.get_data(i)
            if isOK == False:
                continue

            training_data.append((featcrops_np_v,xyzShifts_np_v,voxelsteps_final_np_v))
            nAdded += 1

        self.current_train_entry = i+1
        print(self.current_train_entry)
        if self.current_train_entry == self.nentries_train:
            self.current_train_entry = 0
        return training_data


    def get_data(self, i):
        self.intree.GetEntry(i)


        # np array of x,y,z,StepIDX in full detector voxel coord
        voxelsteps_np       = self.intree.voxelsteps_np.tonumpy().copy()
        # Min Row, Min Cols for the feature images (to offset vox projection)
        minVox_np       = self.intree.minVoxCoords_np.tonumpy().astype(np.int32).copy()
        maxVox_np       = self.intree.maxVoxCoords_np.tonumpy().astype(np.int32).copy()
        # In (xVox,yVox,zVox,32 Feats) a sparse array of feats around the given track
        sparse_feats_np = self.intree.feats_np.tonumpy().copy()

        # if there is only 1 or fewer points on the index map then dont include track
        if voxelsteps_np.shape[0] < 2 or len(sparse_feats_np.shape) != 2:
            print("Skipping issue with Saved ROOT Track Shape")
            return None, None, None, 0

        featcrops_np_v, xyzShifts_np_v, voxelsteps_final_np_v = \
            self.make_track_crops(sparse_feats_np, voxelsteps_np, minVox_np, maxVox_np, self.PARAMS)

        if len(featcrops_np_v) < 2:
            print("Skipping issue with Saved ROOT Track Shape after making track crops")
            return None, None, None, 0

        if self.RAND_FLIP_INPUTS:
            print("Random flipping of images not implemented for complex dataloading")
            assert 1==2

        return featcrops_np_v, xyzShifts_np_v, voxelsteps_final_np_v, 1

    def set_current_entry(self,entry):
        self.current_train_entry = entry
        self.current_val_entry   = entry

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

            featcrops_np_v, xyzShifts_np_v, voxelsteps_final_np_v, isOK = self.get_data(i)
            if isOK == False:
                continue


            val_data.append((featcrops_np_v,xyzShifts_np_v))
            nAdded += 1

        self.current_val_entry = i+1
        if self.current_val_entry == self.nentries_val:
            self.current_val_entry = 0
        return val_data



    def make_track_crops(self, sparse_feats_np, voxelsteps_np, minVox_np, maxVox_np, PARAMS):
        # print("Printing True Track")
        # for idx in range(voxelsteps_np.shape[0]):
        #     print(voxelsteps_np[idx,:])
        feat_idx = sparse_feats_np[:,0:3].copy().astype(np.int32)
        feat_steps_np_v        = []
        xyzShifts_np_v         = []
        noshiftVoxelIdx_np_v   = []
        shiftVoxelIdx_np_v     = []
        targVoxelIdx_np_v      = []
        flattened_positions_v  = []
        area_positions_v       = []
        nSteps = 0
        # Get First and Last Position
        startVoxelPosition = voxelsteps_np[0,:].copy()
        endVoxelPosition   = voxelsteps_np[-1,:].copy()
        thisVoxelPosition  = startVoxelPosition.copy()
        shiftedthisVoxelPosition = thisVoxelPosition.copy()
        # Repopulate the part of the detector we saved for this track:
        denseFeatPartialDetector = np.zeros((maxVox_np[0]-minVox_np[0],maxVox_np[1]-minVox_np[1],maxVox_np[2]-minVox_np[2],PARAMS['NFEATS']))
        denseFeatPartialDetector[feat_idx[:,0]-minVox_np[0],feat_idx[:,1]-minVox_np[1],feat_idx[:,2]-minVox_np[2]] = sparse_feats_np[:,3:]

        nextStepIdx = 0
        voxelsteps_np_idx = 0
        ct = 0
        while np.array_equal(shiftedthisVoxelPosition[0:3],endVoxelPosition[0:3]) != True:
            ct +=1
            noshiftVoxelIdx_np_v.append(shiftedthisVoxelPosition)
            if PARAMS['DO_CROPSHIFT']:
                # for i in range(3):
                #     shift_amt = PARAMS['CROPSHIFT_MAXAMT']
                #     delta = np.random.randint(-shift_amt, shift_amt+1)
                #     shiftedthisVoxelPosition[i] +=delta
                shiftedthisVoxelPosition[np.random.randint(0,3)] += np.random.randint(-PARAMS['CROPSHIFT_MAXAMT'],PARAMS['CROPSHIFT_MAXAMT']+1)


            cropMins   = [int(shiftedthisVoxelPosition[i]-PARAMS['PADDING']-minVox_np[i])   for i in range(3)]
            cropMaxes  = [int(shiftedthisVoxelPosition[i]+PARAMS['PADDING']+1-minVox_np[i]) for i in range(3)]
            currentFeatCrop = denseFeatPartialDetector[cropMins[0]:cropMaxes[0],cropMins[1]:cropMaxes[1],cropMins[2]:cropMaxes[2]].copy()
            if currentFeatCrop.shape != (PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1,PARAMS['PADDING']*2+1,PARAMS['NFEATS']):
                print()
                print("Failure to Crop correct shape of features")
                print(shiftedthisVoxelPosition)
                print(currentFeatCrop.shape)
                print(denseFeatPartialDetector.shape)
                print(cropMins[0],cropMaxes[0],cropMins[1],cropMaxes[1],cropMins[2],cropMaxes[2])
                assert 1==2
            # next step should be:
            #        within cropMins and maxes,
            #        less than or equal to PARAMS['TARG_STEP_DIST']
            #        The highest value in the voxelsteps_np[:,3] 'rank' possible
            currentStepDist = 0
            nextVoxelStep = np.array([-1.,-1.,-1.,-1.])
            for stepIdx in range(voxelsteps_np_idx,voxelsteps_np.shape[0]):
                testStep = voxelsteps_np[stepIdx,:].copy()
                dist = ((shiftedthisVoxelPosition[0] - testStep[0])**2 + (shiftedthisVoxelPosition[1] - testStep[1])**2 + (shiftedthisVoxelPosition[2] - testStep[2])**2)**0.5
                # print("    ", testStep,end='')
                if dist <= PARAMS['TARG_STEP_DIST'] and testStep[3] > nextVoxelStep[3]:
                    nextVoxelStep = testStep
                    nextStepIdx   = stepIdx
                    # print(nextVoxelStep, "Next Step Set",end='')
                # print()



            xyzShift = nextVoxelStep[0:3].copy() - shiftedthisVoxelPosition[0:3].copy()
            # print(xyzShift)
            # if np.sum(xyzShift*xyzShift)**0.5 < 2:
                # print()
                # print("Stepping:",np.sum(xyzShift*xyzShift)**0.5)
                # print("Starting Debug")
                # print("    ",shiftedthisVoxelPosition,"Starting Position")
                # currentStepDist = 0
                # nextVoxelStep = np.array([-1.,-1.,-1.,-1.])
                # for stepIdx in range(voxelsteps_np_idx,voxelsteps_np.shape[0]):
                #     testStep = voxelsteps_np[stepIdx,:].copy()
                #     dist = ((shiftedthisVoxelPosition[0] - testStep[0])**2 + (shiftedthisVoxelPosition[1] - testStep[1])**2 + (shiftedthisVoxelPosition[2] - testStep[2])**2)**0.5
                #     # print("    ", testStep,end='')
                #     if dist <= PARAMS['TARG_STEP_DIST'] and testStep[3] > nextVoxelStep[3]:
                #         nextVoxelStep = testStep
                #         nextStepIdx   = stepIdx
                #         print("        ",testStep,"Test Step",dist, " Target Position Set")
                #     else:
                #         print("        ",testStep,"Test Step",dist)

                # assert 1==2


            xyzShift = xyzShift/PARAMS['CONVERT_OUT_TO_DIST']
            feat_steps_np_v.append(currentFeatCrop)
            xyzShifts_np_v.append(xyzShift)
            targVoxelIdx_np_v.append(nextVoxelStep)
            shiftVoxelIdx_np_v.append(shiftedthisVoxelPosition)
            # Change current position to next position
            shiftedthisVoxelPosition = nextVoxelStep

        cropMins   = [int(endVoxelPosition[i]-PARAMS['PADDING']-minVox_np[i])   for i in range(3)]
        cropMaxes  = [int(endVoxelPosition[i]+PARAMS['PADDING']+1-minVox_np[i]) for i in range(3)]
        feat_steps_np_v.append(denseFeatPartialDetector[cropMins[0]:cropMaxes[0],cropMins[1]:cropMaxes[1],cropMins[2]:cropMaxes[2]].copy())
        xyzShifts_np_v.append(np.zeros((3,)).astype(np.float32))
        targVoxelIdx_np_v.append(endVoxelPosition)
        shiftVoxelIdx_np_v.append(endVoxelPosition)

        feat_steps_np_v = np.stack(feat_steps_np_v,axis=0)
        try:
            xyzShifts_np_v = np.stack(xyzShifts_np_v,axis=0)
        except:
            for xyz in xyzShifts_np_v:
                print(xyz)
            assert 1==2
        # print("Printing Track Info")
        # for idx in range(len(targVoxelIdx_np_v)):
        #     print(shiftVoxelIdx_np_v[idx], targVoxelIdx_np_v[idx], xyzShifts_np_v[idx]*PARAMS['CONVERT_OUT_TO_DIST'], np.sum(xyzShifts_np_v[idx]*xyzShifts_np_v[idx])**0.5*PARAMS['CONVERT_OUT_TO_DIST'])
        # assert 1==2
        return feat_steps_np_v, xyzShifts_np_v, shiftVoxelIdx_np_v


    def getNextStepClass(self, nextStepIdx_v, dim, isEndpoint=False ):
        if isEndpoint == True:
            val = int((dim-1)/2)
            return int(val*dim*dim + val*dim + val) #13, refers to center of cube for endpoint
        else:
            classVal = nextStepIdx_v[0]*dim*dim + nextStepIdx_v[1]*dim + nextStepIdx_v[2]
            return int(classVal)




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

def row_get(y, origin_y, origin_y_plus_height, pixel_height):
    if ((y < origin_y) or (y >= origin_y_plus_height)):
        print("Row out of range", y, origin_y, origin_y_plus_height)
        assert 1==2
    else:
        return int((y-origin_y)/pixel_height)

def col_get(x, origin_x, origin_x_plus_width, pixel_width):
    if ((x < origin_x) or (x >= origin_x_plus_width)):
        print("Row out of range", x, origin_x, origin_x_plus_width)
        assert 1==2
    else:
        return int((x-origin_x)/pixel_width)

def getprojectedpixel_hardcoded(x,y,z):

    nplanes = 3
    fracpixborder = 1.5
    pixel_height = 6.0
    pixel_width  = 1.0
    DriftVelocity = 0.1098
    SamplingRate = 500.0
    min_y        = 2400.0
    max_y        = 8448.0
    rows         = 1008
    origin_y     = 2400.0
    origin_y_plus_height = 8448.0
    min_x = 0.0
    max_x = 3456.0
    origin_x = 0.0
    origin_x_plus_width = 3456.0
    cols  = 3456

    row_border = fracpixborder*pixel_height;
    col_border = fracpixborder*pixel_width;

    img_coords = [-1,-1,-1,-1]
    tick = x/(DriftVelocity*SamplingRate*1.0e-3) + 3200.0;
    if ( tick < min_y ):
        if ( tick > min_y- row_border ):
            # below min_y-border, out of image
            img_coords[0] = rows-1 # note that tick axis and row indicies are in inverse order (same order in larcv2)
        else:
            # outside of image and border
            img_coords[0] = -1
    elif ( tick > max_y ):
        if (tick < max_y+row_border):
            # within upper border
            img_coords[0] = 0;
        else:
            # outside of image and border
            img_coords[0] = -1;

    else:
        # within the image
        img_coords[0] = col_get(tick,origin_y,origin_y_plus_height,pixel_height);


    # Columns
    # xyz = [ x, y, z ]
    xyz = array('d', [x,y,z])

    # there is a corner where the V plane wire number causes an error
    if ( (y>-117.0 and y<-116.0) and z<2.0 ):
        xyz[1] = -116.0;

    for p in range(nplanes):
        wire = larutil.Geometry.GetME().WireCoordinate( xyz, p );

        # get image coordinates
        if ( wire<min_x ):
            if ( wire>min_x-col_border ):
                # within lower border
                img_coords[p+1] = 0;
            else:
                img_coords[p+1] = -1;
        elif ( wire>=max_x ):
            if ( wire<max_x+col_border ):
                # within border
                img_coords[p+1] = cols-1
            else:
                # outside border
                img_coords[p+1] = -1
        else:
        # inside image
            img_coords[p+1] = col_get(wire,origin_x,origin_x_plus_width,pixel_width) #meta.col( wire );
        # end of plane loop

    # there is a corner where the V plane wire number causes an error
    if ( y<-116.3 and z<2.0 and img_coords[1+1]==-1 ):
        img_coords[1+1] = 0;


    return img_coords
