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
    def __init__(self, PARAMS, verbose=False, all_train = False, all_valid = False):
        self.PARAMS = PARAMS
        self.voxelator = Voxelator(self.PARAMS)
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
            # np array of x,y,z,StepIDX in full detector voxel coord
            voxelsteps_np       = self.intree.voxelsteps_np.tonumpy().copy()
            # Min Row, Min Cols for the feature images (to offset vox projection)
            originInFullImg_np       = self.intree.originInFullImg_np.tonumpy().copy()

            # if there is only 1 or fewer points on the index map then dont include track
            if voxelsteps_np.shape[0] < 2:
                continue
            feats_u_np          = self.intree.feats_u_np.tonumpy().copy()
            feats_v_np          = self.intree.feats_v_np.tonumpy().copy()
            feats_y_np          = self.intree.feats_y_np.tonumpy().copy()


            feature_ims_np_v, flat_next_positions, flat_area_positions = \
                self.make_track_crops([feats_u_np,feats_v_np,feats_y_np], voxelsteps_np, originInFullImg_np, self.PARAMS)
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
            # np array of x,y,z,StepIDX in full detector voxel coord
            voxelsteps_np       = self.intree.voxelsteps_np.tonumpy().copy()
            # Min Row, Min Cols for the feature images (to offset vox projection)
            originInFullImg_np       = self.intree.originInFullImg_np.tonumpy().copy()

            # if there is only 1 or fewer points on the index map then dont include track
            if voxelsteps_np.shape[0] < 2:
                continue
            feats_u_np          = self.intree.feats_u_np.tonumpy().copy()
            feats_v_np          = self.intree.feats_v_np.tonumpy().copy()
            feats_y_np          = self.intree.feats_y_np.tonumpy().copy()


            feature_ims_np_v, flat_next_positions, flat_area_positions = \
                self.make_track_crops([feats_u_np,feats_v_np,feats_y_np], voxelsteps_np, originInFullImg_np, self.PARAMS)

            if self.RAND_FLIP_INPUTS:
                print("Random flipping of images not implemented for complex dataloading")
                assert 1==2
            val_data.append((feature_ims_np_v,flat_next_positions,flat_area_positions))
            nAdded += 1

        self.current_val_entry = i+1
        if self.current_val_entry == self.nentries_val:
            self.current_val_entry = 0
        return val_data



    def make_track_crops(self, feat_im_v, voxelsteps_np, originInFullImg_np, PARAMS):
        feat_steps_np_v = []
        flattened_positions_v  = []
        area_positions_v  = []

        nSteps = 0

        # nextfullimPosition  = np.where(stepidx_im_np == np.ma.masked_equal(stepidx_im_np, 0.0, copy=False).min())
        # endPosition         = np.where(stepidx_im_np == np.amax(stepidx_im_np))
        # lastPosition = -1

        # nextfullvoxelPosition   = voxelsteps_np[ 0,:].copy()
        # endPosition             = voxelsteps_np[-1,:].copy()

        # print(nextfullvoxelPosition)
        # print(endPosition)
        # lastPosition = -1

        # endIdx   = np.amax(feat_im_np)
        # isFinished = False
        # while nextfullimPosition != lastPosition:
        for voxIdx in range(voxelsteps_np.shape[0]):
            thisVoxelPosition = voxelsteps_np[voxIdx,:].copy()
            shiftedthisVoxelPosition = thisVoxelPosition.copy()
            if PARAMS['DO_CROPSHIFT']:
                print("CropShifting not implemented for 3d")
                assert 1==2
                for i in range(3):
                    shift_amt = PARAMS['CROPSHIFT_MAXAMT']
                    delta = np.random.randint(-shift_amt,shift_amt+1)
                    shiftedthisVoxelPosition[i] +=delta

            # Grab Feature Images cropped around this 3d position's projections
            pos3d = self.voxelator.get3dCoord(shiftedthisVoxelPosition)
            imgcoords = getprojectedpixel_hardcoded(pos3d[0],pos3d[1],pos3d[2])
            lowcoords   = [int(imgcoords[p]-originInFullImg_np[p]-PARAMS['PADDING']) for p in range(4)]
            highcoords  = [int(imgcoords[p]-originInFullImg_np[p]+PARAMS['PADDING']+1) for p in range(4)]

            feat_crops_np    = [feat_im_v[p][lowcoords[p+1]:highcoords[p+1], lowcoords[0]:highcoords[0],:].copy() for p in range(3)]

            feat_crops_np = np.stack(feat_crops_np,axis=0)




            feat_steps_np_v.append(feat_crops_np)
            if voxIdx == voxelsteps_np.shape[0] - 1:
                flatPosition = self.getNextStepClass(None, dim=PARAMS['VOXCUBESIDE'], isEndpoint=True)
                flattened_positions_v.append(flatPosition)
                if PARAMS['AREA_TARGET']:
                    zeros_np = np.zeros((PARAMS['VOXCUBESIDE']**3))
                    zeros_np[flatPosition] = 1
                    area_positions_v.append(zeros_np)
            else:
                diffVox = voxelsteps_np[voxIdx+1,:].copy() - shiftedthisVoxelPosition + 1 #(Shift to all position vals)
                flatPosition = self.getNextStepClass(diffVox, dim=PARAMS['VOXCUBESIDE'])
                flattened_positions_v.append(flatPosition)
                if PARAMS['AREA_TARGET']:
                    zeros_np = np.zeros((PARAMS['VOXCUBESIDE']**3))
                    zeros_np[flatPosition] = 1
                    area_positions_v.append(zeros_np)
            # if PARAMS['AREA_TARGET']:
                # zeros_np = np.zeros(stepidx_crop.shape)
                # area_positions_v.append(unravel_array(paste_target(zeros_np,nextcropimPosition[0][0],nextcropimPosition[1][0],PARAMS['TARGET_BUFFER'])))

        return feat_steps_np_v, flattened_positions_v, area_positions_v

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
