import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from LArVoxelModel import LArVoxelModel
from VoxelFunctions import Voxelator

class LArVoxLoader():
    def __init__(self, PARAMS, verbose=False):
        self.PARAMS = PARAMS
        self.verbose = verbose
        self.truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
        self.iocv = None
        self.LArVoxelNet = None
        if PARAMS['USE_CONV_IM'] == False:
            self.iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
            self.iocv.set_verbosity(5)
            self.iocv.reverse_all_products() # Do I need this?
            self.iocv.add_in_file(self.PARAMS['INFILE'])
            self.iocv.initialize()
        else:
            self.LArVoxelNet = LArVoxelModel(self.PARAMS)

        # if deploy == True:
        self.iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
        self.iocv.set_verbosity(5)
        self.iocv.reverse_all_products() # Do I need this?
        self.iocv.add_in_file(self.PARAMS['INFILE'])
        self.iocv.initialize()
        self.ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
        self.ioll.add_in_filename(self.PARAMS['INFILE'])
        self.ioll.open()

        self.nentries_ll = self.ioll.get_entries()

        print()
        print("Total Events in File:        ", self.nentries_ll)
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
        self.currentEntry = 0
        self.voxelator = Voxelator(self.PARAMS,"LARVOXNETMICROBOONE")

    def load_fancy(self):
        voxfeatures_vv     = []
        voxSteps_vv        = []
        run_v              = []
        subrun_v           = []
        eventid_v          = []
        entry_v            = []
        mctrack_idx_v      = []
        mctrack_length_v   = []
        mctrack_pdg_v      = []
        mctrack_energy_v   = []
        charge_in_wires_v  = []
        charge_in_truths_v = []
        minVoxCoords_v     = []
        maxVoxCoords_v     = []

        print("Event ", self.currentEntry)
        self.iocv.read_entry(self.currentEntry)
        self.ioll.go_to(self.currentEntry)
        ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
        # Get Wire ADC Image to a Numpy Array
        meta      = None
        run       = -1
        subrun    = -1
        event     = -1
        print("NOT PERFORMING FEATS FROM LARVOX")
        feats, run, subrun, event, meta = self.LArVoxelNet.get_larvoxel_features(self.currentEntry)
        self.meta = meta
        # feats = np.random.rand(1,35)
        # feats[0,0:3] = 0
        # ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
        # anc_v = ev_ancestor.Image2DArray()

        print("Number of Tracks", len(ev_mctrack))
        #Note this is more than all the steps in mctrack, we will interpolate voxels between steps
        # Formatted as x,y,z,stepIdx (voxel coord)
        for mctk_idx in range(0,len(ev_mctrack)):
            print("  TrackNum",mctk_idx)
            # if mctk_idx != 1:
            #     continue
            voxSteps_v = []

            mctrack = ev_mctrack[mctk_idx]
            this_pdg = mctrack.PdgCode()
            if this_pdg not in self.PDG_to_Part or self.PDG_to_Part[this_pdg] not in ["PROTON","MUON","PIPLUS","PIMINUS","PI0"]:
                print("Not right particle", this_pdg)
                continue
            this_length = mctrack_length(mctrack)
            this_energy = mctrack.Start().E()
            if this_length < self.PARAMS['MIN_TRACK_LENGTH']:
                print("Too Short:",this_length)
                continue
            sce_track = self.truthtrack_SCE.applySCE(mctrack)
            xpt_list = []
            ypt_list = []
            pos3d_v, vox3d_v, minVoxCoords, maxVoxCoords = self.collectMCTrackInfo(sce_track)
            if len(vox3d_v) < 2:
                print("Not enough Steps", len(vox3d_v))
                continue

            # Now you have a list of 3D voxels for the mcsteps. Lets grab points
            # along the line segments creating as small a step as possible
            for ptidx in range(1,len(vox3d_v)):
                # print("    MCPt:",ptidx)
                last_x = vox3d_v[ptidx-1][0]
                last_y = vox3d_v[ptidx-1][1]
                last_z = vox3d_v[ptidx-1][2]
                this_x = vox3d_v[ptidx][0]
                this_y = vox3d_v[ptidx][1]
                this_z = vox3d_v[ptidx][2]
                if this_x == last_x and this_y == last_y and this_z == last_z:
                    continue
                # Add Previous Step
                # Fill in steps between [last,this) (inclusive, not inclusive)

                self.addInterpolatedSteps(voxSteps_v, last_x, last_y, last_z, this_x, this_y, this_z)

            # Add last step point
            if not (vox3d_v[-1][0] == vox3d_v[-2][0] and vox3d_v[-1][1] == vox3d_v[-2][1] and vox3d_v[-1][2] == vox3d_v[-2][2]):
                voxSteps_v.append([vox3d_v[-1][0],vox3d_v[-1][1],vox3d_v[-1][2], len(voxSteps_v)+1])

            # for ixx in range(1,len(voxSteps_v)):
            #     if abs(voxSteps_v[ixx][0] - voxSteps_v[ixx-1][0]) > 1:
            #         print(voxSteps_v[ixx-1][:])
            #         print(voxSteps_v[ixx][:])
            #         assert 1==2
            #     if abs(voxSteps_v[ixx][1] - voxSteps_v[ixx-1][1]) > 1:
            #         print(voxSteps_v[ixx-1][:])
            #         print(voxSteps_v[ixx][:])
            #         assert 1==2
            #     if abs(voxSteps_v[ixx][2] - voxSteps_v[ixx-1][2]) > 1:
            #         print(voxSteps_v[ixx-1][:])
            #         print(voxSteps_v[ixx][:])
            #         assert 1==2


            # # TODO HERE
            # # This function crops the features and ancestor image, as well as
            # feats_np_v = [u_feat_np, v_feat_np, y_feat_np]
            # anc_np_v   = [larcv.as_ndarray(anc_v[0]), larcv.as_ndarray(anc_v[1]), larcv.as_ndarray(anc_v[2])]
            # This modifies the minVoxCoords to adjust them to the crop
            cropped_feats_np_v, newminVoxCoords, newmaxVoxCoords = self.cropTrack(feats, minVoxCoords, maxVoxCoords)
            minVoxCoords = newminVoxCoords
            maxVoxCoords = newmaxVoxCoords

            # chg_in_wires  = np.zeros((3))
            # chg_in_truths = np.zeros((3))
            # for p in range(3):
            #     cropped_anc_np_v[p][cropped_anc_np_v[p] < 0] = 0
            #     cropped_anc_np_v[p][cropped_anc_np_v[p] > 0] = 1
            #     chg_in_wire  = np.sum(cropped_feats_np_v[p][:,:,-1]).copy()
            #     chg_in_truth = np.sum(cropped_anc_np_v[p]*cropped_feats_np_v[p][:,:,-1]).copy()
            #     chg_in_wires[p]  = chg_in_wire
            #     chg_in_truths[p] = chg_in_truth

            voxSteps_np_v = np.zeros((len(voxSteps_v),4))
            for idxx in range(len(voxSteps_v)):
                voxSteps_np_v[idxx,0] = voxSteps_v[idxx][0]
                voxSteps_np_v[idxx,1] = voxSteps_v[idxx][1]
                voxSteps_np_v[idxx,2] = voxSteps_v[idxx][2]
                voxSteps_np_v[idxx,3] = voxSteps_v[idxx][3]
            voxSteps_v = np.array(voxSteps_v)
            minVoxCoords_np = np.array(minVoxCoords.copy())
            maxVoxCoords_np = np.array(maxVoxCoords.copy())

            voxfeatures_vv.append(cropped_feats_np_v.copy())
            voxSteps_vv.append(voxSteps_np_v.copy())
            run_v.append(run)
            subrun_v.append(subrun)
            eventid_v.append(event)
            entry_v.append(self.currentEntry)
            mctrack_idx_v.append(mctk_idx)
            mctrack_length_v.append(this_length)
            mctrack_pdg_v.append(this_pdg)
            mctrack_energy_v.append(this_energy)
            # charge_in_wires_v.append(chg_in_wires)
            # charge_in_truths_v.append(chg_in_truths)
            minVoxCoords_v.append(minVoxCoords_np)
            maxVoxCoords_v.append(maxVoxCoords_np)

        self.currentEntry += 1
        returnDict = {}
        returnDict["voxfeatures_vv"]            = voxfeatures_vv
        returnDict["voxSteps_vv"]               = voxSteps_vv
        returnDict["run_v"]                     = run_v
        returnDict["subrun_v"]                  = subrun_v
        returnDict["eventid_v"]                 = eventid_v
        returnDict["entry_v"]                   = entry_v
        returnDict["mctrack_idx_v"]             = mctrack_idx_v
        returnDict["mctrack_length_v"]          = mctrack_length_v
        returnDict["mctrack_pdg_v"]             = mctrack_pdg_v
        returnDict["mctrack_energy_v"]          = mctrack_energy_v
        # returnDict["charge_in_wires_v"]         = charge_in_wires_v
        # returnDict["charge_in_truths_v"]        = charge_in_truths_v
        returnDict["minVoxCoords_v"]            = minVoxCoords_v
        returnDict["maxVoxCoords_v"]            = maxVoxCoords_v
        return returnDict


    def collectMCTrackInfo(self, sce_track):
        pos3d_v     = []
        vox3d_v     = []
        minVoxCoords  = [9999999, 9999999, 9999999]
        maxVoxCoords  = [-1, -1, -1]
        lastVoxcoords = [-1, -1, -1]
        for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
            sce_step = sce_track.LocationAtPoint(pos_idx)
            x = sce_step.X()
            y = sce_step.Y()
            z = sce_step.Z()
            if is_inside_boundaries(x,y,z) == False:
                continue
            if pos_idx != 0 and x == sce_track.LocationAtPoint(pos_idx-1).X() and y == sce_track.LocationAtPoint(pos_idx-1).Y() and z == sce_track.LocationAtPoint(pos_idx-1).Z():
                continue

            thisPos3d = [x,y,z]
            thisVox3d = self.voxelator.getVoxelCoord(thisPos3d)
            if thisVox3d == lastVoxcoords:
                continue
            lastVoxcoords = thisVox3d
            pos3d_v.append(thisPos3d)
            vox3d_v.append(thisVox3d)
            for i in range(3):
                if thisVox3d[i] < minVoxCoords[i]:
                    minVoxCoords[i] = thisVox3d[i]
                if thisVox3d[i] > maxVoxCoords[i]:
                    maxVoxCoords[i] = thisVox3d[i]

        return pos3d_v, vox3d_v, minVoxCoords, maxVoxCoords

    def addInterpolatedSteps(self, voxSteps_v, last_x, last_y, last_z, this_x, this_y, this_z):
        # This is complicated, we're going to interpolate steps between last and this
        # In order to do this we need to move in the fastest changing direction primarily
        # then the medium changing direction, then the slowest change direction
        # To see a 2D version see the FancyLoader.py  (not 3D) That has an option
        # for each case, whereas this determines the fastest and slowest and only
        # gets coded once (no "if x is fastest" statements)
        # print("Going from ")
        # print(last_x, last_y, last_z)
        # print("to")
        # print(this_x, this_y, this_z)


        dx = this_x - last_x
        dy = this_y - last_y
        dz = this_z - last_z
        deltas = [dx,dy,dz]
        deltasAbs = [abs(dx),abs(dy),abs(dz)]
        idxAvail = [0,1,2]
        # Get Order of Fastest changing directions
        fastestChanging = 0
        mediumChanging  = 1
        slowestChanging = 2
        if max(deltasAbs) != min(deltasAbs):
            fastestChanging = deltasAbs.index(max(deltasAbs))
            slowestChanging = deltasAbs.index(min(deltasAbs))
            idxAvail.pop(idxAvail.index(fastestChanging))
            idxAvail.pop(idxAvail.index(slowestChanging))
            mediumChanging = idxAvail[0]

        dxChangeIdx, dyChangeIdx, dzChangeIdx = self.getChangeIdxs(fastestChanging, mediumChanging, slowestChanging)
        dFastest = deltas[fastestChanging]
        dMedium  = deltas[mediumChanging]
        dSlowest = deltas[slowestChanging]

        low = 0         if dFastest > 0 else dFastest+1 #Add one because range is [) inclusive on first arg, exclusive on second
        high = dFastest if dFastest > 0 else 0+1
        ddFastest_list = range(low,high) if dFastest > 0 else reversed(range(low,high))
        # print("    ", low, high, dFastest)
        for ddFastest in ddFastest_list:
            # print("    V:",ddFastest)
            ddMedium  = int(round(ddFastest*(dMedium*1.0)/(dFastest)))
            ddSlowest = int(round(ddFastest*(dSlowest*1.0)/(dFastest)))
            dds = [ddFastest, ddMedium, ddSlowest]
            ddx = dds[dxChangeIdx]
            ddy = dds[dyChangeIdx]
            ddz = dds[dzChangeIdx]
            voxSteps_v.append([last_x+ddx, last_y+ddy, last_z+ddz, len(voxSteps_v)+1])
            # print("    ",[last_x+ddx, last_y+ddy, last_z+ddz, len(voxSteps_v)+1])

            if len(voxSteps_v) > 1:
                if abs(voxSteps_v[-2][0] - voxSteps_v[-1][0]) > 1 or \
                abs(voxSteps_v[-2][1] - voxSteps_v[-1][1]) > 1 or \
                abs(voxSteps_v[-2][2] - voxSteps_v[-1][2]) > 1:
                    print("\nDebugError On Jumping")
                    print(last_x, last_y, last_z)
                    print(this_x, this_y, this_z)
                    print(dx,dy,dz)
                    print()
                    print(voxSteps_v[-2][:])
                    print(voxSteps_v[-1][:])
                    # assert 1==2
        # for i in range(len(voxSteps_v)):
        #     print(voxSteps_v[i])

    def getChangeIdxs(self, fastestChanging, mediumChanging, slowestChanging):
        dxChangeIdx = None
        dyChangeIdx = None
        dzChangeIdx = None
        # Get X
        if fastestChanging == 0:
            dxChangeIdx = 0
        elif mediumChanging == 0:
            dxChangeIdx = 1
        else:
            dxChangeIdx = 2
        # Get Y
        if fastestChanging == 1:
            dyChangeIdx = 0
        elif mediumChanging == 1:
            dyChangeIdx = 1
        else:
            dyChangeIdx = 2
        # Get Z
        if fastestChanging == 2:
            dzChangeIdx = 0
        elif mediumChanging == 2:
            dzChangeIdx = 1
        else:
            dzChangeIdx = 2
        return dxChangeIdx, dyChangeIdx, dzChangeIdx

    def cropTrack(self, feats_np, minVoxCoords, maxVoxCoords):
        # feats_np is a N x 3+nFeatures np array where:
        # 3 is xyz coords
        # nFeatures is the features from larvoxel
        # N is the number of nonzero voxels in the detector
        cropped_feats_np_v = []
        newminVoxCoords = [minVoxCoords[p] - 20 for p in range(3)]
        newmaxVoxCoords = [maxVoxCoords[p] + 20 for p in range(3)]
        for i in range(feats_np.shape[0]):
            if feats_np[i,0] >= newminVoxCoords[0] and feats_np[i,0] < newmaxVoxCoords[0]:
                if feats_np[i,1] >= newminVoxCoords[1] and feats_np[i,1] < newmaxVoxCoords[1]:
                    if feats_np[i,2] >= newminVoxCoords[2] and feats_np[i,2] < newmaxVoxCoords[2]:
                        cropped_feats_np_v.append(feats_np[i,:].copy())
        cropped_feats_np_v = np.array(cropped_feats_np_v)


        return cropped_feats_np_v, newminVoxCoords, newmaxVoxCoords

def is_inside_boundaries(xt,yt,zt,buffer = 0):
    x_in = (xt <  255.999-buffer) and (xt >    0.001+buffer)
    y_in = (yt <  116.499-buffer) and (yt > -116.499+buffer)
    z_in = (zt < 1036.999-buffer) and (zt >    0.001+buffer)
    if x_in == True and y_in == True and z_in == True:
        return True
    else:
        return False


def getprojectedpixel(meta,x,y,z,returnAll=False):

    nplanes = 3
    fracpixborder = 1.5
    row_border = fracpixborder*meta.pixel_height();
    col_border = fracpixborder*meta.pixel_width();

    img_coords = [-1,-1,-1,-1]
    tick = x/(larutil.LArProperties.GetME().DriftVelocity()*larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3) + 3200.0;
    if ( tick < meta.min_y() ):
        if ( tick > meta.min_y()- row_border ):
            # below min_y-border, out of image
            img_coords[0] = meta.rows()-1 # note that tick axis and row indicies are in inverse order (same order in larcv2)
        else:
            # outside of image and border
            img_coords[0] = -1
    elif ( tick > meta.max_y() ):
        if (tick < meta.max_y()+row_border):
            # within upper border
            img_coords[0] = 0;
        else:
            # outside of image and border
            img_coords[0] = -1;

    else:
        # within the image
        img_coords[0] = meta.row( tick );


    # Columns
    # xyz = [ x, y, z ]
    xyz = array('d', [x,y,z])

    # there is a corner where the V plane wire number causes an error
    if ( (y>-117.0 and y<-116.0) and z<2.0 ):
        xyz[1] = -116.0;

    for p in range(nplanes):
        wire = larutil.Geometry.GetME().WireCoordinate( xyz, p );

        # get image coordinates
        if ( wire<meta.min_x() ):
            if ( wire>meta.min_x()-col_border ):
                # within lower border
                img_coords[p+1] = 0;
            else:
                img_coords[p+1] = -1;
        elif ( wire>=meta.max_x() ):
            if ( wire<meta.max_x()+col_border ):
                # within border
                img_coords[p+1] = meta.cols()-1
            else:
                # outside border
                img_coords[p+1] = -1
        else:
        # inside image
            img_coords[p+1] = meta.col( wire );
        # end of plane loop

    # there is a corner where the V plane wire number causes an error
    if ( y<-116.3 and z<2.0 and img_coords[1+1]==-1 ):
        img_coords[1+1] = 0;

    if returnAll:
        # row, colu, colv, coly
        return img_coords
    else:
        col = img_coords[2+1]
        row = img_coords[0]
        return col,row

def mcstep_length(step1,step2):
    # Check both steps inside detector
    if is_inside_boundaries(step1.X(),step1.Y(),step1.Z()) == False or is_inside_boundaries(step2.X(),step2.Y(),step2.Z()) == False:
        return 0
    # Return Distance
    dist = ((step2.X() - step1.X())**2 + (step2.Y() - step1.Y())**2 + (step2.Z() - step1.Z())**2)**0.5
    return dist

def mctrack_length(mctrack_in):
    total_dist = 0
    for step_idx in range(mctrack_in.size()):
        if step_idx != 0:
            total_dist += mcstep_length(mctrack_in[step_idx-1],mctrack_in[step_idx])
    return total_dist
