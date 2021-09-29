import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from LArMatchModel import LArMatchConvNet
from DataLoader import mctrack_length, is_inside_boundaries, getprojectedpixel
from VoxelFunctions import Voxelator
# from DataLoader import

class FancyLoader3D():
    def __init__(self, PARAMS, verbose=False):
        self.PARAMS = PARAMS
        self.verbose = verbose
        self.truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
        self.iocv = None
        self.LArMatchNet = None
        if PARAMS['USE_CONV_IM'] == False:
            self.iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
            self.iocv.set_verbosity(5)
            self.iocv.reverse_all_products() # Do I need this?
            self.iocv.add_in_file(self.PARAMS['INFILE'])
            self.iocv.initialize()
        else:
            self.LArMatchNet = LArMatchConvNet(self.PARAMS)

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
        self.voxelator = Voxelator(self.PARAMS)

    def load_fancy(self):
        features_image_vv    = []
        voxSteps_vv         = []
        run_v = []
        subrun_v = []
        eventid_v = []
        entry_v = []
        mctrack_idx_v = []
        mctrack_length_v = []
        mctrack_pdg_v = []
        mctrack_energy_v = []
        charge_in_wires_v = []
        charge_in_truths_v = []
        minImgCoords_v    = []

        print("Event ", self.currentEntry)
        self.iocv.read_entry(self.currentEntry)
        self.ioll.go_to(self.currentEntry)
        ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
        # Get Wire ADC Image to a Numpy Array
        meta      = None
        run       = -1
        subrun    = -1
        event     = -1

        u_feat_np, v_feat_np, y_feat_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features3D(self.currentEntry)
        self.meta = meta
        ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
        anc_v = ev_ancestor.Image2DArray()

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
            pos3d_v, vox3d_v, minImgCoords, maxImgCoords = self.collectMCTrackInfo(sce_track, meta)
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

                self.addInterpolatedSteps(voxSteps_v, last_x, last_y, last_z, this_x, this_y, this_z,minImgCoords,maxImgCoords)

            # Add last step point
            if not (vox3d_v[-1][0] == vox3d_v[-2][0] and vox3d_v[-1][1] == vox3d_v[-2][1] and vox3d_v[-1][2] == vox3d_v[-2][2]):
                voxSteps_v.append([vox3d_v[-1][0],vox3d_v[-1][1],vox3d_v[-1][2], len(voxSteps_v)+1])


            # TODO HERE
            # This function crops the features and ancestor image, as well as
            feats_np_v = [u_feat_np, v_feat_np, y_feat_np]
            anc_np_v   = [larcv.as_ndarray(anc_v[0]), larcv.as_ndarray(anc_v[1]), larcv.as_ndarray(anc_v[2])]
            # This modifies the minImgCoords to adjust them to the crop
            cropped_feats_np_v, cropped_anc_np_v, newminImgCoords, newmaxImgCoords = self.cropTrack(feats_np_v, anc_np_v, minImgCoords, maxImgCoords)
            minImgCoords = newminImgCoords
            maxImgCoords = newmaxImgCoords

            # tmp_wire_np = np.copy(cropped_track_wire_np)
            # tmp_anc_np     = np.copy(cropped_anc_np)
            chg_in_wires  = np.zeros((3))
            chg_in_truths = np.zeros((3))
            for p in range(3):
                cropped_anc_np_v[p][cropped_anc_np_v[p] < 0] = 0
                cropped_anc_np_v[p][cropped_anc_np_v[p] > 0] = 1
                chg_in_wire  = np.sum(cropped_feats_np_v[p][:,:,-1]).copy()
                chg_in_truth = np.sum(cropped_anc_np_v[p]*cropped_feats_np_v[p][:,:,-1]).copy()
                chg_in_wires[p]  = chg_in_wire
                chg_in_truths[p] = chg_in_truth

            voxSteps_np_v = np.zeros((len(voxSteps_v),4))
            for idxx in range(len(voxSteps_v)):
                voxSteps_np_v[idxx,0] = voxSteps_v[idxx][0]
                voxSteps_np_v[idxx,1] = voxSteps_v[idxx][1]
                voxSteps_np_v[idxx,2] = voxSteps_v[idxx][2]
                voxSteps_np_v[idxx,3] = voxSteps_v[idxx][3]



            # print("Printing 3D Points")
            # print(minImgCoords)
            # print(maxImgCoords)
            # print(len(voxSteps_v))
            # for idxx in range(len(voxSteps_v)):
            #     coord3D = self.voxelator.get3dCoord([voxSteps_np_v[idxx,0],voxSteps_np_v[idxx,1],voxSteps_np_v[idxx,2], len(voxSteps_v)+1])
            #     imgcoords = getprojectedpixel(self.meta,coord3D[0],coord3D[1],coord3D[2], returnAll=True)
            #     print(imgcoords)

            minImgCoords_np = np.array(minImgCoords.copy())
            features_image_vv.append(cropped_feats_np_v.copy())
            voxSteps_vv.append(voxSteps_np_v.copy())
            run_v.append(run)
            subrun_v.append(subrun)
            eventid_v.append(event)
            entry_v.append(self.currentEntry)
            mctrack_idx_v.append(mctk_idx)
            mctrack_length_v.append(this_length)
            mctrack_pdg_v.append(this_pdg)
            mctrack_energy_v.append(this_energy)
            charge_in_wires_v.append(chg_in_wires)
            charge_in_truths_v.append(chg_in_truths)
            minImgCoords_v.append(minImgCoords_np)
        self.currentEntry += 1
        returnDict = {}
        returnDict["features_image_vv"]          = features_image_vv
        returnDict["voxSteps_vv"]               = voxSteps_vv
        returnDict["run_v"]                     = run_v
        returnDict["subrun_v"]                  = subrun_v
        returnDict["eventid_v"]                 = eventid_v
        returnDict["entry_v"]                   = entry_v
        returnDict["mctrack_idx_v"]             = mctrack_idx_v
        returnDict["mctrack_length_v"]          = mctrack_length_v
        returnDict["mctrack_pdg_v"]             = mctrack_pdg_v
        returnDict["mctrack_energy_v"]          = mctrack_energy_v
        returnDict["charge_in_wires_v"]         = charge_in_wires_v
        returnDict["charge_in_truths_v"]        = charge_in_truths_v
        returnDict["minImgCoords_v"]            = minImgCoords_v
        return returnDict


    def collectMCTrackInfo(self, sce_track, meta):
        pos3d_v     = []
        vox3d_v     = []
        minImgCoords  = [9999999, 9999999, 9999999, 9999999]
        maxImgCoords  = [-1, -1, -1, -1]
        lastimgcoords = [-1, -1, -1, -1]
        for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
            sce_step = sce_track.LocationAtPoint(pos_idx)
            x = sce_step.X()
            y = sce_step.Y()
            z = sce_step.Z()
            if is_inside_boundaries(x,y,z) == False:
                continue
            if pos_idx != 0 and x == sce_track.LocationAtPoint(pos_idx-1).X() and y == sce_track.LocationAtPoint(pos_idx-1).Y() and z == sce_track.LocationAtPoint(pos_idx-1).Z():
                continue

            imgcoords = getprojectedpixel(meta,x,y,z, returnAll=True)
            if imgcoords == lastimgcoords:
                continue
            lastimgcoords = imgcoords
            thisPos3d = [x,y,z]
            pos3d_v.append(thisPos3d)
            vox3d_v.append(self.voxelator.getVoxelCoord(thisPos3d))
            for i in range(4):
                if imgcoords[i] < minImgCoords[i]:
                    minImgCoords[i] = imgcoords[i]
                if imgcoords[i] > maxImgCoords[i]:
                    maxImgCoords[i] = imgcoords[i]

        return pos3d_v, vox3d_v, minImgCoords, maxImgCoords

    def addInterpolatedSteps(self, voxSteps_v, last_x, last_y, last_z, this_x, this_y, this_z,minImgCoords,maxImgCoords):
        # This is complicated, we're going to interpolate steps between last and this
        # In order to do this we need to move in the fastest changing direction primarily
        # then the medium changing direction, then the slowest change direction
        # To see a 2D version see the FancyLoader.py  (not 3D) That has an option
        # for each case, whereas this determines the fastest and slowest and only
        # gets coded once (no "if x is fastest" statements)
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

        low = 0         if dFastest > 0 else dFastest
        high = dFastest if dFastest > 0 else 0
        ddFastest_list = range(low,high) if dFastest > 0 else reversed(range(low,high))
        for ddFastest in ddFastest_list:
            ddMedium  = int(ddFastest*(dMedium )/(dFastest))
            ddSlowest = int(ddFastest*(dSlowest)/(dFastest))
            dds = [ddFastest, ddMedium, ddSlowest]
            ddx = dds[dxChangeIdx]
            ddy = dds[dyChangeIdx]
            ddz = dds[dzChangeIdx]
            voxSteps_v.append([last_x+ddx, last_y+ddy, last_z+ddz, len(voxSteps_v)+1])



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

    def cropTrack(self, feats_np_v, anc_np_v, minImgCoords, maxImgCoords):
        cropped_feats_np_v = []
        cropped_anc_np_v = []
        newminImgCoords = [0,0,0,0]
        newmaxImgCoords = [0,0,0,0]

        for p in range(3):
            coldim = maxImgCoords[p+1] - minImgCoords[p+1] + 40
            rowdim = maxImgCoords[0] - minImgCoords[0] + 40
            featdim = feats_np_v[0].shape[2]
            cropped_track_wire_np = np.zeros((coldim,rowdim))
            cropped_larfeat_np    = np.zeros((coldim,rowdim,featdim))
            cropped_anc_np        = np.zeros((coldim,rowdim))
            cropped_track_idx_np  = np.zeros((coldim,rowdim))
            fromx = minImgCoords[p+1] - 20
            tox   = maxImgCoords[p+1] + 20
            fromy = minImgCoords[0] - 20
            toy   = maxImgCoords[0] + 20
            offlowx = 0
            offhighx = 0
            offlowy = 0
            offhighy = 0

            newminImgCoords[p+1] = fromx
            newmaxImgCoords[p+1] = tox
            newminImgCoords[0]   = fromy
            newmaxImgCoords[0]   = toy


            if fromx < 0:
                offlowx = 0 - fromx
                fromx = 0
            if tox > feats_np_v[0].shape[0]:
                offhighx = feats_np_v[0].shape[0] - tox
                tox = feats_np_v[0].shape[0]
            if fromy < 0:
                offlowy = 0 - fromy
                fromy = 0
            if toy > feats_np_v[0].shape[1]:
                offhighy = feats_np_v[0].shape[1] - toy
                toy = feats_np_v[0].shape[1]


            cropped_larfeat_np[0+offlowx:coldim+offhighx,0+offlowy:rowdim+offhighy]    = feats_np_v[p][fromx:tox,fromy:toy].copy()
            cropped_anc_np[0+offlowx:coldim+offhighx,0+offlowy:rowdim+offhighy]        = anc_np_v[p][fromx:tox,fromy:toy].copy()

            cropped_feats_np_v.append(cropped_larfeat_np)
            cropped_anc_np_v.append(cropped_anc_np)

        return cropped_feats_np_v, cropped_anc_np_v, newminImgCoords, newmaxImgCoords
