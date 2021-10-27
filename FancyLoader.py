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
# from DataLoader import

class FancyLoader():
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

    def load_fancy(self,start_entry = -1, end_entry = -1):
        larmatch_feature_image_v = []
        wire_image_v = []
        step_idx_image_v = []
        run_v = []
        subrun_v = []
        eventid_v = []
        entry_v = []
        mctrack_idx_v = []
        mctrack_length_v = []
        mctrack_pdg_v = []
        mctrack_energy_v = []
        charge_in_wire_v = []
        charge_in_truth_v = []
        if start_entry == -1:
            start_entry = 0
        if end_entry == -1 or end_entry > self.nentries_ll:
            end_entry = self.nentries_ll
        # end_entry = self.nentries_ll
        for i in range(start_entry,end_entry):
            print("Event ", i)
            self.iocv.read_entry(i)
            self.ioll.go_to(i)
            ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
            # Get Wire ADC Image to a Numpy Array
            y_wire_np = None
            meta      = None
            run       = -1
            subrun    = -1
            event     = -1
            if self.PARAMS['USE_CONV_IM']:
                y_wire_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features2D(i)
            else:
                print("Skipping Wire Im")
                ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
                img_v = ev_wire.Image2DArray()
                y_wire_image2d = img_v[2]
                y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
                run = ev_wire.run()
                subrun = ev_wire.subrun()
                event = ev_wire.event()
                meta = y_wire_image2d.meta()

            ev_defwire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
            imgdef_v = ev_defwire.Image2DArray()
            y_defwire_image2d = imgdef_v[2]
            y_defwire_np = larcv.as_ndarray(y_defwire_image2d) # I am Speed.

            ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
            anc_v = ev_ancestor.Image2DArray()
            y_anc_image2d = anc_v[2]
            y_anc_np = larcv.as_ndarray(y_anc_image2d) # I am Speed.

            print("Number of Tracks", len(ev_mctrack))
            for mctk_idx in range(0,len(ev_mctrack)):
                min_x = 99999999
                max_x = -1
                min_y = 99999999
                max_y = -1
                print("  TrackNum",mctk_idx)
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
                for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
                    # print("    MCPt:",pos_idx)
                    sce_step = sce_track.LocationAtPoint(pos_idx)
                    x = sce_step.X()
                    y = sce_step.Y()
                    z = sce_step.Z()
                    if is_inside_boundaries(x,y,z) == False:
                        continue
                    col,row = getprojectedpixel(meta,x,y,z)
                    # Dont add projected point if in the same position as previous
                    if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                        continue
                    xpt_list.append(col)
                    ypt_list.append(row)

                # Now you have a list of 2D projected pixels. Lets grab points
                # along the line segment and call them steps

                y_stepidx_np = np.zeros(y_defwire_np.shape)
                points_placed = 0
                for ptidx in range(1,len(xpt_list)):
                    # print("    MCPt:",ptidx)
                    last_x = xpt_list[ptidx-1]
                    last_y = ypt_list[ptidx-1]
                    this_x = xpt_list[ptidx]
                    this_y = ypt_list[ptidx]
                    dx = this_x - last_x
                    dy = this_y - last_y
                    # If moving more in x than y
                    y_stepidx_np[last_x,last_y] = points_placed
                    if last_x < min_x:
                        min_x = last_x
                    if last_x > max_x:
                        max_x = last_x
                    if last_y < min_y:
                        min_y = last_y
                    if last_y > max_y:
                        max_y = last_y
                    points_placed += 1
                    if abs(dx) >= abs(dy):
                        low  = 0  if dx > 0 else dx
                        high = dx if dx > 0 else 0
                        ddx_list = range(low,high) if dx > 0 else reversed(range(low,high))
                        for ddx in ddx_list:
                            # print("        PlacePoint:",points_placed)

                            ddy = int(float(ddx)*float(dy)/float(dx))
                            y_stepidx_np[last_x+ddx,last_y+ddy] = points_placed


                            if last_x+ddx < min_x:
                                min_x = last_x+ddx
                            if last_x+ddx > max_x:
                                max_x = last_x+ddx
                            if last_y+ddy < min_y:
                                min_y = last_y+ddy
                            if last_y+ddy > max_y:
                                max_y = last_y+ddy

                            points_placed +=1
                    else:
                        low  = 0  if dy > 0 else dy
                        high = dy if dy > 0 else 0
                        ddy_list = range(low,high) if dy > 0 else reversed(range(low,high))
                        for ddy in ddy_list:
                            # print("        PlacePoint:",points_placed)
                            ddx = int(float(ddy)*float(dx)/float(dy))
                            y_stepidx_np[last_x+ddx,last_y+ddy] = points_placed

                            if last_x+ddx < min_x:
                                min_x = last_x+ddx
                            if last_x+ddx > max_x:
                                max_x = last_x+ddx
                            if last_y+ddy < min_y:
                                min_y = last_y+ddy
                            if last_y+ddy > max_y:
                                max_y = last_y+ddy

                            points_placed +=1



                y_stepidx_np[xpt_list[-1],ypt_list[-1]] = points_placed
                points_placed += points_placed
                if xpt_list[-1] < min_x:
                    min_x = xpt_list[-1]
                if xpt_list[-1] > max_x:
                    max_x = xpt_list[-1]
                if ypt_list[-1] < min_y:
                    min_y = ypt_list[-1]
                if ypt_list[-1] > max_y:
                    max_y = ypt_list[-1]

                xdim = max_x - min_x + 40
                ydim = max_y - min_y + 40
                zdim = y_wire_np.shape[2]
                print("Shapes:")
                print(xdim, ydim)
                cropped_track_wire_np = np.zeros((xdim,ydim))
                cropped_larfeat_np    = np.zeros((xdim,ydim,zdim))
                cropped_anc_np        = np.zeros((xdim,ydim))
                cropped_track_idx_np  = np.zeros((xdim,ydim))
                fromx = min_x - 20
                tox   = max_x + 20
                fromy = min_y - 20
                toy   = max_y + 20
                offlowx = 0
                offhighx = 0
                offlowy = 0
                offhighy = 0

                if fromx < 0:
                    offlowx = 0 - fromx
                    fromx = 0
                if tox > y_defwire_np.shape[0]:
                    offhighx = y_defwire_np.shape[0] - tox
                    tox = y_defwire_np.shape[0]
                if fromy < 0:
                    offlowy = 0 - fromy
                    fromy = 0
                if toy > y_defwire_np.shape[1]:
                    offhighy = y_defwire_np.shape[1] - toy
                    toy = y_defwire_np.shape[1]
                # print(xdim-1+offhighx-0+offlowx)
                # print(ydim-1+offhighy-0+offlowy)
                # print(fromx,tox)
                # print(fromy,toy)
                # print(0+offlowx,xdim-1+offhighx)
                # print(0+offlowy,ydim-1+offhighy)
                print("Time to Slice")
                cropped_track_wire_np[0+offlowx:xdim+offhighx,0+offlowy:ydim+offhighy] = y_defwire_np[fromx:tox,fromy:toy].copy()
                cropped_larfeat_np[0+offlowx:xdim+offhighx,0+offlowy:ydim+offhighy]    = y_wire_np[fromx:tox,fromy:toy].copy()
                cropped_anc_np[0+offlowx:xdim+offhighx,0+offlowy:ydim+offhighy]        = y_anc_np[fromx:tox,fromy:toy].copy()
                cropped_track_idx_np[0+offlowx:xdim+offhighx,0+offlowy:ydim+offhighy]  = y_stepidx_np[fromx:tox,fromy:toy].copy()



                # tmp_wire_np = np.copy(cropped_track_wire_np)
                # tmp_anc_np     = np.copy(cropped_anc_np)
                cropped_anc_np[cropped_anc_np < 0] = 0
                cropped_anc_np[cropped_anc_np > 0] = 1
                chg_in_wire  = np.sum(cropped_track_wire_np)
                chg_in_truth = np.sum(cropped_anc_np*cropped_track_wire_np)
                print("Time to Store")
                larmatch_feature_image_v.append(cropped_larfeat_np.copy())
                wire_image_v.append(cropped_track_wire_np.copy())
                step_idx_image_v.append(cropped_track_idx_np.copy())
                run_v.append(run)
                subrun_v.append(subrun)
                eventid_v.append(event)
                entry_v.append(i)
                mctrack_idx_v.append(mctk_idx)
                mctrack_length_v.append(this_length)
                mctrack_pdg_v.append(this_pdg)
                mctrack_energy_v.append(this_energy)
                charge_in_wire_v.append(chg_in_wire)
                charge_in_truth_v.append(chg_in_truth)

                # print("Returning Early")
                # return larmatch_feature_image_v, wire_image_v, step_idx_image_v, run_v, \
                #     subrun_v, eventid_v, entry_v, mctrack_idx_v, mctrack_length_v, \
                #     mctrack_pdg_v, mctrack_energy_v, charge_in_wire_v, charge_in_truth_v
        print("Returning Normal")
        return larmatch_feature_image_v, wire_image_v, step_idx_image_v, run_v, \
            subrun_v, eventid_v, entry_v, mctrack_idx_v, mctrack_length_v, \
            mctrack_pdg_v, mctrack_energy_v, charge_in_wire_v, charge_in_truth_v
