import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array, paste_target
from LArMatchModel import LArMatchConvNet
from VoxelFunctions import Voxelator


class DataLoader3D_Deploy:
    def __init__(self, PARAMS, verbose=False, all_train = False,all_valid = False,deploy=False):
        self.PARAMS = PARAMS
        self.verbose = verbose
        self.truthtrack_SCE   = ublarcvapp.mctools.TruthTrackSCE()
        self.SCEUBooNE        = larutil.SpaceChargeMicroBooNE()
        self.NeutrinoVertexer = ublarcvapp.mctools.NeutrinoVertex()
        self.LArbysMC         = ublarcvapp.mctools.LArbysMC()
        self.LArbysMC.initialize()
        self.voxelator = Voxelator(self.PARAMS)

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

        if deploy == True:
            self.iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
            self.iocv.set_verbosity(5)
            self.iocv.reverse_all_products() # Do I need this?
            self.iocv.add_in_file(self.PARAMS['INFILE'])
            self.iocv.initialize()
        self.ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
        self.ioll.add_in_filename(self.PARAMS['INFILE'])
        self.ioll.open()

        self.nentries_ll = self.ioll.get_entries()
        self.nentries_train = int(self.nentries_ll*0.8)
        self.nentries_val   = self.nentries_ll-int(self.nentries_ll*0.8)
        self.nentry_val_buffer = self.nentries_train
        self.currentEntry = 0
        if all_train:
            self.nentries_train = self.nentries_ll
            self.nentries_val   = 0
            self.nentry_val_buffer = self.nentries_train
        elif all_valid:
            self.nentries_val = self.nentries_ll
            self.nentries_train   = 0
            self.nentry_val_buffer = self.nentries_train

        print()
        print("Total Events in File:        ", self.nentries_ll)
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

    def load_dlreco_inputs_onestop(self, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1, is_val=False):
        training_data     = []
        entries_v         = []
        mctrack_idx_v     = []
        mctrack_length_v  = []
        mctrack_pdg_v     = []
        mctrack_energy_v  = []
        runs_v            = []
        subruns_v         = []
        event_ids_v       = []
        buffer    = 0
        max_entry = self.nentries_train
        if is_val:
            buffer = self.nentry_val_buffer
            max_entry = self.nentries_val
        if END_ENTRY > max_entry:
            END_ENTRY = max_entry
        assert END_ENTRY > START_ENTRY
        assert START_ENTRY >= 0
        is_val_string = "validation" if is_val else "training"
        print("Loading Entries from",START_ENTRY+buffer, "to",END_ENTRY+buffer,"for",is_val_string)
        for i in range(START_ENTRY+buffer, END_ENTRY+buffer):
            if self.verbose:
                print()
                print("Loading Entry:", i, "of range", START_ENTRY+buffer, END_ENTRY+buffer)
            if self.PARAMS['USE_CONV_IM'] == False:
                self.iocv.read_entry(i)
            else:
                self.iocv.read_entry(i)

            self.ioll.go_to(i)

            ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
            # Get Wire ADC Image to a Numpy Array
            meta     = None
            run      = -1
            subrun   = -1
            event    = -1
            if self.PARAMS['USE_CONV_IM']:
                y_wire_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features(i)
            else:
                ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
                img_v = ev_wire.Image2DArray()
                y_wire_image2d = img_v[2]
                y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
                run = ev_wire.run()
                subrun = ev_wire.subrun()
                event = ev_wire.event()
                meta = y_wire_image2d.meta()

            ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
            img_v = ev_wire.Image2DArray()
            y_wire_image2d = img_v[2]
            y_defwire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.

            ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
            anc_v = ev_ancestor.Image2DArray()
            y_anc_image2d = anc_v[2]
            y_anc_np = larcv.as_ndarray(y_anc_image2d) # I am Speed.

            # Get MC Track X Y Points
            trk_xpt_list = []
            trk_ypt_list = []
            this_event_track_pdgs = []
            trk_idx = -1
            if self.verbose:
                print("N Tracks", len(ev_mctrack))
            for mctrack in ev_mctrack:
                trk_idx += 1
                if mctrack.PdgCode() not in self.PDG_to_Part or self.PDG_to_Part[mctrack.PdgCode()] not in ["PROTON","MUON","PIPLUS","PIMINUS","PI0"]:
                    continue
                print(mctrack.PdgCode())
                track_length = mctrack_length(mctrack)
                if self.verbose:
                    print("Track Index:",trk_idx)
                    print("     Track Length", track_length)
                    if mctrack.PdgCode()  in self.PDG_to_Part:
                        print("     Track PDG:", self.PDG_to_Part[mctrack.PdgCode()])
                    else:
                        print("     Track PDG:", mctrack.PdgCode())
                if track_length < self.PARAMS['MIN_TRACK_LENGTH']:
                    if self.verbose:
                        print("Skipping Short Track")
                    continue
                xpt_list = []
                ypt_list = []
                last_x   = 0
                last_y   = 0
                last_z   = 0
                step_idx = -1
                sce_track = self.truthtrack_SCE.applySCE(mctrack)
                for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
                    sce_step = sce_track.LocationAtPoint(pos_idx)
                    step_idx += 1
                    x = sce_step.X()
                    y = sce_step.Y()
                    z = sce_step.Z()
                    if is_inside_boundaries(x,y,z) == False:
                        continue
                    if step_idx != 0:
                        step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                        # step_dist_3d.append(step_dist)
                    last_x = x
                    last_y = y
                    last_z = z
                    # if trk_idx == 6:
                    #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                    col,row = getprojectedpixel(meta,x,y,z)
                    if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                        continue
                    xpt_list.append(col)
                    ypt_list.append(row)

                full_image = y_wire_np
                steps_x = xpt_list
                steps_y = ypt_list
                # FLAG
                if self.verbose:
                    print("         Original Track Points", len(steps_x))
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,self.PARAMS['PADDING'],always_edge=self.PARAMS['ALWAYS_EDGE'])
                steps_x = new_steps_x
                steps_y = new_steps_y
                if self.verbose:
                    print("         After Inserted Track Points", len(steps_x))
                if len(steps_x) < 2: #Don't  include tracks without a step and then endpoint
                    continue
                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                stepped_wire_images = []
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
                xy_shifts = [] # list of X,Y shifts to take the next step
                charge_in_wire_v  = []
                charge_in_truth_v = []
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
                    cropped_step_image = cropped_np(full_image, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_wire_image = cropped_np(y_defwire_np, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_anc_image  = cropped_np(y_anc_np, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_anc_image[cropped_anc_image < 0] = 0
                    cropped_anc_image[cropped_anc_image > 0] = 1
                    chg_in_wire  = np.sum(cropped_wire_image)
                    chg_in_truth = np.sum(cropped_anc_image*cropped_wire_image)
                    charge_in_wire_v.append(chg_in_wire)
                    charge_in_truth_v.append(chg_in_truth)

                    required_padding_x = self.PARAMS['PADDING'] - step_x
                    required_padding_y = self.PARAMS['PADDING'] - step_y

                    stepped_images.append(cropped_step_image)
                    flat_stepped_images.append(unravel_array(cropped_step_image))
                    stepped_wire_images.append(cropped_wire_image)

                    if idx != len(steps_x)-1:
                        target_x = required_padding_x + next_step_x
                        target_y = required_padding_y + next_step_y
                        np_step_target = np.array([target_x*1.0,target_y*1.0])
                        flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                        if self.PARAMS['AREA_TARGET']:
                            zeros_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(zeros_np,target_x,target_y,self.PARAMS['TARGET_BUFFER'])))
                        next_positions.append(np_step_target)
                        flat_next_positions.append(flat_np_step_target)
                        np_xy_shift = np.array([target_x*1.0-self.PARAMS['PADDING'],target_y*1.0-self.PARAMS['PADDING'] ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        if self.PARAMS['AREA_TARGET']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(targ_np,self.PARAMS['PADDING'],self.PARAMS['PADDING'],self.PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        elif self.PARAMS['CENTERPOINT_ISEND']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        else:
                            next_positions.append(np.array([-1.0,-1.0]))
                            flat_next_positions.append(self.PARAMS['NUM_CLASSES']-1)
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                if self.PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    training_data.append((stepped_images,flat_next_positions,flat_area_positions,stepped_wire_images,charge_in_wire_v,charge_in_truth_v))
                    entries_v.append(i)
                    mctrack_idx_v.append(trk_idx)
                    mctrack_length_v.append(track_length)
                    mctrack_pdg_v.append(mctrack.PdgCode())
                    mctrack_energy_v.append(mctrack.Start().E())
                    runs_v.append(run)
                    subruns_v.append(subrun)
                    event_ids_v.append(event)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v
                else:
                    training_data.append((stepped_images,xy_shifts))
                    entries_v.append(i)
                    mctrack_idx_v.append(trk_idx)
                    mctrack_length_v.append(track_length)
                    mctrack_pdg_v.append(mctrack.PdgCode())
                    mctrack_energy_v.append(mctrack.Start().E())
                    runs_v.append(run)
                    subruns_v.append(subrun)
                    event_ids_v.append(event)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v
                # FLAG
                # End of MCTrack Loop

        is_val_string = "validation" if is_val else "training"
        print("Loading ",len(training_data), "tracks for",is_val_string)
        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v

    def load_deploy_versatile(self, mode='MCNU', prongDict=None):
        deployDict = {}
        deployDict['seedVoxelIdx']   = None
        deployDict['seed3dPos']      = None
        deployDict['entry']          = None
        deployDict['run']            = None
        deployDict['subrun']         = None
        deployDict['event']          = None
        deployDict['meta']           = None
        deployDict['featureImages_v']  = None
        deployDict['wireImages_v']  = None
        deployDict['mcProngs']       = None
        deployDict['mcProngs_thresh']= None

        print("Loading Entry:", self.currentEntry)
        deployDict['entry'] = self.currentEntry
        self.iocv.read_entry(self.currentEntry)
        self.ioll.go_to(self.currentEntry)
        if prongDict != None:
            self.LArbysMC.process(self.iocv, self.ioll)
            nPart        = self.LArbysMC._nproton + self.LArbysMC._nlepton + self.LArbysMC._nmeson
            nPart_thresh = self.LArbysMC._nproton_60mev + self.LArbysMC._nlepton_35mev + self.LArbysMC._nmeson_35mev
            deployDict['mcProngs'] = nPart
            deployDict['mcProngs_thresh'] = nPart_thresh

        if mode in ["MCNU","MCNU_NUE","MCNU_BNB"]:
            passFlag = self.getDeployDictMCNU(deployDict)
            self.currentEntry += 1
            return deployDict, passFlag



    def getDeployDictMCNU(self, deployDict):
        neutrino_vertex = self.NeutrinoVertexer.getPos3DwSCE(self.ioll, self.SCEUBooNE)

        if is_inside_boundaries(neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2]) == False:
            if self.verbose:
                print(neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2], " Out of Bounds")
            return 0

        larmatchFeat_u, larmatchFeat_v, larmatchFeat_y, deployDict['run'], deployDict['subrun'], deployDict['event'], meta = \
                                                     self.LArMatchNet.get_larmatch_features3D(self.currentEntry)

        deployDict['featureImages_v'] = [larmatchFeat_u, larmatchFeat_v, larmatchFeat_y]
        adc_v = self.iocv.get_data(larcv.kProductImage2D, "wire").Image2DArray()
        deployDict['wireImages_v'] = [larcv.as_ndarray(adc_v[p]) for p in range(3)]
        deployDict['meta'] = meta
        deployDict['seed3dPos'] = [neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2]]
        deployDict['seedVoxelIdx'] = self.voxelator.getVoxelCoord(deployDict['seed3dPos'])
        return 1

    def load_dlreco_inputs_onestop_deploy_neutrinovtx(self, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1, run_backwards = False, is_val=True, showermode=False):
        training_data     = []
        entries_v         = []
        mctrack_idx_v     = []
        mctrack_length_v  = []
        mctrack_pdg_v     = []
        mctrack_energy_v  = []
        runs_v            = []
        subruns_v         = []
        event_ids_v       = []
        larmatch_images_v = []
        wire_images_v     = []
        x_starts_v        = []
        y_starts_v        = []
        ssnettrack_ims_v  = []
        ssnetshower_ims_v = []
        n_mcProngs        = []
        n_mcProngs_thresh = []



        buffer    = 0
        max_entry = self.nentries_train
        if is_val:
            buffer = self.nentry_val_buffer
            max_entry = self.nentries_val
        if END_ENTRY > max_entry:
            END_ENTRY = max_entry
        assert END_ENTRY > START_ENTRY
        assert START_ENTRY >= 0
        is_val_string = "validation" if is_val else "training"
        print("Loading Entries from",START_ENTRY+buffer, "to",END_ENTRY+buffer,"for",is_val_string)
        for i in range(START_ENTRY+buffer, END_ENTRY+buffer):
            if self.verbose:
                print()
                print("Loading Entry:", i, "of range", START_ENTRY+buffer, END_ENTRY+buffer)
            self.iocv.read_entry(i)
            self.ioll.go_to(i)
            self.LArbysMC.process(self.iocv, self.ioll)
            nPart        = self.LArbysMC._nproton + self.LArbysMC._nlepton + self.LArbysMC._nmeson
            nPart_thresh = self.LArbysMC._nproton_60mev + self.LArbysMC._nlepton_35mev + self.LArbysMC._nmeson_35mev

            meta     = None
            run      = -1
            subrun   = -1
            event    = -1
            if self.PARAMS['USE_CONV_IM']:
                y_wire_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features(i)
            else:
                ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
                img_v = ev_wire.Image2DArray()
                y_wire_image2d = img_v[2]
                y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
                run = ev_wire.run()
                subrun = ev_wire.subrun()
                event = ev_wire.event()
                meta = y_wire_image2d.meta()\
            # Deply needs wire image as well:
            ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
            img_v = ev_wire.Image2DArray()
            y_wire_image2d = img_v[2]
            y_defwire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.

            ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
            anc_v = ev_ancestor.Image2DArray()
            y_anc_image2d = anc_v[2]
            y_anc_np = larcv.as_ndarray(y_anc_image2d) # I am Speed.

            # ev_ssnet    = self.iocv.get_data(larcv.kProductImage2D,"ubspurn_plane2")
            ev_ssnet    = self.iocv.get_data(larcv.kProductSparseImage,"sparseuresnetout")

			# // 0 -> HIP (Pions+ProtonsTruth)
			# // 1 -> MIP (Muons)
			# // 2 -> Shower
			# // 3 -> Delta Ray
			# // 4 -> Michel
            ssnet_v = ev_ssnet.SparseImageArray().at(2).as_Image2D();
            # y_ssnet_image2d = ssnet_v[2]
            y_ssnet_track_np = larcv.as_ndarray(ssnet_v[0]) +  larcv.as_ndarray(ssnet_v[1])# I am Speed.
            y_ssnet_shower_np = larcv.as_ndarray(ssnet_v[2]) +  larcv.as_ndarray(ssnet_v[3]) +  larcv.as_ndarray(ssnet_v[4])# I am Speed.

            ########
            neutrino_vertex = self.NeutrinoVertexer.getPos3DwSCE(self.ioll, self.SCEUBooNE)
            # ev_mctruth = self.ioll.get_data(larlite.data.kMCTruth,"generator");
            # mctruth = ev_mctruth.at(0)
            # start = mctruth.GetNeutrino().Nu().Trajectory().front()
            # tick = CrossingPointsAnaMethods.getTick(start, 4050.0, None)
            # x = start.X()
            # y = start.Y()
            # z = start.Z()
            # neutrino_vertex = [x,y,z]
            ########
            if is_inside_boundaries(neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2]) == False:
                if self.verbose:
                    print(neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2], " Out of Bounds")
                continue

            col,row = getprojectedpixel(meta,neutrino_vertex[0],neutrino_vertex[1],neutrino_vertex[2])
            stepped_images = [] # List of cropped images as 2D numpy array
            stepped_wire_images = []
            flat_stepped_images = [] # list of cropped images as flattened 1D np array
            next_positions = [] # list of next step positions as np(x,y)
            flat_next_positions = [] # list of next step positions in flattened single coord idx
            flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
            xy_shifts = [] # list of X,Y shifts to take the next step
            charge_in_wire_v  = []
            charge_in_truth_v = []

            cropped_step_image = cropped_np(y_wire_np, col, row, self.PARAMS['PADDING'])
            cropped_wire_image = cropped_np(y_defwire_np, col, row, self.PARAMS['PADDING'])
            cropped_anc_image  = cropped_np(y_anc_np, col, row, self.PARAMS['PADDING'])
            cropped_anc_image[cropped_anc_image < 0] = 0
            cropped_anc_image[cropped_anc_image > 0] = 1
            chg_in_wire  = np.sum(cropped_wire_image)
            chg_in_truth = np.sum(cropped_anc_image*cropped_wire_image)
            charge_in_wire_v.append(chg_in_wire)
            charge_in_truth_v.append(chg_in_truth)

            required_padding_x = self.PARAMS['PADDING'] - col
            required_padding_y = self.PARAMS['PADDING'] - row

            stepped_images.append(cropped_step_image)
            flat_stepped_images.append(unravel_array(cropped_step_image))
            stepped_wire_images.append(cropped_wire_image)

            if self.PARAMS['AREA_TARGET']:
                next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                flat_area_positions.append(unravel_array(paste_target(targ_np,self.PARAMS['PADDING'],self.PARAMS['PADDING'],self.PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                np_xy_shift = np.array([0.0,0.0])
                xy_shifts.append(np_xy_shift)
            elif self.PARAMS['CENTERPOINT_ISEND']:
                next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                np_xy_shift = np.array([0.0,0.0])
                xy_shifts.append(np_xy_shift)
            else:
                next_positions.append(np.array([-1.0,-1.0]))
                flat_next_positions.append(self.PARAMS['NUM_CLASSES']-1)
                np_xy_shift = np.array([0.0,0.0])
                xy_shifts.append(np_xy_shift)

            training_data.append((stepped_images,flat_next_positions,flat_area_positions,stepped_wire_images,charge_in_wire_v,charge_in_truth_v))
            entries_v.append(i)
            mctrack_idx_v.append(0) # Always 1 neutrino idx
            mctrack_length_v.append(-1) # No track length for a vertex
            mctrack_pdg_v.append(-1)#mctrack.PdgCode()) # No PDG
            mctrack_energy_v.append(-1)#mctrack.Start().E())
            runs_v.append(run)
            subruns_v.append(subrun)
            event_ids_v.append(event)
            larmatch_images_v.append(np.copy(y_wire_np))
            wire_images_v.append(np.copy(y_defwire_np))
            x_starts_v.append(col)
            y_starts_v.append(row)
            ssnettrack_ims_v.append(y_ssnet_track_np)
            ssnetshower_ims_v.append(y_ssnet_shower_np)
            n_mcProngs.append(nPart)
            n_mcProngs_thresh.append(nPart_thresh)

            if len(training_data) == MAX_TRACKS_PULL:
                print("Clipping Training Load Size at ",len(training_data))
                is_val_string = "validation" if is_val else "training"
                print("Loading ",len(training_data), "tracks for",is_val_string)
                return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v, larmatch_images_v, wire_images_v, x_starts_v, y_starts_v, ssnettrack_ims_v, ssnetshower_ims_v, n_mcProngs, n_mcProngs_thresh

        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v, larmatch_images_v, wire_images_v, x_starts_v, y_starts_v, ssnettrack_ims_v, ssnetshower_ims_v, n_mcProngs, n_mcProngs_thresh


    def load_dlreco_inputs_onestop_deploy(self, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1, run_backwards = False, is_val=True, showermode=False):
        training_data     = []
        entries_v         = []
        mctrack_idx_v     = []
        mctrack_length_v  = []
        mctrack_pdg_v     = []
        mctrack_energy_v  = []
        runs_v            = []
        subruns_v         = []
        event_ids_v       = []
        larmatch_images_v = []
        wire_images_v     = []
        x_starts_v        = []
        y_starts_v        = []

        buffer    = 0
        max_entry = self.nentries_train
        if is_val:
            buffer = self.nentry_val_buffer
            max_entry = self.nentries_val
        if END_ENTRY > max_entry:
            END_ENTRY = max_entry
        assert END_ENTRY > START_ENTRY
        assert START_ENTRY >= 0
        is_val_string = "validation" if is_val else "training"
        print("Loading Entries from",START_ENTRY+buffer, "to",END_ENTRY+buffer,"for",is_val_string)
        for i in range(START_ENTRY+buffer, END_ENTRY+buffer):
            if self.verbose:
                print()
                print("Loading Entry:", i, "of range", START_ENTRY+buffer, END_ENTRY+buffer)
            if self.PARAMS['USE_CONV_IM'] == False:
                self.iocv.read_entry(i)
            else:
                self.iocv.read_entry(i)
            self.ioll.go_to(i)

            ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
            if showermode:
                # ev_mctrack = self.ioll.get_data(larlite.data.kMCTruth,  "generator" );
                ev_mctrack = self.ioll.get_data(larlite.data.kMCShower, "mcreco")
            # Get Wire ADC Image to a Numpy Array
            meta     = None
            run      = -1
            subrun   = -1
            event    = -1
            if self.PARAMS['USE_CONV_IM']:
                y_wire_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features(i)
            else:
                ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
                img_v = ev_wire.Image2DArray()
                y_wire_image2d = img_v[2]
                y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
                run = ev_wire.run()
                subrun = ev_wire.subrun()
                event = ev_wire.event()
                meta = y_wire_image2d.meta()\
            # Deply needs wire image as well:
            ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
            img_v = ev_wire.Image2DArray()
            y_wire_image2d = img_v[2]
            y_defwire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.

            ev_ancestor    = self.iocv.get_data(larcv.kProductImage2D,"ancestor")
            anc_v = ev_ancestor.Image2DArray()
            y_anc_image2d = anc_v[2]
            y_anc_np = larcv.as_ndarray(y_anc_image2d) # I am Speed.

            # Get MC Track X Y Points
            trk_xpt_list = []
            trk_ypt_list = []
            this_event_track_pdgs = []
            trk_idx = -1
            if self.verbose:
                print("N Tracks", len(ev_mctrack))
            mctrack = None
            nTracks = len(ev_mctrack) #if not showermode else 1
            # for mctrack in ev_mctrack:
            for mc_idx in range(nTracks):
                mctrack = ev_mctrack.at(mc_idx)
                # mctrack = ev_mctrack.at(mc_idx) if not showermode else ev_mctrack.at(0).GetNeutrino().Nu()
                trk_idx += 1
                if showermode:
                    if trk_idx != 0:
                        continue
                if not showermode:
                    if mctrack.PdgCode() not in self.PDG_to_Part or self.PDG_to_Part[mctrack.PdgCode()] not in ["PROTON","MUON","PIPLUS","PIMINUS","PI0"]:
                        continue
                print(mctrack.PdgCode())
                track_length = -1
                if not showermode:
                    track_length = mctrack_length(mctrack)
                    if self.verbose:
                        print("Track Index:",trk_idx)
                        print("     Track Length", track_length)
                        if mctrack.PdgCode()  in self.PDG_to_Part:
                            print("     Track PDG:", self.PDG_to_Part[mctrack.PdgCode()])
                        else:
                            print("     Track PDG:", mctrack.PdgCode())
                    if track_length < self.PARAMS['MIN_TRACK_LENGTH']:
                        if self.verbose:
                            print("Skipping Short Track")
                        continue

                xpt_list = []
                ypt_list = []
                last_x   = 0
                last_y   = 0
                last_z   = 0
                step_idx = -1
                sce_track = None
                if showermode:
                    sce_track = mctrack
                    sce_step  = sce_track.Start()
                    x = sce_step.X()
                    y = sce_step.Y()
                    z = sce_step.Z()
                    if is_inside_boundaries(x,y,z) == False:
                        if self.verbose:
                            print(x,y,z, " Out of Bounds")
                        continue
                    if step_idx != 0:
                        step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                        # step_dist_3d.append(step_dist)
                    last_x = x
                    last_y = y
                    last_z = z
                    # if trk_idx == 6:
                    #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                    col,row = getprojectedpixel(meta,x,y,z)
                    if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                        continue
                    xpt_list.append(col)
                    ypt_list.append(row)
                    track_length = ((sce_track.End().Z() - z)**2 + (sce_track.End().Y() - y)**2 + (sce_track.End().X() - x)**2 )**0.2
                    if self.verbose:
                        print("Breaking Shower")

                else:
                    sce_track = self.truthtrack_SCE.applySCE(mctrack)
                    for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
                        sce_step = None
                        if run_backwards:
                            sce_step = sce_track.LocationAtPoint(sce_track.NumberTrajectoryPoints()-1-pos_idx)
                        else:
                            sce_step = sce_track.LocationAtPoint(pos_idx)


                        step_idx += 1
                        x = sce_step.X()
                        y = sce_step.Y()
                        z = sce_step.Z()
                        if is_inside_boundaries(x,y,z) == False:
                            continue
                        if step_idx != 0:
                            step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                            # step_dist_3d.append(step_dist)
                        last_x = x
                        last_y = y
                        last_z = z
                        # if trk_idx == 6:
                        #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                        col,row = getprojectedpixel(meta,x,y,z)
                        if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                            continue
                        xpt_list.append(col)
                        ypt_list.append(row)
                        break
                        if len(xpt_list) == 1:
                            print("Printing Col and Row:\n",col, row)

                full_image = y_wire_np
                steps_x = xpt_list
                steps_y = ypt_list
                # FLAG
                if self.verbose:
                    print("         Original Track Points", len(steps_x))
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,self.PARAMS['PADDING'],always_edge=self.PARAMS['ALWAYS_EDGE'])
                steps_x = new_steps_x
                steps_y = new_steps_y
                if self.verbose:
                    print("         After Inserted Track Points", len(steps_x))
                # if len(steps_x) < 2: #Don't  include tracks without a step and then endpoint
                #     continue
                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                stepped_wire_images = []
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
                xy_shifts = [] # list of X,Y shifts to take the next step
                charge_in_wire_v  = []
                charge_in_truth_v = []
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

                    cropped_step_image = cropped_np(full_image, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_wire_image = cropped_np(y_defwire_np, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_anc_image  = cropped_np(y_anc_np, step_x, step_y, self.PARAMS['PADDING'])
                    cropped_anc_image[cropped_anc_image < 0] = 0
                    cropped_anc_image[cropped_anc_image > 0] = 1
                    chg_in_wire  = np.sum(cropped_wire_image)
                    chg_in_truth = np.sum(cropped_anc_image*cropped_wire_image)
                    charge_in_wire_v.append(chg_in_wire)
                    charge_in_truth_v.append(chg_in_truth)

                    required_padding_x = self.PARAMS['PADDING'] - step_x
                    required_padding_y = self.PARAMS['PADDING'] - step_y

                    stepped_images.append(cropped_step_image)
                    flat_stepped_images.append(unravel_array(cropped_step_image))
                    stepped_wire_images.append(cropped_wire_image)

                    if idx != len(steps_x)-1:
                        target_x = required_padding_x + next_step_x
                        target_y = required_padding_y + next_step_y
                        np_step_target = np.array([target_x*1.0,target_y*1.0])
                        flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                        if self.PARAMS['AREA_TARGET']:
                            zeros_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(zeros_np,target_x,target_y,self.PARAMS['TARGET_BUFFER'])))
                        next_positions.append(np_step_target)
                        flat_next_positions.append(flat_np_step_target)
                        np_xy_shift = np.array([target_x*1.0-self.PARAMS['PADDING'],target_y*1.0-self.PARAMS['PADDING'] ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        if self.PARAMS['AREA_TARGET']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(targ_np,self.PARAMS['PADDING'],self.PARAMS['PADDING'],self.PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        elif self.PARAMS['CENTERPOINT_ISEND']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        else:
                            next_positions.append(np.array([-1.0,-1.0]))
                            flat_next_positions.append(self.PARAMS['NUM_CLASSES']-1)
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                if self.PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    training_data.append((stepped_images,flat_next_positions,flat_area_positions,stepped_wire_images,charge_in_wire_v,charge_in_truth_v))
                    entries_v.append(i)
                    mctrack_idx_v.append(trk_idx)
                    mctrack_length_v.append(track_length)
                    mctrack_pdg_v.append(mctrack.PdgCode())
                    mctrack_energy_v.append(mctrack.Start().E())
                    runs_v.append(run)
                    subruns_v.append(subrun)
                    event_ids_v.append(event)
                    larmatch_images_v.append(np.copy(full_image))
                    wire_images_v.append(np.copy(y_defwire_np))
                    x_starts_v.append(steps_x[0])
                    y_starts_v.append(steps_y[0])

                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v, larmatch_images_v, wire_images_v, x_starts_v, y_starts_v
                else:
                    training_data.append((stepped_images,xy_shifts))
                    entries_v.append(i)
                    mctrack_idx_v.append(trk_idx)
                    mctrack_length_v.append(track_length)
                    mctrack_pdg_v.append(mctrack.PdgCode())
                    mctrack_energy_v.append(mctrack.Start().E())
                    runs_v.append(run)
                    subruns_v.append(subrun)
                    event_ids_v.append(event)
                    larmatch_images_v.append(np.copy(full_image))
                    wire_images_v.append(np.copy(y_defwire_np))
                    x_starts_v.append(steps_x[0])
                    y_starts_v.append(steps_y[0])
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v, larmatch_images_v, wire_images_v, x_starts_v, y_starts_v
                # FLAG
                # End of MCTrack Loop

        is_val_string = "validation" if is_val else "training"
        print("Loading ",len(training_data), "tracks for",is_val_string)
        return training_data, entries_v, mctrack_idx_v, mctrack_length_v, mctrack_pdg_v, mctrack_energy_v, runs_v, subruns_v, event_ids_v, larmatch_images_v, wire_images_v, x_starts_v, y_starts_v

    def get_net_inputs_mc(self, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1, is_val=False):
        image_list, xs, ys, runs, subruns, events, filepaths, entries, track_pdgs = self.load_rootfile_MC_Positions(START_ENTRY, END_ENTRY, is_val=is_val)
        training_data = []
        full_images = []
        event_ids = []
        steps_x = []
        steps_y = []
        for EVENT_IDX in range(len(image_list)):
            if self.verbose:
                print("Doing Event:", EVENT_IDX)
                print("N MC Tracks:", len(xs[EVENT_IDX]))
            for TRACK_IDX in range(len(xs[EVENT_IDX])):
                # if TRACK_IDX != 0:
                #     continue
                if self.verbose:
                    print("     Doing Track:", TRACK_IDX)
                full_image = image_list[EVENT_IDX]
                steps_x = xs[EVENT_IDX][TRACK_IDX]
                steps_y = ys[EVENT_IDX][TRACK_IDX]

                if self.verbose:
                    print("         Original Track Points", len(steps_x))
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,self.PARAMS['PADDING'],always_edge=self.PARAMS['ALWAYS_EDGE'])
                steps_x = new_steps_x
                steps_y = new_steps_y
                if self.verbose:
                    print("         After Inserted Track Points", len(steps_x))
                if len(steps_x) < 2: #Don't  include tracks without a step and then endpoint
                    continue

                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
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
                    cropped_step_image = cropped_np(full_image, step_x, step_y, self.PARAMS['PADDING'])
                    required_padding_x = self.PARAMS['PADDING'] - step_x
                    required_padding_y = self.PARAMS['PADDING'] - step_y

                    stepped_images.append(cropped_step_image)
                    flat_stepped_images.append(unravel_array(cropped_step_image))

                    if idx != len(steps_x)-1:
                        target_x = required_padding_x + next_step_x
                        target_y = required_padding_y + next_step_y
                        np_step_target = np.array([target_x*1.0,target_y*1.0])
                        flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                        if self.PARAMS['AREA_TARGET']:
                            zeros_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(zeros_np,target_x,target_y,self.PARAMS['TARGET_BUFFER'])))
                        next_positions.append(np_step_target)
                        flat_next_positions.append(flat_np_step_target)
                        np_xy_shift = np.array([target_x*1.0-self.PARAMS['PADDING'],target_y*1.0-self.PARAMS['PADDING'] ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        if self.PARAMS['AREA_TARGET']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(targ_np,self.PARAMS['PADDING'],self.PARAMS['PADDING'],self.PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        elif self.PARAMS['CENTERPOINT_ISEND']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        else:
                            next_positions.append(np.array([-1.0,-1.0]))
                            flat_next_positions.append(self.PARAMS['NUM_CLASSES']-1)
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                if self.PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    training_data.append((stepped_images,flat_next_positions,flat_area_positions))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data
                else:
                    training_data.append((stepped_images,xy_shifts))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data
        # rse_pdg_dict = {}
        # rse_pdg_dict['runs'] = runs
        # rse_pdg_dict['subruns'] = subruns
        # rse_pdg_dict['events'] = events
        # rse_pdg_dict['filepaths'] = filepaths
        # rse_pdg_dict['pdgs'] = track_pdgs
        # rse_pdg_dict['file_idx'] = entries
        is_val_string = "validation" if is_val else "training"
        print("Loading ",len(training_data), "tracks for",is_val_string)
        return training_data #Could return more: full_images, steps_x, steps_y, event_ids, rse_pdg_dict

    def get_net_inputs_mc_fullout(self, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1, is_val=False):
        image_list, xs, ys, runs, subruns, events, filepaths, entries, track_pdgs = self.load_rootfile_MC_Positions(START_ENTRY, END_ENTRY, is_val=is_val)
        training_data = []
        full_images = []
        event_ids = []
        steps_x = []
        steps_y = []
        entry_num = []
        track_idx = []
        for EVENT_IDX in range(len(image_list)):
            if self.verbose:
                print("Doing Event:", EVENT_IDX)
                print("N MC Tracks:", len(xs[EVENT_IDX]))
            for TRACK_IDX in range(len(xs[EVENT_IDX])):
                # if TRACK_IDX != 0:
                #     continue
                if self.verbose:
                    print("     Doing Track:", TRACK_IDX)
                full_image = image_list[EVENT_IDX]
                steps_x = xs[EVENT_IDX][TRACK_IDX]
                steps_y = ys[EVENT_IDX][TRACK_IDX]

                if self.verbose:
                    print("         Original Track Points", len(steps_x))
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,self.PARAMS['PADDING'],always_edge=self.PARAMS['ALWAYS_EDGE'])
                steps_x = new_steps_x
                steps_y = new_steps_y
                if self.verbose:
                    print("         After Inserted Track Points", len(steps_x))
                if len(steps_x) < 2: #Don't  include tracks without a step and then endpoint
                    continue

                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
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
                    cropped_step_image = cropped_np(full_image, step_x, step_y, self.PARAMS['PADDING'])
                    required_padding_x = self.PARAMS['PADDING'] - step_x
                    required_padding_y = self.PARAMS['PADDING'] - step_y

                    stepped_images.append(cropped_step_image)
                    flat_stepped_images.append(unravel_array(cropped_step_image))

                    if idx != len(steps_x)-1:
                        target_x = required_padding_x + next_step_x
                        target_y = required_padding_y + next_step_y
                        np_step_target = np.array([target_x*1.0,target_y*1.0])
                        flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                        if self.PARAMS['AREA_TARGET']:
                            zeros_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(zeros_np,target_x,target_y,self.PARAMS['TARGET_BUFFER'])))
                        next_positions.append(np_step_target)
                        flat_next_positions.append(flat_np_step_target)
                        np_xy_shift = np.array([target_x*1.0-self.PARAMS['PADDING'],target_y*1.0-self.PARAMS['PADDING'] ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        if self.PARAMS['AREA_TARGET']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(targ_np,self.PARAMS['PADDING'],self.PARAMS['PADDING'],self.PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        elif self.PARAMS['CENTERPOINT_ISEND']:
                            next_positions.append(np.array([self.PARAMS['PADDING'],self.PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((self.PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        else:
                            next_positions.append(np.array([-1.0,-1.0]))
                            flat_next_positions.append(self.PARAMS['NUM_CLASSES']-1)
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                if self.PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    training_data.append((stepped_images,flat_next_positions,flat_area_positions))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data
                else:
                    training_data.append((stepped_images,xy_shifts))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        is_val_string = "validation" if is_val else "training"
                        print("Loading ",len(training_data), "tracks for",is_val_string)
                        return training_data
        rse_pdg_dict = {}
        rse_pdg_dict['runs'] = runs
        rse_pdg_dict['subruns'] = subruns
        rse_pdg_dict['events'] = events
        rse_pdg_dict['filepaths'] = filepaths
        rse_pdg_dict['pdgs'] = track_pdgs
        rse_pdg_dict['file_idx'] = entries
        is_val_string = "validation" if is_val else "training"
        print("Loading ",len(training_data), "tracks for",is_val_string)
        return training_data, full_images, steps_x, steps_y, event_ids, rse_pdg_dict

    def load_rootfile_MC_Positions(self, start_entry, end_entry, is_val = False):
        full_image_list = []
        ev_trk_xpt_list = []
        ev_trk_ypt_list = []
        runs      = []
        subruns   = []
        events    = []
        filepaths = []
        entries   = []
        track_pdgs= []
        buffer    = 0
        max_entry = self.nentries_train
        if is_val:
            buffer = self.nentry_val_buffer
            max_entry = self.nentries_val
        assert end_entry < max_entry
        assert end_entry > start_entry
        assert start_entry >= 0
        # if end_entry > max_entry or end_entry == -1:
        #     end_entry = max_entry
        # if start_entry > end_entry or start_entry < 0:
        #     start_entry = 0
        is_val_string = "validation" if is_val else "training"
        print("Loading Entries from",start_entry+buffer, "to",end_entry+buffer,"for",is_val_string)
        for i in range(start_entry+buffer, end_entry+buffer):
            if self.verbose:
                print()
                print("Loading Entry:", i, "of range", start_entry+buffer, end_entry+buffer)
            if self.PARAMS['USE_CONV_IM'] == False:
                self.iocv.read_entry(i)
            self.ioll.go_to(i)

            ev_mctrack = self.ioll.get_data(larlite.data.kMCTrack, "mcreco")
            # Get Wire ADC Image to a Numpy Array
            meta     = None
            run      = -1
            subrun   = -1
            event    = -1
            if self.PARAMS['USE_CONV_IM']:
                y_wire_np, run, subrun, event, meta = self.LArMatchNet.get_larmatch_features(i)
            else:
                ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
                img_v = ev_wire.Image2DArray()
                y_wire_image2d = img_v[2]
                y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
                run = ev_wire.run()
                subrun = ev_wire.subrun()
                event = ev_wire.event()
                meta = y_wire_image2d.meta()
            full_image_list.append(np.copy(y_wire_np))
            runs.append(run)
            subruns.append(subrun)
            events.append(event)
            filepaths.append(self.PARAMS['INFILE'])
            entries.append(i)
            # Get MC Track X Y Points

            trk_xpt_list = []
            trk_ypt_list = []
            this_event_track_pdgs = []
            trk_idx = -1
            if self.verbose:
                print("N Tracks", len(ev_mctrack))
            for mctrack in ev_mctrack:
                trk_idx += 1
                if mctrack.PdgCode() not in self.PDG_to_Part or self.PDG_to_Part[mctrack.PdgCode()] not in ["PROTON","MUON"]:
                    continue
                track_length = mctrack_length(mctrack)
                if self.verbose:
                    print("Track Index:",trk_idx)
                    print("     Track Length", track_length)
                    if mctrack.PdgCode()  in self.PDG_to_Part:
                        print("     Track PDG:", self.PDG_to_Part[mctrack.PdgCode()])
                    else:
                        print("     Track PDG:", mctrack.PdgCode())
                if track_length < 3.0:
                    if self.verbose:
                        print("Skipping Short Track")
                    continue
                xpt_list = []
                ypt_list = []
                last_x   = 0
                last_y   = 0
                last_z   = 0
                step_idx = -1
                sce_track = self.truthtrack_SCE.applySCE(mctrack)
                for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
                # for mcstep in mctrack:
                    sce_step = sce_track.LocationAtPoint(pos_idx)
                    step_idx += 1
                    x = sce_step.X()
                    y = sce_step.Y()
                    z = sce_step.Z()
                    if is_inside_boundaries(x,y,z) == False:
                        continue
                    if step_idx != 0:
                        step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                        # step_dist_3d.append(step_dist)
                    last_x = x
                    last_y = y
                    last_z = z
                    # if trk_idx == 6:
                    #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                    col,row = getprojectedpixel(meta,x,y,z)
                    if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                        continue
                    xpt_list.append(col)
                    ypt_list.append(row)
                trk_xpt_list.append(xpt_list)
                trk_ypt_list.append(ypt_list)
                this_event_track_pdgs.append(mctrack.PdgCode())
            ev_trk_xpt_list.append(trk_xpt_list)
            ev_trk_ypt_list.append(trk_ypt_list)
            track_pdgs.append(this_event_track_pdgs)

        return full_image_list, ev_trk_xpt_list, ev_trk_ypt_list, runs, subruns, events, filepaths, entries, track_pdgs



def get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY, MAX_TRACKS_PULL = -1):
    # This function takes a root file path, a start entry and an end entry
    # and returns the mc track information in a form the network is
    # prepared to take as input
    print("Loading Network Inputs")
    step_dist_3d = []
    image_list, xs, ys, runs, subruns, events, filepaths, entries, track_pdgs = load_rootfile_training(PARAMS, step_dist_3d, START_ENTRY, END_ENTRY)

    training_data = []
    for x in range(1):
        print('\n\n\n')
        # This is just a loop to see if this is blowing up memory (it will if you loop)
        print(x)
        print('\n\n\n')
        full_images = []
        event_ids = []

        steps_x = []
        steps_y = []
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
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,PARAMS['PADDING'],always_edge=PARAMS['ALWAYS_EDGE'])
                steps_x = new_steps_x
                steps_y = new_steps_y

                print("         After Inserted Track Points", len(steps_x))
                if len(steps_x) < 2: #Don't  include tracks without a step and then endpoint
                    continue

                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                flat_area_positions = [] # list of np_zeros with 1s pasted in a square around target
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
                    cropped_step_image = cropped_np(full_image, step_x, step_y, PARAMS['PADDING'])
                    required_padding_x = PARAMS['PADDING'] - step_x
                    required_padding_y = PARAMS['PADDING'] - step_y

                    stepped_images.append(cropped_step_image)
                    flat_stepped_images.append(unravel_array(cropped_step_image))

                    if idx != len(steps_x)-1:
                        target_x = required_padding_x + next_step_x
                        target_y = required_padding_y + next_step_y
                        np_step_target = np.array([target_x*1.0,target_y*1.0])
                        flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
                        if PARAMS['AREA_TARGET']:
                            zeros_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(zeros_np,target_x,target_y,PARAMS['TARGET_BUFFER'])))
                        next_positions.append(np_step_target)
                        flat_next_positions.append(flat_np_step_target)
                        np_xy_shift = np.array([target_x*1.0-PARAMS['PADDING'],target_y*1.0-PARAMS['PADDING'] ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        if PARAMS['AREA_TARGET']:
                            next_positions.append(np.array([PARAMS['PADDING'],PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            targ_np = np.zeros((cropped_step_image.shape[0],cropped_step_image.shape[1]))
                            flat_area_positions.append(unravel_array(paste_target(targ_np,PARAMS['PADDING'],PARAMS['PADDING'],PARAMS['TARGET_BUFFER']))) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        elif PARAMS['CENTERPOINT_ISEND']:
                            next_positions.append(np.array([PARAMS['PADDING'],PARAMS['PADDING']])) #should correspond to centerpoint
                            flat_next_positions.append((PARAMS['NUM_CLASSES']-1)/2) #should correspond to centerpoint
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                        else:
                            next_positions.append(np.array([-1.0,-1.0]))
                            flat_next_positions.append(PARAMS['NUM_CLASSES']-1)
                            np_xy_shift = np.array([0.0,0.0])
                            xy_shifts.append(np_xy_shift)
                if PARAMS['CLASSIFIER_NOT_DISTANCESHIFTER']:
                    training_data.append((stepped_images,flat_next_positions,flat_area_positions))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        break
                else:
                    training_data.append((stepped_images,xy_shifts))
                    event_ids.append(EVENT_IDX)
                    if len(training_data) == MAX_TRACKS_PULL:
                        print("Clipping Training Load Size at ",len(training_data))
                        break
        rse_pdg_dict = {}
        rse_pdg_dict['runs'] = runs
        rse_pdg_dict['subruns'] = subruns
        rse_pdg_dict['events'] = events
        rse_pdg_dict['filepaths'] = filepaths
        rse_pdg_dict['pdgs'] = track_pdgs
        rse_pdg_dict['file_idx'] = entries
    return training_data, full_images, steps_x, steps_y, event_ids, rse_pdg_dict

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


def load_rootfile_training(PARAMS, step_dist_3d, start_entry = 0, end_entry = -1):
    truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
    infile = PARAMS['INFILE']
    iocv = None
    LArMatchNet = None
    if PARAMS['USE_CONV_IM'] == False:
        iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
        iocv.set_verbosity(5)
        iocv.reverse_all_products() # Do I need this?
        iocv.add_in_file(infile)
        iocv.initialize()
    else:
        LArMatchNet = LArMatchConvNet(PARAMS)
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(infile)
    ioll.open()

    nentries_ll = ioll.get_entries()


    full_image_list = []
    ev_trk_xpt_list = []
    ev_trk_ypt_list = []
    runs = []
    subruns   = []
    events    = []
    filepaths = []
    entries   = []
    track_pdgs= []

    PDG_to_Part = {
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

    if end_entry > nentries_ll or end_entry == -1:
        end_entry = nentries_ll
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
    # for i in range(8,9):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        if PARAMS['USE_CONV_IM'] == False:
            iocv.read_entry(i)
        ioll.go_to(i)

        ev_mctrack = ioll.get_data(larlite.data.kMCTrack, "mcreco")
        # Get Wire ADC Image to a Numpy Array
        meta     = None
        run      = -1
        subrun   = -1
        event    = -1
        if PARAMS['USE_CONV_IM']:
            y_wire_np, run, subrun, event, meta = LArMatchNet.get_larmatch_features(i)
        else:
            ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
            img_v = ev_wire.Image2DArray()
            y_wire_image2d = img_v[2]
            y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
            run = ev_wire.run()
            subrun = ev_wire.subrun()
            event = ev_wire.event()
            meta = y_wire_image2d.meta()
        full_image_list.append(np.copy(y_wire_np))
        runs.append(run)
        subruns.append(subrun)
        events.append(event)
        filepaths.append(infile)
        entries.append(i)
        # Get MC Track X Y Points

        trk_xpt_list = []
        trk_ypt_list = []
        this_event_track_pdgs = []
        trk_idx = -1
        print("N Tracks", len(ev_mctrack))
        for mctrack in ev_mctrack:
            trk_idx += 1
            if mctrack.PdgCode() not in PDG_to_Part or PDG_to_Part[mctrack.PdgCode()] not in ["PROTON","MUON"]:
                continue
            print("Track Index:",trk_idx)
            if mctrack.PdgCode()  in PDG_to_Part:
                print("     Track PDG:", PDG_to_Part[mctrack.PdgCode()])
            else:
                print("     Track PDG:", mctrack.PdgCode())
            track_length = mctrack_length(mctrack)
            print("     Track Length", track_length)
            if track_length < 3.0:
                print("Skipping Short Track")
                continue
            xpt_list = []
            ypt_list = []
            last_x   = 0
            last_y   = 0
            last_z   = 0
            step_idx = -1
            sce_track = truthtrack_SCE.applySCE(mctrack)
            for pos_idx  in range(sce_track.NumberTrajectoryPoints()):
            # for mcstep in mctrack:
                sce_step = sce_track.LocationAtPoint(pos_idx)
                step_idx += 1
                x = sce_step.X()
                y = sce_step.Y()
                z = sce_step.Z()
                if is_inside_boundaries(x,y,z) == False:
                    continue
                if step_idx != 0:
                    step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                    step_dist_3d.append(step_dist)
                last_x = x
                last_y = y
                last_z = z
                # if trk_idx == 6:
                #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                col,row = getprojectedpixel(meta,x,y,z)
                if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                    continue
                xpt_list.append(col)
                ypt_list.append(row)
            trk_xpt_list.append(xpt_list)
            trk_ypt_list.append(ypt_list)
            this_event_track_pdgs.append(mctrack.PdgCode())
        ev_trk_xpt_list.append(trk_xpt_list)
        ev_trk_ypt_list.append(trk_ypt_list)
        track_pdgs.append(this_event_track_pdgs)
    ioll.close()
    iocv.finalize()
    return full_image_list, ev_trk_xpt_list, ev_trk_ypt_list, runs, subruns, events, filepaths, entries, track_pdgs

def load_rootfile_deploy(filename, start_entry = 0, end_entry = -1, seed_MC=False):
    truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
    infile = filename
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(infile)
    ioll.open()

    nentries_ll = iocv.get_n_entries()



    full_image_list = []
    x_starts = []
    y_starts = []
    runs = []
    subruns   = []
    events    = []
    filepaths = []
    entries   = []
    track_pdgs= []

    PDG_to_Part = {
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

    if end_entry > nentries_ll or end_entry == -1:
        end_entry = nentries_ll
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(i)
        ioll.go_to(i)

        ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
        pgraph_arr  = iocv.get_data(larcv.kProductPGraph, "inter_par").PGraphArray()
        # Get Wire ADC Image to a Numpy Array
        img_v = ev_wire.Image2DArray()
        y_wire_image2d = img_v[2]
        rows = y_wire_image2d.meta().rows()
        cols = y_wire_image2d.meta().cols()
        y_wire_np = np.zeros((cols,rows))

        y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
        # full_image_list.append(y_wire_np)
        runs.append(ev_wire.run())
        subruns.append(ev_wire.subrun())
        events.append(ev_wire.event())
        filepaths.append(infile)
        entries.append(i)

        print("SHAPE TEST")
        print(y_wire_np.shape)

        # Get MC Track X Y Points
        meta = y_wire_image2d.meta()
        trk_xpt_list = []
        trk_ypt_list = []
        this_event_track_pdgs = []
        trk_idx = -1
        for vtx_ix in range(pgraph_arr.size()):
            full_image_list.append(y_wire_np)
            this_pgraph = pgraph_arr.at(vtx_ix)
            this_part = this_pgraph.ParticleArray().at(0)

            vtx = this_part
            vtx_x = vtx.X()
            vtx_y = vtx.Y()
            vtx_z = vtx.Z()
            col,row = getprojectedpixel(meta,vtx_x,vtx_y,vtx_z)
            x_starts.append(col)
            y_starts.append(row)

    return full_image_list, x_starts, y_starts, runs, subruns, events, filepaths, entries, track_pdgs


def insert_cropedge_steps(x_pts_list, y_pts_list, padding, always_edge=False):
    # Lists of true x,y points in a track, and the padding of a crop
    # (Crop Dimension = 2xPadding+1)
    # Since each crop is centered on the current point, the padding says how far
    # you can shift in x or y before you go out of crop. In those cases we
    # insert new points at the edge of the crop.

    # If always_edge is set to True then this function will always place the next
    # on the edge of the crop. This is only advised for small crops.
     # Using it on larger crops potentially misses bends in the track
    new_x_pts = []
    new_y_pts = []
    idx = 0
    while idx < len(x_pts_list):
        if idx == 0:
            new_x_pts.append(x_pts_list[idx])
            new_y_pts.append(y_pts_list[idx])
            idx += 1
        else:
            dx = x_pts_list[idx]-new_x_pts[-1]
            dy = y_pts_list[idx]-new_y_pts[-1]
            if ((abs(dx) >= padding) or (abs(dy) >= padding)):
                # print( "Padding Jump")
                # Next point is outside of the crop, redefine
                # a next step inside the crop in the correct direction
                if dx == 0: # Check for undefined slope, straight up or down
                    new_dy = 0
                    new_dx = 0
                    if dy > 0: # straight up
                        new_dy = padding
                    else: #straight down
                        new_dy = -1.0*padding
                    new_x_pts.append(new_x_pts[-1]+new_dx)
                    new_y_pts.append(new_y_pts[-1]+new_dy)
                elif dy == 0: #Straight right or left
                    new_dy = 0
                    new_dx = 0
                    if dx > 0: #straight right
                        new_dx = padding
                    else:
                        new_dx = -1.0*padding
                    new_x_pts.append(new_x_pts[-1]+new_dx)
                    new_y_pts.append(new_y_pts[-1]+new_dy)
                else:
                    # Eight Possible Cases: (think clock hours other than 3,6,9,12)
                    slope = float(dy)/float(dx)
                    new_dx = 0
                    new_dy = 0
                    if ((dx > 0) and (dy > 0) and (slope < 1)): #Clock Hour 2
                        new_dx = padding
                        new_dy = round(slope*padding)
                    elif ((dx > 0) and (dy > 0) and (slope >= 1)): #Clock Hour 1
                        new_dy = padding
                        new_dx = round(padding/slope)
                    elif ((dx < 0) and (dy > 0) and (slope <= -1)): #Clock Hour 11
                        new_dy = padding
                        new_dx = round(padding/slope)
                    elif ((dx < 0) and (dy > 0) and (slope > -1)): #Clock Hour 10
                        new_dx = -1.0*padding
                        new_dy = round(slope*-1.0*padding)
                    elif ((dx < 0) and (dy < 0) and (slope < 1)): #Clock Hour 8
                        new_dx = -1.0*padding
                        new_dy = round(slope*-1.0*padding)
                    elif ((dx < 0) and (dy < 0) and (slope >= 1)): #Clock Hour 7
                        new_dy = -1.0*padding
                        new_dx = round(-1.0*padding/slope)
                    elif ((dx > 0) and (dy < 0) and (slope <= -1)): #Clock Hour 5
                        new_dy = -1.0*padding
                        new_dx = round(-1.0*padding/slope)
                    elif ((dx > 0) and (dy < 0) and (slope > -1)): #Clock Hour 4
                        new_dx = padding
                        new_dy = round(padding*slope)
                    else:
                        print("PROBLEM, No clock options left.")
                        return -1
                    new_x_pts.append(new_x_pts[-1]+new_dx)
                    new_y_pts.append(new_y_pts[-1]+new_dy)
            else:
                # if you didn't have to place a special point,
                # advance the real track idx
                if always_edge == False:
                    # Point within crop, advance normally. If always_edge is true then
                    # points are not placed as normal, but only interpolated at the
                    # edges of the crop box.
                    if x_pts_list[idx] != new_x_pts[-1] and y_pts_list[idx] != new_y_pts[-1]:
                        new_x_pts.append(x_pts_list[idx])
                        new_y_pts.append(y_pts_list[idx])
                elif idx == len(x_pts_list)-1:
                    # Always add the last point of the track
                    new_x_pts.append(x_pts_list[idx])
                    new_y_pts.append(y_pts_list[idx])
                idx += 1
    return new_x_pts, new_y_pts
