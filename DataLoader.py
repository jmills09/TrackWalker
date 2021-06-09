import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
from MiscFunctions import cropped_np, unravel_array, reravel_array
from LArMatchModel import get_larmatch_features

def get_net_inputs_mc(PARAMS, START_ENTRY, END_ENTRY):
    # This function takes a root file path, a start entry and an end entry
    # and returns the mc track information in a form the network is
    # prepared to take as input
    print("Loading Network Inputs")
    steps_x = []
    steps_y = []
    full_image = []
    training_data = []
    event_ids = []
    step_dist_3d = []

    image_list, xs, ys, runs, subruns, events, filepaths, entries, track_pdgs = load_rootfile_training(PARAMS, step_dist_3d, START_ENTRY, END_ENTRY)
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
                    next_positions.append(np_step_target)
                    flat_next_positions.append(flat_np_step_target)
                    np_xy_shift = np.array([target_x*1.0-PARAMS['PADDING'],target_y*1.0-PARAMS['PADDING'] ])
                    xy_shifts.append(np_xy_shift)
                else:
                    if PARAMS['CENTERPOINT_ISEND']:
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
                training_data.append((flat_stepped_images,flat_next_positions))
                event_ids.append(EVENT_IDX)
            else:
                training_data.append((flat_stepped_images,xy_shifts))
                event_ids.append(EVENT_IDX)
    rse_pdg_dict = {}
    rse_pdg_dict['runs'] = runs
    rse_pdg_dict['subruns'] = subruns
    rse_pdg_dict['events'] = events
    rse_pdg_dict['filepaths'] = filepaths
    rse_pdg_dict['pdgs'] = track_pdgs
    rse_pdg_dict['file_idx'] = entries
    return training_data, full_image, steps_x, steps_y, event_ids, rse_pdg_dict

def is_inside_boundaries(xt,yt,zt,buffer = 0):
    x_in = (xt <  255.999-buffer) and (xt >    0.001+buffer)
    y_in = (yt <  116.499-buffer) and (yt > -116.499+buffer)
    z_in = (zt < 1036.999-buffer) and (zt >    0.001+buffer)
    if x_in == True and y_in == True and z_in == True:
        return True
    else:
        return False


def getprojectedpixel(meta,x,y,z):

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
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(infile)
    ioll.open()

    nentries_cv = iocv.get_n_entries()

    # Get Rid of those pesky IOManager Warning Messages (orig cxx)
	# larcv::logger larcv_logger
	# larcv::msg::Level_t log_level = larcv::msg::kCRITICAL
	# larcv_logger.force_level(log_level)
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

    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
    # for i in range(8,9):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(i)
        ioll.go_to(i)

        ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
        ev_mctrack = ioll.get_data(larlite.data.kMCTrack, "mcreco")
        # Get Wire ADC Image to a Numpy Array
        img_v = ev_wire.Image2DArray()
        y_wire_image2d = img_v[2]
        rows = y_wire_image2d.meta().rows()
        cols = y_wire_image2d.meta().cols()
        # for c in range(cols):
        #     for r in range(rows):
        #         y_wire_np[c][r] = y_wire_image2d.pixel(r,c)
        if PARAMS['USE_CONV_IM']:
            y_wire_np = get_larmatch_features(PARAMS, y_wire_image2d)
        else:
            y_wire_np = larcv.as_ndarray(y_wire_image2d) # I am Speed.
        full_image_list.append(y_wire_np)
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
        print("N Tracks", len(ev_mctrack))
        for mctrack in ev_mctrack:
            trk_idx += 1
            if mctrack.PdgCode() in PDG_to_Part and PDG_to_Part[mctrack.PdgCode()] not in ["PROTON","MUON"]:
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

    nentries_cv = iocv.get_n_entries()

    # Get Rid of those pesky IOManager Warning Messages (orig cxx)
	# larcv::logger larcv_logger
	# larcv::msg::Level_t log_level = larcv::msg::kCRITICAL
	# larcv_logger.force_level(log_level)
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

    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
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
