import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil

def is_inside_boundaries(xt,yt,zt,buffer = 0):
    x_in = (xt <  255.999-buffer) and (xt >    0.001+buffer)
    y_in = (yt <  116.499-buffer) and (yt > -116.499+buffer)
    z_in = (zt < 1036.999-buffer) and (zt >    0.001+buffer)
    if x_in == True and y_in == True and z_in == True:
        return True
    else:
        return False


def getprojectedpixel(meta,x,y,z,t):

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


def load_rootfile(filename,step_dist_3d,step_times):
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
    ev_trk_xpt_list = []
    ev_trk_ypt_list = []

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


    # for i in range(nentries):
    for i in range(1):
        iocv.read_entry(i)
        ioll.go_to(i)

        ev_wire    = iocv.get_data(larcv.kProductImage2D,"wire")
        ev_mctrack = ioll.get_data(larlite.data.kMCTrack, "mcreco")
        # Get Wire ADC Image to a Numpy Array
        img_v = ev_wire.Image2DArray()
        y_wire_image2d = img_v[2]
        rows = y_wire_image2d.meta().rows()
        cols = y_wire_image2d.meta().cols()
        y_wire_np = np.zeros((cols,rows))
        for c in range(cols):
            for r in range(rows):
                y_wire_np[c][r] = y_wire_image2d.pixel(r,c)
        full_image_list.append(y_wire_np)

        # Get MC Track X Y Points
        meta = y_wire_image2d.meta()
        trk_xpt_list = []
        trk_ypt_list = []
        trk_idx = -1
        for mctrack in ev_mctrack:
            trk_idx += 1
            print("Track Index:",trk_idx)
            if mctrack.PdgCode()  in PDG_to_Part:
                print("     Track PDG:", PDG_to_Part[mctrack.PdgCode()])
            else:
                print("     Track PDG:", mctrack.PdgCode())
            print("     Track Length", mctrack_length(mctrack))
            xpt_list = []
            ypt_list = []
            last_x   = 0
            last_y   = 0
            last_z   = 0
            last_t   = 0
            step_idx = -1
            for mcstep in mctrack:
                step_idx += 1
                x = mcstep.X()
                y = mcstep.Y()
                z = mcstep.Z()
                if is_inside_boundaries(x,y,z) == False:
                    continue
                t = mcstep.T()
                if step_idx != 0:
                    step_dist = ((x-last_x)**2 + (y-last_y)**2 + (z-last_z)**2)**0.5
                    step_dist_3d.append(step_dist)
                    step_times.append(t-last_t)
                last_x = x
                last_y = y
                last_z = z
                last_t = t
                # if trk_idx == 6:
                #     print(str(round(x)).zfill(4),str(round(y)).zfill(4),str(round(z)).zfill(4),str(round(t)).zfill(4))
                col,row = getprojectedpixel(meta,x,y,z,t)
                if len(xpt_list) !=0 and col == xpt_list[len(xpt_list)-1] and row == ypt_list[len(ypt_list)-1]:
                    continue
                xpt_list.append(col)
                ypt_list.append(row)
            trk_xpt_list.append(xpt_list)
            trk_ypt_list.append(ypt_list)
        ev_trk_xpt_list.append(trk_xpt_list)
        ev_trk_ypt_list.append(trk_ypt_list)
    return full_image_list, ev_trk_xpt_list, ev_trk_ypt_list

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
