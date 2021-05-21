import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil


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

def load_rootfile(filename):
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
        for mctrack in ev_mctrack:
            xpt_list = []
            ypt_list = []
            for mcstep in mctrack:
                x = mcstep.X()
                y = mcstep.Y()
                z = mcstep.Z()
                t = mcstep.T()
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
