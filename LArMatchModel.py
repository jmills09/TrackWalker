import ROOT
import numpy as np
import time

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
larcv.PSet
from ublarcvapp import ublarcvapp
from larflow import larflow
import torch
import os,sys
sys.path.append('/home/jmills/workdir/ubdl/larflow/larmatchnet')
from larmatch import LArMatch
from larmatch_ssnet_classifier import LArMatchSSNetClassifier
from larmatch_keypoint_classifier import LArMatchKeypointClassifier
from larmatch_kpshift_regressor   import LArMatchKPShiftRegressor
from larmatch_affinityfield_regressor import LArMatchAffinityFieldRegressor

from ctypes import c_int,c_double
import numpy as np
from MiscFunctions import save_im

class LArMatchConvNet:
    def __init__(self, PARAMS):
        self.params    = PARAMS

        checkpointfile = self.params['LARMATCH_CKPT']
        checkpoint = torch.load( checkpointfile, map_location={"cuda:0":self.params['DEVICE'],
                                                               "cuda:1":self.params['DEVICE']} )
        self.adc_producer      = 'wire'
        self.chstatus_producer = 'wire'
        self.use_gapch         = True
        self.use_skip_limit    = None
        self.use_unet          = True
        self.supera            = self.params['INFILE']
        self.has_wirecell      = self.params['MASK_WC']
        self.tickbackwards     = True
        self.min_score         = 0.5
        self.model             = LArMatch(use_unet=self.use_unet).to(self.params['DEVICE'])
        self.preplarmatch      = larflow.prep.PrepMatchTriplets()
        self.iomanager         = None # Is Set Below
        self.nentries          = 0    # Is Set Below
        self.hitmaker          = None # Is Set Below
        self.badchmaker        = None # Is Set Below



        if self.use_skip_limit  is not None:
            print("Set Triplet Max where we will skip event: ",self.use_skip_limit )
            self.preplarmatch.setStopAtTripletMax( True, self.use_skip_limit  )


        self.model.load_state_dict(checkpoint["state_larmatch"])
        self.model.eval()


        tickdir = larcv.IOManager.kTickForward
        if self.tickbackwards:
            tickdir = larcv.IOManager.kTickBackward
        self.iomanager         = larcv.IOManager( larcv.IOManager.kREAD, "larcvio", tickdir )
        self.iomanager.add_in_file( self.supera )
        self.iomanager.set_verbosity(5)
        self.iomanager.specify_data_read( larcv.kProductImage2D,  "larflow" )
        self.iomanager.specify_data_read( larcv.kProductImage2D,  self.adc_producer )
        self.iomanager.specify_data_read( larcv.kProductChStatus, self.chstatus_producer )
        if self.has_wirecell:
            self.iomanager.specify_data_read( larcv.kProductChStatus, "thrumu" )
        self.iomanager.reverse_all_products()
        self.iomanager.initialize()

        self.nentries = self.iomanager.get_n_entries()


        # setup the hit maker
        self.hitmaker = larflow.prep.FlowMatchHitMaker()
        self.hitmaker.set_score_threshold( self.min_score )

        # setup badch maker
        self.badchmaker = ublarcvapp.EmptyChannelAlgo()

        # flush standard out buffer before beginning
        sys.stdout.flush()
        print("Loaded and Initialized MODEL")

    def get_larmatch_features(self, entry):

        ############################
        # Entry loop was here:
        ############################
        if (entry > self.nentries) or ( entry < 0 ):
            print("ERROR, entry outside file size", entry, self.nentries)
            return -1

        t_start_larmatch = time.time()

        self.iomanager.read_entry(entry)

        print("==========================================")
        print("LArMatch ConvNet on Entry {}\n\n\n\n\n\n\n".format(entry))

        # clear the hit maker
        self.hitmaker.clear();
        ev_adc = self.iomanager.get_data(larcv.kProductImage2D, self.adc_producer)
        adc_v = ev_adc.Image2DArray()
        ev_badch    = self.iomanager.get_data(larcv.kProductChStatus, self.chstatus_producer)
        print("SKIPPING LARMATCHFEATGRAB")
        return np.zeros((3456, 1008, 16)), ev_adc.run(), ev_adc.subrun(), ev_adc.event(), adc_v[2].meta()


        if self.has_wirecell:
            # make wirecell masked image
            # print("making wirecell masked image")
            ev_wcthrumu = self.iomanager.get_data(larcv.kProductImage2D,"thrumu")
            ev_wcwire   = self.iomanager.get_data(larcv.kProductImage2D,"wirewc")
            for p in range(adc_v.size()):
                adc = larcv.Image2D(adc_v[p]) # a copy
                np_adc = larcv.as_ndarray(adc)
                np_wc  = larcv.as_ndarray(ev_wcthrumu.Image2DArray()[p])
                np_adc[ np_wc>0.0 ] = 0.0
                masked = larcv.as_image2d_meta( np_adc, adc.meta() )
                ev_wcwire.Append(masked)
            adc_v = ev_wcwire.Image2DArray()

        badch_v = self.badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, ev_badch )
        # print("Number of badcv images: ",badch_v.size())
        gapch_v = self.badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 )
        for p in range(badch_v.size()):
            for c in range(badch_v[p].meta().cols()):
                if ( gapch_v[p].pixel(0,c)>0 ):
                    badch_v[p].paint_col(c,255);
        # print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump())

        # run the larflow match prep classes
        self.preplarmatch.process( adc_v, badch_v, 10.0, False )

        # Prep sparse ADC numpy arrays
        sparse_np_v = [ self.preplarmatch.make_sparse_image(p) for p in range(2,3) ]
        coord_t = [ torch.from_numpy( sparse_np_v[p][:,0:2].astype(np.long) ).to(self.params['DEVICE']) for p in range(0,1) ]
        feat_t  = [ torch.from_numpy( sparse_np_v[p][:,2].reshape(  (coord_t[p].shape[0], 1) ) ).to(self.params['DEVICE']) for p in range(0,1) ]


        with torch.no_grad():
            outfeat_y = self.model.forward_features_oneplane( coord_t[0], feat_t[0],
                                                           1, verbose=False )

        # Take output tensor and make it a dense numpy array
        detach_outfeat_y = outfeat_y.cpu().detach().numpy()
        detach_outfeat_y = np.reshape(detach_outfeat_y,(outfeat_y.shape[1],outfeat_y.shape[2],outfeat_y.shape[3]))
        detach_outfeat_y_transposed = np.transpose(detach_outfeat_y,(2,1,0))
        #slice away the padded extra pixels larmatchnet uses
        detach_outfeat_y_transposed = detach_outfeat_y_transposed[:,0:1008,:]


        # y_wire_image2d = adc_v[2]
        # y_wire_np = larcv.as_ndarray(y_wire_image2d)
        # save_im(y_wire_np,savename="larmatchfeat_im_test/adc_check_yplane",canv_x=4000,canv_y=1000)
        # print("\n\nShapecheck:",detach_outfeat_y_transposed.shape)
        # for f in range(16):
        #     save_im(detach_outfeat_y_transposed[:,:,f],savename='larmatchfeat_im_test/larmatch_feat_'+str(f),canv_x=4000,canv_y=1000)
        # convert_cmd = "convert "+"larmatchfeat_im_test/*.png "+'larmatchfeat_im_test/featcheck.pdf'
        # print(convert_cmd)
        # os.system(convert_cmd)
        # assert 1==2
        print("\n\n\n\nLArMatch Done\n")
        print(time.time() - t_start_larmatch, "Seconds for LArMatch\n\n\n\n")
        return detach_outfeat_y_transposed, ev_adc.run(), ev_adc.subrun(), ev_adc.event(), adc_v[2].meta()
