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
sys.path.append('/home/jmills/workdir/ubdl_gen2/larflow/larmatchnet')
from larmatchvoxel import LArMatchVoxel

from ctypes import c_int,c_double
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME


import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
import larvoxel_engine
from MiscFunctions import save_im

class LArVoxelModel:
    def __init__(self, PARAMS):
        self.PARAMS    = PARAMS


        checkpointfile = self.PARAMS['LARVOXEL_CKPT']
        print("CHECKPOINT FILE: ",checkpointfile)
        checkpoint = torch.load( checkpointfile, map_location={"cuda:0":"cpu",
                                                               "cuda:1":"cpu"} )
        self.ADC_PRODUCER        ="wire"
        self.CHSTATUS_PRODUCER   ="wire"
        self.USE_GAPCH=True
        self.RETURN_TRUTH=False
        self.BATCHSIZE = 4
        self.supera            = self.PARAMS['INFILE']
        self.has_wirecell      = self.PARAMS['MASK_WC']
        self.has_mc            = False #Don't want truth for training TrackNet
        self.use_skip_limit    = None
        self.tickbackwards     = True
        self.min_score         = 0.5

        # DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
        # we use a config file
        self.preplarmatch = larflow.prep.PrepMatchTriplets()
        if self.use_skip_limit is not None:
            print("Set Triplet Max where we will skip event: ",self.use_skip_limit)
            self.preplarmatch.setStopAtTripletMax( True, self.use_skip_limit )

        # MULTI-HEAD LARMATCH MODEL
        config = self.load_config_file( self.PARAMS['LARVOX_CFG'] )
        self.model, self.model_dict = larvoxel_engine.get_larmatch_model( config, dump_model=False )
        self.model = self.model.to(torch.device(self.PARAMS['DEVICE']))
        checkpoint_data = larvoxel_engine.load_model_weights( self.model, checkpointfile )
        print("loaded MODEL")

        # setup filename


        tickdir = larcv.IOManager.kTickForward
        if self.tickbackwards:
            tickdir = larcv.IOManager.kTickBackward
        self.io = larcv.IOManager( larcv.IOManager.kREAD, "larcvio", tickdir )
        self.io.add_in_file( self.supera )
        self.io.set_verbosity(1)
        self.io.specify_data_read( larcv.kProductImage2D,  "larflow" )
        self.io.specify_data_read( larcv.kProductImage2D,  self.ADC_PRODUCER )
        self.io.specify_data_read( larcv.kProductChStatus, self.CHSTATUS_PRODUCER)
        if self.has_wirecell:
            self.io.specify_data_read( larcv.kProductChStatus, "thrumu" )
        self.io.reverse_all_products()
        self.io.initialize()



        self.ssnet_softmax = torch.nn.Softmax(dim=1)
        self.larmatch_softmax = torch.nn.Softmax( dim=1 )


        self.dt_prep  = 0.
        self.dt_chunk = 0.
        self.dt_net   = 0.
        self.dt_save  = 0.

        # setup the hit maker
        self.hitmaker = larflow.voxelizer.LArVoxelHitMaker()
        self.hitmaker._voxelizer.set_voxel_size_cm( 1.0 )
        self.hitmaker._hit_score_threshold = self.min_score

        # setup badch maker
        self.badchmaker = ublarcvapp.EmptyChannelAlgo()

        # flush standard out buffer before beginning
        sys.stdout.flush()
        print("Loaded and Initialized LARVOX Model")

    def get_larvoxel_features(self, ientry):
        print("\n\n")
        print("Getting LArVoxel Features\n----------------------------------\n")

        self.io.read_entry(ientry)

        print("==========================================")
        print("Entry {}".format(ientry))

        # clear the hit maker
        self.hitmaker.clear();

        ev_wire       = self.io.get_data(larcv.kProductImage2D,self.ADC_PRODUCER)
        run           = ev_wire.run()
        subrun        = ev_wire.subrun()
        event         = ev_wire.event()
        adc_v         = ev_wire.Image2DArray()
        meta          = adc_v[2].meta()
        ev_badch      = self.io.get_data(larcv.kProductChStatus,self.CHSTATUS_PRODUCER)

        if self.has_mc:
            print("Retrieving larflow truth...")
            ev_larflow = self.io.get_data(larcv.kProductImage2D,"larflow")
            flow_v     = ev_larflow.Image2DArray()

        if self.has_wirecell:
            # make wirecell masked image
            print("making wirecell masked image")
            start_wcmask = time.time()
            ev_wcthrumu = self.io.get_data(larcv.kProductImage2D,"thrumu")
            ev_wcwire   = self.io.get_data(larcv.kProductImage2D,"wirewc")
            for p in range(adc_v.size()):
                adc = larcv.Image2D(adc_v[p]) # a copy
                np_adc = larcv.as_ndarray(adc)
                np_wc  = larcv.as_ndarray(ev_wcthrumu.Image2DArray()[p])
                np_adc[ np_wc>0.0 ] = 0.0
                masked = larcv.as_image2d_meta( np_adc, adc.meta() )
                ev_wcwire.Append(masked)
            adc_v = ev_wcwire.Image2DArray()
            end_wcmask = time.time()
            print("time to mask: ",end_wcmask-start_wcmask," secs")

        t_badch = time.time()
        badch_v = self.badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, ev_badch )
        print("Number of badcv images: ",badch_v.size())
        gapch_v = self.badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 )
        for p in range(badch_v.size()):
            for c in range(badch_v[p].meta().cols()):
                if ( gapch_v[p].pixel(0,c)>0 ):
                    badch_v[p].paint_col(c,255);
        dt_badch = time.time()-t_badch
        print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump(), " elasped=",dt_badch," secs")

        # run the larflow match prep classes
        t_prep = time.time()

        self.hitmaker._voxelizer.process_fullchain( self.io, self.ADC_PRODUCER, self.CHSTATUS_PRODUCER, False )

        if self.has_mc:
            print("processing larflow truth...")
            self.preplarmatch.make_truth_vector( flow_v )

        data = self.hitmaker._voxelizer.make_voxeldata_dict()
        print(data.keys())
        for k in data.keys():
            try:
                print(k,data[k].shape)
            except:
                print(k,len(data[k]))

        coord   = torch.from_numpy( data["voxcoord"] ).int().to(torch.device(self.PARAMS['DEVICE']))
        feat    = torch.from_numpy( np.clip( data["voxfeat"]/40.0, 0, 10.0 ) ).to(torch.device(self.PARAMS['DEVICE']))
        coords, feats = ME.utils.sparse_collate(coords=[coord], feats=[feat])
        xinput = ME.SparseTensor( features=feats, coordinates=coords.to(torch.device(self.PARAMS['DEVICE'])) )

        t_prep = time.time()-t_prep
        print("  time to prep matches: ",t_prep,"secs")
        self.dt_prep += t_prep

        # we can run the whole sparse images through the network
        #  to get the individual feature vectors at each coodinate
        t_start = time.time()

        # use UNET portion to first get feature vectors
        self.model.eval()
        with torch.no_grad():
            trackerNetFeats = self.model_dict["larmatch"].forwardTracker( xinput ).F
            dt_net_feats = time.time()-t_start
            print("forward time: ",dt_net_feats,"secs")
            self.dt_net += dt_net_feats
            print(type(trackerNetFeats))


        print("time elapsed: prep=",self.dt_prep," net=",self.dt_net," save=",self.dt_save)

        # End of flow direction loop
        sys.stdout.flush()
        voxcoords_np     = data["voxcoord"]
        voxfeats_np      = trackerNetFeats.cpu().detach().numpy()
        # print(voxcoords_np.shape,"voxcoords_np.shape")
        # print(voxfeats_np.shape,"voxfeats_np.shape")
        feats            =  np.concatenate((voxcoords_np,voxfeats_np),axis=1)
        print(feats.shape,"feats.shape")
        return feats, run, subrun, event, meta

    def load_config_file(self, cfg, dump_to_stdout=False ):
        stream = open(cfg, 'r')
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)
        if dump_to_stdout:
            for key, value in dictionary.items():
                print (key + " : " + str(value))
        stream.close()
        return dictionary
