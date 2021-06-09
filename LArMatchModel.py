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
import sys
sys.path.append('/home/jmills/workdir/ubdl/larflow/larmatchnet')
from larmatch import LArMatch
from larmatch_ssnet_classifier import LArMatchSSNetClassifier
from larmatch_keypoint_classifier import LArMatchKeypointClassifier
from larmatch_kpshift_regressor   import LArMatchKPShiftRegressor
from larmatch_affinityfield_regressor import LArMatchAffinityFieldRegressor

def get_larmatch_features_v2(PARAMS, img_2d):


    img_feat = img_2d
    return img_feat

def get_larmatch_features(PARAMS, img_2d):
    DEVICE=PARAMS['DEVICE']
    checkpointfile = PARAMS['LARMATCH_CKPT']
    checkpoint = torch.load( checkpointfile, map_location={"cuda:0":DEVICE,
                                                           "cuda:1":DEVICE} )
    NUM_PAIRS=30000
    ADC_PRODUCER='wire'
    CHSTATUS_PRODUCER='wire'
    USE_GAPCH=True
    RETURN_TRUTH=False
    BATCHSIZE = 1
    ARGS ={}
    ARGS['use_skip_limit'] = None
    ARGS['use_unet'] = True
    ARGS['supera'] = PARAMS['INFILE']
    ARGS['has_wirecell'] = PARAMS['MASK_WC']
    ARGS['tickbackwards'] = True
    ARGS['num_events'] = 1
    ARGS['min_score'] = 0.5
    ARGS['has_mc'] = False # no reason to use mc for lf here
    # DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
    # we use a config file
    preplarmatch = larflow.prep.PrepMatchTriplets()
    if ARGS['use_skip_limit']  is not None:
        print("Set Triplet Max where we will skip event: ",ARGS['use_skip_limit'] )
        preplarmatch.setStopAtTripletMax( True, ARGS['use_skip_limit']  )

    # MULTI-HEAD LARMATCH MODEL
    model_dict = {"larmatch":LArMatch(use_unet=ARGS['use_unet']).to(DEVICE)
                  # "ssnet":LArMatchSSNetClassifier().to(DEVICE),
                  # "kplabel":LArMatchKeypointClassifier().to(DEVICE),
                  # "kpshift":LArMatchKPShiftRegressor().to(DEVICE),
                  # "paf":LArMatchAffinityFieldRegressor(layer_nfeatures=[64,64,64]).to(DEVICE)
                  }

    # hack: for runnning with newer version of SCN where group-convolutions are possible
    for name,arr in checkpoint["state_larmatch"].items():
        #print(name,arr.shape)
        if ( ("resnet" in name and "weight" in name and len(arr.shape)==3) or
             ("stem" in name and "weight" in name and len(arr.shape)==3) or
             ("unet_layers" in name and "weight" in name and len(arr.shape)==3) or
             ("feature_layer.weight" == name and len(arr.shape)==3 ) ):
            print("reshaping ",name)
            checkpoint["state_larmatch"][name] = arr.reshape( (arr.shape[0], 1, arr.shape[1], arr.shape[2]) )

    for name,model in model_dict.items():
        # model.load_state_dict(checkpoint["state_"+name])
        model.eval()

    print("NOT loaded MODEL")

    # setup filename


    tickdir = larcv.IOManager.kTickForward
    if ARGS['tickbackwards']:
        tickdir = larcv.IOManager.kTickBackward
    io = larcv.IOManager( larcv.IOManager.kREAD, "larcvio", tickdir )
    io.add_in_file( ARGS['supera'] )
    io.set_verbosity(1)
    io.specify_data_read( larcv.kProductImage2D,  "larflow" )
    io.specify_data_read( larcv.kProductImage2D,  ADC_PRODUCER )
    io.specify_data_read( larcv.kProductChStatus, CHSTATUS_PRODUCER )
    if ARGS['has_wirecell']:
        io.specify_data_read( larcv.kProductChStatus, "thrumu" )
    io.reverse_all_products()
    io.initialize()

    sigmoid = torch.nn.Sigmoid()
    ssnet_softmax = torch.nn.Softmax(dim=0)

    NENTRIES = io.get_n_entries()

    if ARGS['num_events']>0 and ARGS['num_events']<NENTRIES:
        NENTRIES = ARGS['num_events']

    dt_prep  = 0.
    dt_chunk = 0.
    dt_net   = 0.
    dt_save  = 0.

    # setup the hit maker
    hitmaker = larflow.prep.FlowMatchHitMaker()
    hitmaker.set_score_threshold( ARGS['min_score'] )

    # setup badch maker
    badchmaker = ublarcvapp.EmptyChannelAlgo()

    # flush standard out buffer before beginning
    sys.stdout.flush()

    for ientry in range(NENTRIES):

        io.read_entry(ientry)

        print("==========================================")
        print("Entry {}".format(ientry))

        # clear the hit maker
        hitmaker.clear();

        adc_v = io.get_data(larcv.kProductImage2D,ADC_PRODUCER).Image2DArray()
        ev_badch    = io.get_data(larcv.kProductChStatus,CHSTATUS_PRODUCER)
        if ARGS['has_mc']:
            print("Retrieving larflow truth...")
            ev_larflow = io.get_data(larcv.kProductImage2D,"larflow")
            flow_v     = ev_larflow.Image2DArray()

        if ARGS['has_wirecell']:
            # make wirecell masked image
            print("making wirecell masked image")
            start_wcmask = time.time()
            ev_wcthrumu = io.get_data(larcv.kProductImage2D,"thrumu")
            ev_wcwire   = io.get_data(larcv.kProductImage2D,"wirewc")
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
        badch_v = badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, ev_badch )
        print("Number of badcv images: ",badch_v.size())
        gapch_v = badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 )
        for p in range(badch_v.size()):
            for c in range(badch_v[p].meta().cols()):
                if ( gapch_v[p].pixel(0,c)>0 ):
                    badch_v[p].paint_col(c,255);
        dt_badch = time.time()-t_badch
        print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump(), " elasped=",dt_badch," secs")

        # run the larflow match prep classes
        t_prep = time.time()
        preplarmatch.process( adc_v, badch_v, 10.0, False )
        if ARGS['has_mc']:
            print("processing larflow truth...")
            preplarmatch.make_truth_vector( flow_v )
        t_prep = time.time()-t_prep
        print("  time to prep matches: ",t_prep,"secs")
        dt_prep += t_prep

        # for debugging
        #if True:
        #    print("stopping after prep")
        #sys.exit(0)

        # Prep sparse ADC numpy arrays
        sparse_np_v = [ preplarmatch.make_sparse_image(p) for p in range(3) ]
        coord_t = [ torch.from_numpy( sparse_np_v[p][:,0:2].astype(np.long) ).to(DEVICE) for p in range(3) ]
        feat_t  = [ torch.from_numpy( sparse_np_v[p][:,2].reshape(  (coord_t[p].shape[0], 1) ) ).to(DEVICE) for p in range(3) ]

        # we can run the whole sparse images through the network
        #  to get the individual feature vectors at each coodinate
        t_start = time.time()
        print("computing features")
        with torch.no_grad():
            for i in range(len(coord_t)):
                print(coord_t[i].shape)
                print(feat_t[i].shape)
            outfeat_u, outfeat_v, outfeat_y = model_dict['larmatch'].forward_features( coord_t[0], feat_t[0],
                                                                                       coord_t[1], feat_t[1],
                                                                                       coord_t[2], feat_t[2],
                                                                                       1, verbose=False )
    feature_image_y = None
    return feature_image_y
