# TrackWalker
Neural Net Solutions to reconstructing particle tracks in MicroBooNE

This repository is designed to develop a DL NN solution to reconstructing particle tracks in either 2D or 3D in MicroBooNE's LArTPC detector. Contact joshua.mills@tufts.edu 
or taritree.wongjirad@tufts.edu for inquiries. 

Very broadly, the repository is broken into 3 files:
DataLoader.py         - This is responsible for loading in the ROOT objects and transforming them into Net Inputs
TrackerWalkerTest.py  - This file contains the network model, and a script to run the network. Call with `python TrackerWalkerTest.py`
MiscFunctions.py      - All custom functions outside those living in UBDL are placed in MiscFunctions.py

Special Requirements:
UBDL: https://github.com/LArbys/ubdl/
ROOT: https://root.cern/
