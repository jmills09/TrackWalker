import ROOT
import numpy as np
import os, sys
import signal
import socket
from datetime import datetime
from tensorboardX import SummaryWriter
from larlite import larutil
import math

# These functions are to help voxelize the MicroBooNE detector for 3D track reco

class Voxelator:
    def __init__(self, PARAMS, detectorType="MICROBOONE"):
        self.setupDetector(detectorType)


    def setupDetector(self, detectorType="MICROBOONE"):
        # Takes the detectorType and returns a 3D array of zeros voxelizing the
        # detector
        if detectorType == "MICROBOONE":
            # Return voxelized uboone detector
            self.voxelSize = 0.3
            self.driftVel  = larutil.LArProperties.GetME().DriftVelocity()

            self.xmin = 0. #(2399-3200)*0.5*driftVel
            self.ymin = -116.5
            self.zmin = 0.
            self.xmax = 256.
            self.ymax = 116.5
            self.zmax = 1037.
            # Plus one for rounding error
            self.nXVoxels = int((self.xmax - self.xmin) / self.voxelSize + 1)
            self.nYVoxels = int((self.ymax - self.ymin) / self.voxelSize + 1)
            self.nZVoxels = int((self.zmax - self.zmin) / self.voxelSize + 1)
        elif detectorType == "LARVOXNETMICROBOONE":
            self.voxelSize = 1.0
            self.driftVel  = larutil.LArProperties.GetME().DriftVelocity()

            self.xmin = -43.9749 #(2399-3200)*0.5*driftVel
            self.ymin = -120.0
            self.zmin = 0.
            self.xmax = 288.719
            self.ymax = 120.0
            self.zmax = 1037.
            # Plus one for rounding error
            self.nXVoxels = int(math.ceil((self.xmax - self.xmin) / self.voxelSize))
            self.nYVoxels = int(math.ceil((self.ymax - self.ymin) / self.voxelSize))
            self.nZVoxels = int(math.ceil((self.zmax - self.zmin) / self.voxelSize))
        else:
            print("DetectorType not set for Voxelization")
            assert 1==2

    def getVoxelCoord(self,hit3d):
        # Takes a cartesian 3D hit and returns the 3D voxel coordinate
        xVoxIdx = int((hit3d[0] - self.xmin) / self.voxelSize)
        yVoxIdx = int((hit3d[1] - self.ymin) / self.voxelSize)
        zVoxIdx = int((hit3d[2] - self.zmin) / self.voxelSize)
        # Place coord at edge of detector if outside detector
        if xVoxIdx < 0:
            xVoxIdx = 0
        elif xVoxIdx > self.nXVoxels:
            xVoxIdx = self.nXVoxels
        if yVoxIdx < 0:
            yVoxIdx = 0
        elif yVoxIdx > self.nYVoxels:
            yVoxIdx = self.nYVoxels
        if zVoxIdx < 0:
            zVoxIdx = 0
        elif zVoxIdx > self.nZVoxels:
            zVoxIdx = self.nZVoxels
        # Return voxel space coord
        return [xVoxIdx, yVoxIdx, zVoxIdx]

    def get3dCoord(self, voxel3d):
        # Takes a 3d voxel index and returns the center of the voxel in 3d
        # cartesian coordinates
        # 0.5 gives center of voxel
        x = (voxel3d[0]+0.5)*self.voxelSize + self.xmin
        y = (voxel3d[1]+0.5)*self.voxelSize + self.ymin
        z = (voxel3d[2]+0.5)*self.voxelSize + self.zmin
        return [x,y,z]

    def getimgcoords(self, voxel3d):
        # takes a 3d voxel and returns the imgcoords from getprojectedpixel
        xyz = self.get3dCoord(voxel3d)


    def saveDetector3D(self,vox3d_v):
        # Takes a list of voxels and puts them in a th3d

        mins = [9999,9999,9999]
        maxes = [-1,-1,-1]
        for vox in vox3d_v:
            for p in range(3):
                if vox[p] < mins[p]:
                    mins[p] = vox[p]
                elif vox[p] > maxes[p]:
                    maxes[p] = vox[p]
        canv = ROOT.TCanvas('canv','canv',5000,4000)
        bins = [maxes[p]+1-mins[p] for p in range(3)]
        for p in range(3):
            print(bins[p], mins[p], maxes[p]+1)
        print()
        print('Detector','Detector',bins[0],0,bins[0],bins[1],0,bins[1],bins[2],0,bins[2])
        histDet = ROOT.TH3D('Detector','Detector',bins[0],mins[0],maxes[0]+1,bins[1],mins[1],maxes[1]+1,bins[2],mins[2],maxes[2]+1)
        # histDet = ROOT.TH3D('Detector','Detector',bins[0],0,bins[0],bins[1],0,bins[1],bins[2],0,bins[2])
        #
        for vox in vox3d_v:
            val = 0
            if len(vox) == 3:
                val = 1
            elif len(vox) == 4:
                val = 100#vox[3]
            else:
                print("Voxels not size 3 or 4, undefined behavior")
                assert 1==2
            print(vox)
            histDet.SetBinContent(vox[0]-mins[0]+1 ,vox[1]-mins[1]+1, vox[2]-mins[2]+1, val)
            # histDet.Fill(vox[0]-mins[0],vox[1]-mins[1],vox[2]-mins[2], val)
            # print("Bin",histDet.FindBin(vox[0],vox[1],vox[3]))

        # histDet = ROOT.TH3D('Detector','Detector',10,0,10,10,0,10,10,0,10)
        # for x in range(10):
        #     histDet.SetBinContent(x,x,x,x)
        canv = ROOT.TCanvas('canv','canv',1200,1000)
        histDet.Draw("")
        canv.SaveAs("TestNone.png")
        histDet.Draw("iso")
        canv.SaveAs('TestIso.png')
        histDet.Draw("box2 z")
        canv.SaveAs('TestBox.png')
