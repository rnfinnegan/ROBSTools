#!/usr/bin/python

"""
Script name:
Author:      Robert Finnegan
Date:        XX/XX/XX
Description
-----------

"""

from __future__ import print_function
import os, sys

idInList  = ['04','05','17','22','32']
idOutList = ['01','02','03','04','05']

inputDir = '/home/robbie/Documents/University/PhD/Research/CardiacContouring/Data'
structList =   ['AORTICVALVE',
                'ASCENDINGAORTA',
                'DESCENDINGAORTA',
                'LANTDESCARTERY',
                'LCIRCUMFLEXARTERY',
                'LCORONARYARTERY',
                'LEFTATRIUM',
                'LEFTVENTRICLE',
                'MITRALVALVE',
                'PULMONARYARTERY',
                'PULMONICVALVE',
                'RCORONARYARTERY',
                'RIGHTATRIUM',
                'RIGHTVENTRICLE',
                'TRICUSPIDVALVE',
                'WHOLEHEART']

for (idIn, idOut) in zip(idInList, idOutList):
    imFileIn = '{0}/Images/Case_{1}.nii.gz'.format(inputDir, idIn)
    imFileOut = './Case_{0}/Case_{0}_CT.nii.gz'.format(idOut)

    os.system('cp {0} {1}'.format(imFileIn, imFileOut))

    for struct in structList:
        sFileIn = '{0}/Structures/Case_{1}_{2}_mean.nii.gz'.format(inputDir, idIn, struct)
        sFileOut = './Case_{0}/Case_{0}_SProb_{1}.nii.gz'.format(idOut, struct)

        os.system('cp {0} {1}'.format(sFileIn, sFileOut))

        sFileIn = '{0}/Structures/Case_{1}_{2}_vote.nii.gz'.format(inputDir, idIn, struct)
        sFileOut = './Case_{0}/Case_{0}_SBin_{1}.nii.gz'.format(idOut, struct)

        os.system('cp {0} {1}'.format(sFileIn, sFileOut))
