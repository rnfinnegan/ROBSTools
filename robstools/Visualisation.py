#!/usr/bin/python

"""
Module name: Visualisation
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- Medical Imaging
- AutoSegmentation Output: deformation fields, delineations
- Slice generation
---------------------------------

"""

from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid, ImageGrid


def returnSlice(axis,index):
    if axis=='x':
        s = (slice(None), slice(None), index)
    if axis=='y':
        s = (slice(None), index, slice(None))
    if axis=='z':
        s = (index, slice(None), slice(None))

    return s

def keepLargestConnectedComponent(binaryImage):
    componentLabelledImage = sitk.ConnectedComponent(binaryImage)
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(componentLabelledImage)
    labels = labelShapeFilter.GetLabels()
    largestLabel = max(labels, key=labelShapeFilter.GetNumberOfPixels)
    return sitk.Equal(componentLabelledImage, largestLabel)

def probabilityThreshold(inputImage, lowerThreshold=0.5):
    runningImage = sitk.BinaryThreshold(inputImage, lowerThreshold=lowerThreshold, upperThreshold=1.0)
    runningImage = sitk.BinaryFillhole(runningImage)
    return keepLargestConnectedComponent(runningImage)

def displaySlice(im, axis='ortho', cut=None, figSize=6, cmap=plt.cm.Greys, window=[-250, 500]):
    if type(im)==sitk.Image:
        nda = sitk.GetArrayFromImage(im)
    else:
        print('Image type not recognised, must be SimpleITK image format.')

    (AxSize, CorSize, SagSize) = nda.shape
    spPlane, _, spSlice = im.GetSpacing()
    asp = (1.0*spSlice)/spPlane

    if axis=='ortho':
        fSize = (figSize, figSize*(asp*AxSize+CorSize)/(1.0*SagSize+CorSize))

        fig, ((axAx, blank), (axCor, axSag)) = plt.subplots(2, 2, figsize=fSize, gridspec_kw = {'height_ratios':[(CorSize)/(asp*AxSize),1]})
        blank.axis('off')

        if not cut:
            sliceAx  = int(AxSize/2.0)
            sliceCor = int(CorSize/2.0)
            sliceSag = int(SagSize/2.0)

            cut = [sliceAx, sliceCor, sliceSag]

        sAx  = returnSlice('z', cut[0])
        sCor = returnSlice('y', cut[1])
        sSag = returnSlice('x', cut[2])

        imAx = axAx.imshow(nda.__getitem__(sAx), aspect=1.0, interpolation=None, cmap=plt.cm.gray, clim=(window[0], window[0]+window[1]))
        imCor = axCor.imshow(nda.__getitem__(sCor), origin='lower', aspect=asp, interpolation=None, cmap=plt.cm.gray, clim=(window[0], window[0]+window[1]))
        imSag = axSag.imshow(nda.__getitem__(sSag), origin='lower', aspect=asp, interpolation=None, cmap=plt.cm.gray, clim=(window[0], window[0]+window[1]))

        axAx.axis('off')
        axCor.axis('off')
        axSag.axis('off')

        fig.subplots_adjust(left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0)

    else:
        if axis=='x' or axis=='sag':
            fSize = (figSize, figSize*(asp*SagSize)/(1.0*CorSize))
            fig, ax = plt.subplots(1,1, figsize=(fSize))
            org = 'lower'
            if not cut:
                cut = int(SagSize/2.0)

        if axis=='y' or axis=='cor':
            fSize = (figSize, figSize*(asp*AxSize)/(1.0*SagSize))
            fig, ax = plt.subplots(1,1, figsize=(fSize))
            org = 'lower'
            if not cut:
                cut = int(CorSize/2.0)

        if axis=='z' or axis=='ax':
            asp=1
            fSize = (figSize, figSize*(asp*CorSize)/(1.0*SagSize))
            fig, ax = plt.subplots(1,1, figsize=(fSize))
            org = 'upper'
            if not cut:
                cut = int(AxSize/2.0)

        s = returnSlice(axis, cut)
        ax.imshow(nda.__getitem__(s), aspect=asp, interpolation=None, origin=org, cmap=plt.cm.gray, clim=(window[0], window[0]+window[1]))
        ax.axis('off')

    return fig, axis, cut

def overlayContour(contourIm, fig, axis, cut, colorBase=plt.cm.Blues):

    # Test types of contours
    if type(contourIm)==sitk.Image:
        #print('Single contour detected.')
        nda = sitk.GetArrayFromImage(contourIm)

        color = [colorBase(126)]

        flag='single'

    else:
        try:
            if len(contourIm) > 1:
                #print('Contour list detected.')
                ndaList = [sitk.GetArrayFromImage(i) for i in contourIm]

                colors = colorBase(np.array(255.0/len(ndaList)*np.arange(len(ndaList)), dtype=np.int))
                colors = colors[:,None]*np.ones((len(ndaList),4))

                flag = 'multi'
        except:
            print('Could not determine contour image type.')
            return None


    # Test types of axes
    axes = fig.axes
    if len(axes)==1:
        ax = axes[0]
        s = returnSlice(axis, cut)
        if flag=='single':
            try:
                ax.contour(nda.__getitem__(s), colors=color, levels=[0], alpha=0.8, linewidths=1.5)
            except:
                None

        elif flag=='multi':
            for index, nda in enumerate(ndaList):
                try:
                    ax1.contour(nda.__getitem__(s), colors=colors[index], levels=[0], alpha=0.8, linewidths=1)
                except:
                    None

    elif len(axes)==4:
        axAx, blank, axCor, axSag = axes

        sAx  = returnSlice('z', cut[0])
        sCor = returnSlice('y', cut[1])
        sSag = returnSlice('x', cut[2])

        if flag=='single':
            try:
                axAx.contour(nda.__getitem__(sAx), colors=color, levels=[0], alpha=0.8, linewidths=1.5)
            except:
                None
            try:
                axSag.contour(nda.__getitem__(sSag), colors=color, levels=[0], alpha=0.8, linewidths=1.5)
            except:
                None
            try:
                axCor.contour(nda.__getitem__(sCor), colors=color, levels=[0], alpha=0.8, linewidths=1.5)
            except:
                None

        elif flag=='multi':
            for index, nda in enumerate(ndaList):
                try:
                    axAx.contour(nda.__getitem__(sAx), colors=colors[index], levels=[0], alpha=0.8, linewidths=1)
                except:
                    None
                try:
                    axSag.contour(nda.__getitem__(sSag), colors=colors[index], levels=[0], alpha=0.8, linewidths=1)
                except:
                    None
                try:
                    axCor.contour(nda.__getitem__(sCor), colors=colors[index], levels=[0], alpha=0.8, linewidths=1)
                except:
                    None

    return fig

def overlayBox(box, fig, axis, color='r'):

    # Test types of contours
    if len(box)!=6 or any([type(i)!=int for i in box]):
        print('Box not formatted correctly.')
        return None

    else:
        sag0, cor0, ax0, sagD, corD, axD = box

    # Test types of axes
    axes = fig.axes
    if len(axes)==1:
        ax = axes[0]

        if axis=='z' or axis=='ax':
            axAx.plot([sag0,sag0,sag0+sagD,sag0+sagD, sag0], [cor0,cor0+corD,cor0+corD,cor0,cor0], lw=2, c=color)
        if axis=='y' or axis=='cor':
            axCor.plot([sag0,sag0+sagD,sag0+sagD,sag0,sag0], [ax0,ax0,ax0+axD,ax0+axD,ax0], lw=2, c=color)
        if axis=='x' or axis=='sag':
            axSag.plot([cor0,cor0+corD,cor0+corD,cor0,cor0], [ax0,ax0,ax0+axD,ax0+axD,ax0], lw=2, c=color)


    elif len(axes)==4:
        axAx, blank, axCor, axSag = axes

        axAx.plot([sag0,sag0,sag0+sagD,sag0+sagD, sag0], [cor0,cor0+corD,cor0+corD,cor0,cor0], lw=2, c=color)
        axCor.plot([sag0,sag0+sagD,sag0+sagD,sag0,sag0], [ax0,ax0,ax0+axD,ax0+axD,ax0], lw=2, c=color)
        axSag.plot([cor0,cor0+corD,cor0+corD,cor0,cor0], [ax0,ax0,ax0+axD,ax0+axD,ax0], lw=2, c=color)

    return fig




def main(arguments):
    return True

if __name__ == '__main__':
    main()
