import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

def LabelMean(label):
	"""
	Determines the centroid slice locations and returns as an array
	"""
	nd=sitk.GetArrayFromImage(label)
	yloc = nd.sum(axis=(0,2)).argmax()
	xloc = nd.sum(axis=(0,1)).argmax()
	zloc = nd.sum(axis=(1,2)).argmax()
	return xloc, yloc, zloc
	
	
def PlotSeq(image, label, [xloc, yloc, zloc])

image = sitk.ReadImage('../../../Cardiac Contouring/Images/Case_08.nii.gz')
organ1 = sitk.ReadImage('../../../Cardiac Contouring/Structures/Case_08_CombinedLung.nii.gz')
organ2 = sitk.ReadImage('../../../Cardiac Contouring/Structures/Case_08_WHOLEHEART.nii.gz')

# [z,y,x]
xloc, yloc, zloc = LabelMean(organ2)

fig=plt.figure(figsize=(18,6))

#Coronal
xcontour = sitk.GetArrayFromImage(sitk.LabelContour(organ2))[:,196,:][::-1]
xcontour = np.array(np.where(xcontour>0))[::-1]
ximage = sitk.GetArrayFromImage(image)[:,196,:][::-1]

#Transverse
ycontour = sitk.GetArrayFromImage(sitk.LabelContour(organ2))[64,:,:]
ycontour = np.array(np.where(ycontour>0))[::-1]
yimage = sitk.GetArrayFromImage(image)[64,:,:]
#Sagittal
zcontour = sitk.GetArrayFromImage(sitk.LabelContour(organ2))[:,:,253][::-1]
zcontour = np.array(np.where(zcontour>0))[::-1]
zimage = sitk.GetArrayFromImage(image)[:,:,253][::-1]


ax=fig.add_subplot(1,3,1)
ax.imshow(ximage, interpolation='none', cmap='gray')
ax.hold(True)
ax.scatter(xcontour[0], xcontour[1], edgecolor='none', c='g', s=10)

ax=fig.add_subplot(1,3,2)
ax.imshow(yimage, interpolation='none', cmap='gray')
ax.hold(True)
ax.scatter(ycontour[0], ycontour[1], edgecolor='none', c='g', s=10)

ax=fig.add_subplot(1,3,3)
ax.imshow(zimage, interpolation='none', cmap='gray')
ax.hold(True)
ax.scatter(zcontour[0], zcontour[1], edgecolor='none', c='g', s=10)

fig.show()
