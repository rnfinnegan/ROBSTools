import SimpleITK as sitk
import numpy as np

def surfaceMetrics(imFixed, imMoving, verbose=False):
    """
    HD, meanSurfDist, medianSurfDist, maxSurfDist, stdSurfDist
    """
    hausdorffDistance = sitk.HausdorffDistanceImageFilter()
    hausdorffDistance.Execute(imFixed, imMoving)
    HD = hausdorffDistance.GetHausdorffDistance()

    meanSDList = []
    maxSDList = []
    stdSDList = []
    medianSDList = []
    numPoints = []
    for (imA, imB) in ((imFixed, imMoving), (imMoving, imFixed)):

        labelIntensityStat = sitk.LabelIntensityStatisticsImageFilter()
        referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(imA, squaredDistance=False, useImageSpacing=True))
        movingLabelContour = sitk.LabelContour(imB)
        labelIntensityStat.Execute(movingLabelContour,referenceDistanceMap)

        meanSDList.append(labelIntensityStat.GetMean(1))
        maxSDList.append(labelIntensityStat.GetMaximum(1))
        stdSDList.append(labelIntensityStat.GetStandardDeviation(1))
        medianSDList.append(labelIntensityStat.GetMedian(1))

        numPoints.append(labelIntensityStat.GetNumberOfPixels(1))

    if verbose:
        print("        Boundary points:  {0}  {1}".format(numPoints[0], numPoints[1]))

    meanSurfDist = np.dot(meanSDList, numPoints)/np.sum(numPoints)
    maxSurfDist = np.max(maxSDList)
    stdSurfDist = np.sqrt( np.dot(numPoints,np.add(np.square(stdSDList), np.square(np.subtract(meanSDList,meanSurfDist)))) )
    medianSurfDist = np.mean(medianSDList)

    return HD, meanSurfDist, medianSurfDist, maxSurfDist, stdSurfDist


def volumeMetrics(imFixed, imMoving):
    """
    DSC, VolOverlap, FracOverlap, TruePosFrac, TrueNegFrac, FalsePosFrac, FalseNegFrac
    """
    arrFixed = sitk.GetArrayFromImage(imFixed).astype(bool)
    arrMoving = sitk.GetArrayFromImage(imMoving).astype(bool)

    arrInter = arrFixed & arrMoving
    arrUnion = arrFixed | arrMoving

    voxVol = np.product(imFixed.GetSpacing())/1000. # Conversion to cm^3

    # 2|A & B|/(|A|+|B|)
    DSC =  (2.0*arrInter.sum())/(arrFixed.sum()+arrMoving.sum())

    #  |A & B|/|A | B|
    FracOverlap = arrInter.sum()/arrUnion.sum().astype(float)
    VolOverlap = arrInter.sum() * voxVol

    TruePos = arrInter.sum()
    TrueNeg = (np.invert(arrFixed) & np.invert(arrMoving)).sum()
    FalsePos = arrMoving.sum()-TruePos
    FalseNeg = arrFixed.sum()-TruePos

    #
    TruePosFrac = (1.0*TruePos)/(TruePos+FalseNeg)
    TrueNegFrac = (1.0*TrueNeg)/(TrueNeg+FalsePos)
    FalsePosFrac = (1.0*FalsePos)/(TrueNeg+FalsePos)
    FalseNegFrac = (1.0*FalseNeg)/(TruePos+FalseNeg)


    return DSC, VolOverlap, FracOverlap, TruePosFrac, TrueNegFrac, FalsePosFrac, FalseNegFrac


def CalcVolume(sitkImFixed):
    """
    Calculates binary volume
    """
    arr = sitk.GetArrayFromImage(sitkImFixed)
    voxVol = np.product(sitkImFixed.GetSpacing())/1000. # Conversion to cm^3
    return arr.sum()*voxVol
