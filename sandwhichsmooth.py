# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:31:30 2019

@author: shini
"""

import numpy
import imageio
import fftcompression
import matplotlib.pyplot as mpl
import scipy.interpolate as interp
import scipy.signal as scisig
import scipy.special as scispe
from skimage import restoration
from scipy.misc import face
from scipy.signal.signaltools import wiener

def findmin(image):
    minimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if i == 0:
                if j == 0:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j + 1][i]:
                        minimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
                else:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j + 1][i] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
            elif i == image.shape[1] - 1:
                if j == 0:
                    if image[j][i] <= image[j][i - 1] and image[j][i] <= image[j + 1][i]:
                        minimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] <= image[j][i - 1] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
                else:
                    if image[j][i] <= image[j][i - 1] and image[j][i] <= image[j + 1][i] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
            else:
                if j == 0:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j][i - 1] and image[j][i] <= image[j + 1][i]:
                        minimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j][i - 1] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
                else:
                    if image[j][i] <= image[j][i + 1] and image[j][i] <= image[j][i - 1] and image[j][i] <= image[j + 1][i] and image[j][i] <= image[j - 1][i]:
                        minimg[j][i] = 1
    return minimg

def findmax(image):
    maximg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if i == 0:
                if j == 0:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j + 1][i]:
                        maximg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
                else:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j + 1][i] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
            elif i == image.shape[1] - 1:
                if j == 0:
                    if image[j][i] >= image[j][i - 1] and image[j][i] >= image[j + 1][i]:
                        maximg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] >= image[j][i - 1] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
                else:
                    if image[j][i] >= image[j][i - 1] and image[j][i] >= image[j + 1][i] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
            else:
                if j == 0:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j][i - 1] and image[j][i] >= image[j + 1][i]:
                        maximg[j][i] = 1
                elif j == image.shape[0] - 1:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j][i - 1] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
                else:
                    if image[j][i] >= image[j][i + 1] and image[j][i] >= image[j][i - 1] and image[j][i] >= image[j + 1][i] and image[j][i] >= image[j - 1][i]:
                        maximg[j][i] = 1
    return maximg

def findmid(image):
    midimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if i == 0:
                if j == 0:
                    midimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    midimg[j][i] = 1
                else:
                    midimg[j][i] = 1
            elif i == image.shape[1] - 1:
                if j == 0:
                    midimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    midimg[j][i] = 1
                else:
                    midimg[j][i] = 1
            else:
                if j == 0:
                    midimg[j][i] = 1
                elif j == image.shape[0] - 1:
                    midimg[j][i] = 1
                else:
                    if ((image[j][i] >= image[j][i + 1]) != (image[j][i] >= image[j][i - 1])) or ((image[j][i] >= image[j + 1][i]) != (image[j][i] >= image[j - 1][i])) or ((image[j][i] >= image[j + 1][i]) != (image[j][i] >= image[j][i + 1])):
                        midimg[j][i] = 1
    return midimg

def findlow(image):
    lowimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            lowcnt = 0
            if i > 0:
                if image[j][i] <= image[j][i - 1]:
                    lowcnt = lowcnt + 1
            if j > 0:
                if image[j][i] <= image[j - 1][i]:
                    lowcnt = lowcnt + 1
            if i < image.shape[1] - 2:
                if image[j][i] <= image[j][i + 1]:
                    lowcnt = lowcnt + 1
            if j < image.shape[0] - 2:
                if image[j][i] <= image[j + 1][i]:
                    lowcnt = lowcnt + 1
            if lowcnt >= 2:
                lowimg[j][i] = 1
    return lowimg

def findhih(image):
    hihimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            hihcnt = 0
            if i > 0:
                if image[j][i] >= image[j][i - 1]:
                    hihcnt = hihcnt + 1
            if j > 0:
                if image[j][i] >= image[j - 1][i]:
                    hihcnt = hihcnt + 1
            if i < image.shape[1] - 2:
                if image[j][i] >= image[j][i + 1]:
                    hihcnt = hihcnt + 1
            if j < image.shape[0] - 2:
                if image[j][i] >= image[j + 1][i]:
                    hihcnt = hihcnt + 1
            if hihcnt >= 2:
                hihimg[j][i] = 1
    return hihimg

def interpmerge(image):
    maximg = findmax(image)
    minimg = findmin(image)
    maxx = []
    maxy = []
    maxz = []
    minx = []
    miny = []
    minz = []
    minz = []
    xarr = []
    yarr = []
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if maximg[j][i] == 1:
                maxx.append(i)
                maxy.append(j)
                maxz.append(image[j][i])
            if minimg[j][i] == 1:
                minx.append(i)
                miny.append(j)
                minz.append(image[j][i])
            xarr.append(i)
            yarr.append(j)
    maxtrp = interp.griddata((maxx, maxy), maxz, (xarr, yarr), method = 'cubic')
    mintrp = interp.griddata((minx, miny), minz, (xarr, yarr), method = 'cubic')
    mrgtrp = numpy.zeros(image.shape)
    for k in range(len(xarr)):
        if numpy.isnan(maxtrp[k]) and numpy.isnan(mintrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = image[yarr[k]][xarr[k]]
        elif numpy.isnan(maxtrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + mintrp[k]) / 2
        elif numpy.isnan(mintrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + maxtrp[k]) / 2
        else:
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + maxtrp[k] + mintrp[k]) / 3
    return mrgtrp

def extramerge(image):
    maximg = findhih(image)
    minimg = findlow(image)
    maxx = []
    maxy = []
    maxz = []
    minx = []
    miny = []
    minz = []
    minz = []
    xarr = []
    yarr = []
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if maximg[j][i] == 1:
                maxx.append(i)
                maxy.append(j)
                maxz.append(image[j][i])
            if minimg[j][i] == 1:
                minx.append(i)
                miny.append(j)
                minz.append(image[j][i])
            xarr.append(i)
            yarr.append(j)
    maxtrp = interp.griddata((maxx, maxy), maxz, (xarr, yarr), method = 'cubic')
    mintrp = interp.griddata((minx, miny), minz, (xarr, yarr), method = 'cubic')
    mrgtrp = numpy.zeros(image.shape)
    for k in range(len(xarr)):
        if numpy.isnan(maxtrp[k]) and numpy.isnan(mintrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = image[yarr[k]][xarr[k]]
        elif numpy.isnan(maxtrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + mintrp[k]) / 2
        elif numpy.isnan(mintrp[k]):
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + maxtrp[k]) / 2
        else:
            mrgtrp[yarr[k]][xarr[k]] = (image[yarr[k]][xarr[k]] + maxtrp[k] + mintrp[k]) / 3
    return mrgtrp

def hillshire(actimg):
    convoi = numpy.ones((5, 5)) * 5
    convld = scisig.convolve2d(actimg, convoi, mode = 'same')
    return restoration.wiener(convld, convoi, 1100)

def cycleinterp(image, radius):
    xarr = []
    yarr = []
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            xarr.append(i)
            yarr.append(j)
    maximg = findmax(image)
    minimg = findmin(image)
    midimg = findmid(image)
    tarden = 1 / (4 * radius**2)
    maxwht = (numpy.sum(maximg) / (image.shape[0] * image.shape[1]) - tarden) / (.5 - tarden)
    print(tarden)
    #maxrad = (2 * numpy.pi * numpy.sum(maximg) / (image.shape[0] * image.shape[1]))
    #minrad = (2 * numpy.pi * numpy.sum(minimg) / (image.shape[0] * image.shape[1]))
    #midrad = (numpy.log(20) * (image.shape[0] * image.shape[1]) / (numpy.pi * numpy.sum(midimg)))**.5
    #midwht = .5 * (1 - scispe.erf(2.33 * (midrad - ((radius + 1) / 2)) / (radius - 1)))
    cpyimg = numpy.copy(image)
    while maxwht > .0525: # midwht < .9475 )
        print([maxwht, ((image.shape[0] * image.shape[1]) / numpy.sum(maximg))**.5 / 2])
        #maximg = findmax(cpyimg)
        #minimg = findmin(cpyimg)
        #midimg = findmid(image)
        maxx = []
        maxy = []
        maxz = []
        minx = []
        miny = []
        minz = []
        minz = []
        midx = []
        midy = []
        midz = []
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                if maximg[j][i] == 1:
                    maxx.append(i)
                    maxy.append(j)
                    maxz.append(cpyimg[j][i])
                if minimg[j][i] == 1:
                    minx.append(i)
                    miny.append(j)
                    minz.append(cpyimg[j][i])
                if midimg[j][i] == 1:
                    midx.append(i)
                    midy.append(j)
                    midz.append(cpyimg[j][i])
        maxtrp = interp.griddata((maxx, maxy), maxz, (xarr, yarr), method = 'cubic')
        mintrp = interp.griddata((minx, miny), minz, (xarr, yarr), method = 'cubic')
        midtrp = interp.griddata((midx, midy), midz, (xarr, yarr), method = 'cubic')
        mrgtrp = numpy.zeros(image.shape)
        for k in range(len(xarr)):
            if numpy.isnan(maxtrp[k]) and numpy.isnan(mintrp[k]) and numpy.isnan(midtrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = cpyimg[yarr[k]][xarr[k]]
            elif numpy.isnan(maxtrp[k]) and numpy.isnan(mintrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = midtrp[k]
            elif numpy.isnan(mintrp[k]) and numpy.isnan(midtrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = maxtrp[k]
            elif numpy.isnan(maxtrp[k]) and numpy.isnan(midtrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = mintrp[k]
            elif numpy.isnan(maxtrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = midtrp[k] * (1 - maxwht) + mintrp[k] * maxwht
            elif numpy.isnan(mintrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = midtrp[k] * (1 - maxwht) + maxtrp[k] * maxwht
            elif numpy.isnan(midtrp[k]):
                mrgtrp[yarr[k]][xarr[k]] = (maxtrp[k] + mintrp[k]) / 2
            else:
                mrgtrp[yarr[k]][xarr[k]] = midtrp[k] * (1 - maxwht) + (maxtrp[k] + mintrp[k]) * maxwht / 2
        cpyimg = mrgtrp
        midimg = findmid(cpyimg)
        maximg = findmax(cpyimg)
        minimg = findmin(cpyimg)
        #midrad = (numpy.log(20) * (image.shape[0] * image.shape[1]) / (numpy.pi * numpy.sum(midimg)))**.5
        #midwht = .5 * (1 - scispe.erf(2.33 * (midrad - ((radius + 1) / 2)) / (radius - 1)))
        maxwht = (numpy.sum(maximg) / (image.shape[0] * image.shape[1]) - tarden) / (.5 - tarden)
    return cpyimg

def rstrct(orgimg, nwdims):
    # This method condenses an image to a smaller, grainier version of itself. 
    # This compression will be done by averaging the pixels within a box 
    # defined by orgimg.size / nwdims
    newimg = []
    counts = []
    for j in range(int(nwdims[1])):
        newrow = []
        cntrow = []
        for i in range(int(nwdims[0])):
            newrow.append(0)
            cntrow.append(0)
        newimg.append(newrow)
        counts.append(cntrow)
    stpsze = [orgimg.shape[0] / nwdims[0], orgimg.shape[1] / nwdims[1]]
    for j in range(orgimg.shape[0]):
        for i in range(orgimg.shape[1]):
            rstctx = int(numpy.floor(i / stpsze[1]))
            rstcty = int(numpy.floor(j / stpsze[0]))
            newimg[rstcty][rstctx] = newimg[rstcty][rstctx] + orgimg[j][i]
            counts[rstcty][rstctx] = counts[rstcty][rstctx] + 1
    newimg = numpy.array(newimg) / numpy.array(counts)
    return newimg

def prlong(orgimg, nwdims):
    # This method expands an image to a larger version of itself. This 
    # expansion will be done by taking a single pixel and duplicating it to 
    # each pixel that would fit within the area of the original pixel if it 
    # were subdivided to make enough pixels to fit the new dimensions
    newimg = []
    for j in range(nwdims[0]):
        newrow = []
        for i in range(nwdims[1]):
            newrow.append([])
        newimg.append(newrow)
    stpsze = [nwdims[0] / orgimg.shape[0], nwdims[1] / orgimg.shape[1]]
    for j in range(nwdims[0]):
        for i in range(nwdims[1]):
            prlngx = int(numpy.floor(i / stpsze[1]))
            prlngy = int(numpy.floor(j / stpsze[0]))
            newimg[j][i] = orgimg[prlngy][prlngx]
    return newimg

def compressmaxima(image, lowcom, hihcom):
    liklim = numpy.zeros(image.shape)
    for i in range(lowcom, hihcom + 1):
        compim = rstrct(image, (int(numpy.round(image.shape[0] / 2**i)), int(numpy.round(image.shape[1] / 2**i))))
        maximg = prlong(findmax(compim), image.shape)
        liklim = liklim + numpy.array(maximg)
    return liklim

if __name__ == '__main__':
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    if ftype == 'csv':
        imgdat = numpy.genfromtxt(flname, delimiter = ',')
    else:
        imgdat = imageio.imread(flname)
    #itrpon = interpmerge(imgdat)
    #itrptw = interpmerge(itrpon)
    #itrpth = interpmerge(itrptw)
    #f, axearr = mpl.subplots(2, 2)
    #axearr[0][0].imshow(imgdat)
    #axearr[0][1].imshow(itrpon)
    #axearr[1][0].imshow(itrptw)
    #axearr[1][1].imshow(itrpth)
# =============================================================================
#     mapone = compressmaxima(imgdat, 2, 2)
#     maptwo = compressmaxima(imgdat, 2, 3)
#     mapthr = compressmaxima(imgdat, 2, 4)
#     mpl.figure()
#     mpl.imshow(imgdat)
#     mpl.figure()
#     mpl.imshow(mapone)
#     mpl.figure()
#     mpl.imshow(maptwo)
#     mpl.figure()
#     mpl.imshow(mapthr)
# =============================================================================
    idlrad = fftcompression.findcharacteristicperiod(imgdat, noisereductionfactor = 2, mode = 'average', feedback = True)
    refind = cycleinterp(imgdat, idlrad)
    viener = wiener(imgdat, (5, 5))
    #mpl.figure()
    #mpable = mpl.imshow(imgdat)
    #mpl.colorbar(mpable)
    mpl.figure()
    mpable = mpl.imshow(refind)
    mpl.colorbar(mpable)
    mpl.figure()
    mpable = mpl.imshow(viener)
    mpl.colorbar(mpable)
# =============================================================================
#     mpl.figure()
#     mpable = mpl.imshow(findmid(imgdat))
#     mpl.colorbar(mpable)
# =============================================================================
# =============================================================================
#     mpl.figure()
#     mpable = mpl.imshow(itrpth)
#     mpl.colorbar(mpable)
# =============================================================================
