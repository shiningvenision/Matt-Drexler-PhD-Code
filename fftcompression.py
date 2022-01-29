# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:31:07 2020

@author: shini
"""

import numpy
#import imageio
import scipy.interpolate as interp
#import scipy.optimize as sciopt
import matplotlib.pyplot as mpl
#from mpl_toolkits.mplot3d import Axes3D

def doublegaussian(pos, xone, hone, sone, xtwo, htwo, stwo):
    return hone * numpy.exp(-1 * (pos - xone)**2 / (2 * sone**2)) + htwo * numpy.exp(-1 * (pos - xtwo)**2 / (2 * stwo**2))

def findcompressionradius(stemimage, noisy = False):
    # Find the FFT of the original image and turn it into a human-readable 
    # format
    stemfft = numpy.fft.fft2(stemimage)**2
    stemfft = numpy.log10((numpy.real(stemfft)**2 + numpy.imag(stemfft)**2)**.5)
    rearfft = numpy.zeros(stemfft.shape)
    if stemimage.shape[0] % 2 == 0:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2), :]
    else:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2 + 1):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2 + 1)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2 + 1):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2 + 1), :]
    if noisy:
        mpl.figure()
        mpl.imshow(stemfft)
    # Get the radial intensity profile of the 
    centerpoint = numpy.array([int(stemfft.shape[0] / 2), int(stemfft.shape[1] / 2)])
    intensityprofile = numpy.zeros((int(numpy.ceil(((stemfft.shape[0] / 2)**2 + (stemfft.shape[1] / 2)**2)**.5) + 1), 2))
    for i in range(stemfft.shape[0]):
        for j in range(stemfft.shape[1]):
            actualdistance = ((i - centerpoint[0])**2 + (j - centerpoint[1])**2)**.5
            lowerguidepointpercent = 1 - (actualdistance - numpy.floor(actualdistance))
            intensityprofile[int(actualdistance)][0] = intensityprofile[int(actualdistance)][0] + stemfft[i][j] * lowerguidepointpercent
            intensityprofile[int(actualdistance)][1] = intensityprofile[int(actualdistance)][1] + lowerguidepointpercent
            intensityprofile[int(actualdistance) + 1][0] = intensityprofile[int(actualdistance) + 1][0] + stemfft[i][j] * (1 - lowerguidepointpercent)
            intensityprofile[int(actualdistance) + 1][1] = intensityprofile[int(actualdistance) + 1][1] + (1 - lowerguidepointpercent)
    intensityprofile[:, 0] = intensityprofile[:, 0] / intensityprofile[:, 1]
    # Find the minimum radius that contains all diffraction points. This can 
    # be done by sorting the intensity profile and forming a Lorenz curve, 
    # then finding the two rectangles contained within the curve with the 
    # highest area. 
    sortedradius = numpy.sort(intensityprofile[:, 0])
    lorenzcurve = numpy.zeros(intensityprofile.shape)
    for i in range(sortedradius.shape[0]):
        lorenzcurve[i][0] = sortedradius[i]
        lorenzcurve[i][1] = (i + 1) / sortedradius.shape[0]
    interpolatedx = numpy.arange(sortedradius[0], sortedradius[sortedradius.shape[0] - 1], .01)
    interpolatedy = interp.griddata((lorenzcurve[:, 0]), (lorenzcurve[:,1]), interpolatedx)
    bigarea = 0
    combop = [0, 1]
    for i in range(1, interpolatedx.shape[0] - 2):
        for j in range(2, interpolatedx.shape[0] - 1):
            area = (interpolatedy[i] - numpy.min(interpolatedy)) * (interpolatedx[interpolatedx.shape[0] - 1] - interpolatedx[i]) + (interpolatedy[j] - interpolatedy[i]) * (interpolatedx[interpolatedx.shape[0] - 1] - interpolatedx[j])
            if area > bigarea:
                combop = [i, j]
                bigarea = area
    # Determining the correct radius is difficult. You can't just use the 
    # highest x-value with a y-value greater than the value, so you need to 
    # find the point with the lowest value that has the most number of points 
    # greater than the threshold below it and less than the threshold above it
    if noisy:
        print([[interpolatedx[combop[0]], interpolatedy[combop[0]]], [interpolatedx[combop[1]], interpolatedy[combop[1]]]])
    threshold = interpolatedx[combop[0]]
    #threshold = numpy.sum(stemfft) / (stemfft.shape[0] * stemfft.shape[1])
    numberhigher = 0
    numberlower = 0
    for i in range(intensityprofile.shape[0]):
        if intensityprofile[i][0] > threshold:
            numberhigher = numberhigher + 1
        else:
            numberlower = numberlower + 1
    scorearray = numpy.zeros((intensityprofile.shape[0]))
    highcount = 0
    lowcount = numberlower
    highscore = 0
    for i in range(intensityprofile.shape[0]):
        if intensityprofile[i][0] > threshold:
            highcount = highcount + 1
        else:
            lowcount = lowcount - 1
        scorearray[i] = highcount / (i + 1) + lowcount / (intensityprofile.shape[0] - i)
        if scorearray[i] > scorearray[highscore]:
            highscore = i
    compressionfactor = highscore / (stemimage.shape[0] / 2)
    if noisy:
        print(compressionfactor**-1)
        mpl.figure()
        mpl.scatter(range(intensityprofile.shape[0]), intensityprofile[:, 0])
        mpl.scatter(range(intensityprofile.shape[0]), scorearray)
        mpl.plot(range(intensityprofile.shape[0]), threshold * numpy.ones((intensityprofile.shape[0])))
        mpl.figure()
        mpl.scatter(interpolatedx, interpolatedy)
    return int(numpy.ceil(compressionfactor**-1))

def findaveragefftintensity(stemimage):
    # Find the FFT of the original image and turn it into a human-readable 
    # format
    stemfft = numpy.fft.fft2(stemimage)**2
    stemfft = numpy.log10((numpy.real(stemfft)**2 + numpy.imag(stemfft)**2)**.5)
    rearfft = numpy.zeros(stemfft.shape)
    rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]]
    rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2)]
    stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :]
    stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2), :]
    averageintensity = numpy.sum(stemfft) / (stemfft.shape[0] * stemfft.shape[1])
    print(averageintensity)

def fftfilter(stemimage, mitigatefactor = .01, bounds = [[0, 1024, 0, 180]], noisy = False):
    center = [int(numpy.floor(stemimage.shape[1] / 2)), int(numpy.floor(stemimage.shape[0] / 2))]
    stemfft = numpy.fft.fft2(stemimage)
    rearfft = numpy.zeros(stemfft.shape)
    if stemimage.shape[0] % 2 == 0:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2), :]
    else:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2 + 1):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2 + 1)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2 + 1):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2 + 1), :]
    if noisy:
        displaypre = numpy.log10((numpy.real(stemfft)**2 + numpy.imag(stemfft)**2)**.5)
        mpl.figure()
        mpl.imshow(displaypre)
    for j in range(stemfft.shape[0]):
        for i in range(stemfft.shape[1]):
            outbounds = True
            deltax = i - center[0]
            deltay = j - center[1]
            radiusij = (deltax**2 + deltay**2)**.5
            angleij = 0
            if deltax == 0:
                if deltay > 0:
                    angleij = 90
                elif deltay < 0:
                    angleij = 270
                else:
                    angleij = 90
                    outbounds = False
            else:
                angleij = numpy.arccos(deltax / radiusij)
                if deltay < 0:
                    angleij = 2 * numpy.pi - angleij
                angleij = angleij * 180 / numpy.pi
            for k in range(len(bounds)):
                if radiusij >= bounds[k][0] and radiusij <= bounds[k][1]:
                    if (angleij >= bounds[k][2] and angleij <= bounds[k][3]) or (angleij >= bounds[k][2] + 180 and angleij <= bounds[k][3] + 180):
                        outbounds = False
            if outbounds:
                stemfft[j][i] = stemfft[j][i] * mitigatefactor
    if noisy:
        displaypost = numpy.log10((numpy.real(stemfft)**2 + numpy.imag(stemfft)**2)**.5)
        mpl.figure()
        mpl.imshow(displaypost)
    if stemimage.shape[0] % 2 == 0:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2), :]
    else:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2 + 1):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2 + 1)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2 + 1):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2 + 1), :]
    rebuilt = numpy.fft.ifft2(stemfft)
    reconstructed = (numpy.real(rebuilt)**2 + numpy.imag(rebuilt)**2)**.5
    return reconstructed

def findcharacteristicperiod(stemimage, noisereductionfactor = 1, vectortolerance = 1, connecttolerance = 2, mode = 'lowest', feedback = False):
    stemfft = numpy.fft.fft2(stemimage)
    stemfft = numpy.log10((numpy.real(stemfft)**2 + numpy.imag(stemfft)**2)**.5)
    rearfft = numpy.zeros(stemfft.shape)
    if stemimage.shape[0] % 2 == 0:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2), :]
    else:
        rearfft[:, 0:int(rearfft.shape[1] / 2)] = stemfft[:, int(rearfft.shape[1] / 2 + 1):rearfft.shape[1]]
        rearfft[:, int(rearfft.shape[1] / 2):rearfft.shape[1]] = stemfft[:, 0:int(rearfft.shape[1] / 2 + 1)]
        stemfft[0:int(rearfft.shape[0] / 2), :] = rearfft[int(rearfft.shape[0] / 2 + 1):rearfft.shape[0], :]
        stemfft[int(rearfft.shape[0] / 2):rearfft.shape[0], :] = rearfft[0:int(rearfft.shape[0] / 2 + 1), :]
    if feedback:
        mpl.figure()
        mpl.imshow(stemimage)
        mpl.figure()
        mpl.imshow(stemfft)
    fftspots = identifyfftpattern(stemfft, compressionfactor = noisereductionfactor, falloff = vectortolerance, gracerange = connecttolerance, noisy = feedback)
    if mode == 'lowest':
        frequency = 0
        for n in range(len(fftspots)):
            if ((fftspots[n][0] - stemfft.shape[1] / 2)**2 + (fftspots[n][1] - stemfft.shape[0] / 2)**2)**.5 > frequency:
                frequency = ((fftspots[n][0] - stemfft.shape[1] / 2)**2 + (fftspots[n][1] - stemfft.shape[0] / 2)**2)**.5
    elif mode == 'average':
        frequency = 0
        frequencycount = 0
        for n in range(len(fftspots)):
            if numpy.abs(fftspots[n][0] - stemfft.shape[1] / 2) > 1 and numpy.abs(fftspots[n][1] - stemfft.shape[0] / 2) > 1:
                frequency = frequency + ((fftspots[n][0] - stemfft.shape[1] / 2)**2 + (fftspots[n][1] - stemfft.shape[0] / 2)**2)**.5
                frequencycount = frequencycount + 1
        frequency = frequency / frequencycount
    elif mode == 'highest':
        frequency = stemfft.shape[0]
        for n in range(len(fftspots)):
            if ((fftspots[n][0] - stemfft.shape[1] / 2)**2 + (fftspots[n][1] - stemfft.shape[0] / 2)**2)**.5 < frequency and numpy.abs(fftspots[n][0] - stemfft.shape[1] / 2) > 1 and numpy.abs(fftspots[n][1] - stemfft.shape[0] / 2) > 1:
                frequency = ((fftspots[n][0] - stemfft.shape[1] / 2)**2 + (fftspots[n][1] - stemfft.shape[0] / 2)**2)**.5
    return int(numpy.ceil(stemfft.shape[0] / frequency))

def identifyfftpattern(orgimg, compressionfactor = 1, falloff = 1, gracerange = 2, noisy = False):
    cndnsd = poolimage(orgimg, twopower = compressionfactor)
    prbmap = numpy.zeros(cndnsd.shape)
    derivs = getsobel(cndnsd, convolve = False)
    dermag = (derivs[0]**2 + derivs[1]**2)**.5
    if noisy:
        mpl.figure()
        mpl.imshow(cndnsd)
        mpl.figure()
        mpl.imshow(dermag)
    avgmag = numpy.average(dermag)
    stdmag = numpy.std(dermag)
    edglst = []
    for l in range(dermag.shape[0]):
        for k in range(dermag.shape[1]):
            if dermag[l][k] >= avgmag + 3 * stdmag:
                edglst.append([k, l])
    n = 0
    while n < len(edglst):
        isinvl = True
        m = 0
        while m < len(edglst) and isinvl:
            if n != m:
                if ((edglst[m][0] - edglst[n][0])**2 + (edglst[m][1] - edglst[n][1])**2)**.5 <= gracerange and ((derivs[0][edglst[n][1]][edglst[n][0]] * derivs[0][edglst[m][1]][edglst[m][0]]) + (derivs[1][edglst[n][1]][edglst[n][0]] * derivs[1][edglst[m][1]][edglst[m][0]])) >= 0:
                    isinvl = False
                else:
                    m = m + 1
            else:
                m = m + 1
        if isinvl:
            edglst.pop(n)
        else:
            n = n + 1
    edgcpy = []
    for edg in edglst:
        edgcpy.append(edg)
    grplst = []
    while len(edgcpy) > 0:
        curgrp = [edgcpy.pop(0)]
        m = 0
        while m < len(edgcpy):
            nonmem = True
            n = 0
            while n < len(curgrp) and nonmem:
                if ((edgcpy[m][0] - curgrp[n][0])**2 + (edgcpy[m][1] - curgrp[n][1])**2)**.5 <= gracerange and (derivs[0][edgcpy[m][1]][edgcpy[m][0]] * derivs[0][curgrp[n][1]][curgrp[n][0]] + derivs[1][edgcpy[m][1]][edgcpy[m][0]] * derivs[1][curgrp[n][1]][curgrp[n][0]]) > 0:
                    nonmem = False
                else:
                    n = n + 1
            if nonmem:
                m = m + 1
            else:
                curgrp.append(edgcpy.pop(m))
        #if len(curgrp) > 5:
            #grplst.append(curgrp)
        anglst = []
        for m in range(len(curgrp)):
            if derivs[1][curgrp[m][1]][curgrp[m][0]] < 0:
                anglst.append(2 * numpy.pi - numpy.arccos(derivs[0][curgrp[m][1]][curgrp[m][0]] / (derivs[0][curgrp[m][1]][curgrp[m][0]]**2 + derivs[1][curgrp[m][1]][curgrp[m][0]]**2)**.5))
            else:
                anglst.append(numpy.arccos(derivs[0][curgrp[m][1]][curgrp[m][0]] / (derivs[0][curgrp[m][1]][curgrp[m][0]]**2 + derivs[1][curgrp[m][1]][curgrp[m][0]]**2)**.5))
        angarr = numpy.sort(numpy.array(anglst))
        biggap = 0
        for m in range(angarr.shape[0] - 1):
            if angarr[m + 1] - angarr[m] > biggap:
                biggap = angarr[m + 1] - angarr[m]
        if 2 * numpy.pi - angarr[angarr.shape[0] - 1] + angarr[0] > biggap:
            biggap = 2 * numpy.pi - angarr[angarr.shape[0] - 1] + angarr[0]
        if biggap < 5 * numpy.pi / 4:
            #print(angarr)
            #print(biggap)
            grplst.append(curgrp)
    cenlst = []
    for n in range(len(grplst)):
        higprb = 0
        curcen = [0, 0]
        prbmap = numpy.zeros(cndnsd.shape)
        for j in range(prbmap.shape[0]):
            for i in range(prbmap.shape[1]):
                for m in range(len(grplst[n])):
                    if (i - (grplst[n][m][0] + 1)) * derivs[0][grplst[n][m][1]][grplst[n][m][0]] + (j - (grplst[n][m][1] + 1)) * derivs[1][grplst[n][m][1]][grplst[n][m][0]] >= 0:
                        sidrad = ((i - (grplst[n][m][0] + 1)) * derivs[1][grplst[n][m][1]][grplst[n][m][0]] + (j - (grplst[n][m][1] + 1)) * -1 * derivs[0][grplst[n][m][1]][grplst[n][m][0]]) / dermag[grplst[n][m][1]][grplst[n][m][0]]
                        prbmap[j][i] = prbmap[j][i] + numpy.exp(-1 * sidrad**2 / (2 * falloff**2)) / (2**.5 * numpy.pi**.5 * falloff)
                if prbmap[j][i] > higprb:
                    higprb = prbmap[j][i]
                    curcen = [i, j]
        cenlst.append(curcen)
    if noisy:
        markedimage = numpy.zeros((cndnsd.shape[0], cndnsd.shape[1], 3))
        for n in range(len(cenlst)):
            groupcolor = numpy.random.rand(3)
            for m in range(len(grplst[n])):
                markedimage[grplst[n][m][1] + 1, grplst[n][m][0] + 1, :] = groupcolor
            markedimage[cenlst[n][1], cenlst[n][0], :] = (groupcolor + numpy.ones((3))) / 2
        mpl.figure()
        mpl.imshow(markedimage)
    gsslst = []
    for n in range(len(grplst)):
        estrad = 0
        weight = 0
        for m in range(len(grplst[n])):
            estrad = estrad + ((cenlst[n][0] - (grplst[n][m][0] + 1))**2 + (cenlst[n][1] - (grplst[n][m][1] + 1))**2)**.5 * dermag[grplst[n][m][1]][grplst[n][m][0]]
            weight = weight + dermag[grplst[n][m][1]][grplst[n][m][0]]
        gsslst.append([cenlst[n][0], cenlst[n][1], estrad / weight])
    fnllst = []
    for n in range(len(gsslst)):
        eyerad = 2**compressionfactor * int(numpy.ceil(2 * gsslst[n][2]))
        bscord = [gsslst[n][0] * 2**compressionfactor, gsslst[n][1] * 2**compressionfactor]
        locscr = 0
        loccrd = [gsslst[n][0] * 2**compressionfactor, gsslst[n][1] * 2**compressionfactor]
        for v in range(2**compressionfactor):
            for u in range(2**compressionfactor):
                pupscr = 0
                iriscr = 0
                pupwht = 0
                iriwht = 0
                for l in range(-1 * eyerad, eyerad + 1):
                    for k in range(-1 * int(numpy.round((eyerad**2 - (numpy.abs(l) - .5)**2)**.5)), int(numpy.round((eyerad**2 - (numpy.abs(l) - .5)**2)**.5)) + 1):
                        if bscord[0] + u + k >= 0 and bscord[0] + u + k < orgimg.shape[1] and bscord[1] + v + l >= 0 and bscord[1] + v + l < orgimg.shape[0]:
                            pupscr = pupscr + orgimg[bscord[1] + v + l][bscord[0] + u + k] * numpy.exp(-1 * (k**2 + l**2) / (2 * gsslst[n][2]**2))
                            iriscr = iriscr + orgimg[bscord[1] + v + l][bscord[0] + u + k] * numpy.exp(-1 * ((k**2 + l**2)**.5 - eyerad)**2 / (2 * gsslst[n][2]**2))
                            pupwht = pupwht + numpy.exp(-1 * (k**2 + l**2) / (2 * gsslst[n][2]**2))
                            iriwht = iriwht + numpy.exp(-1 * ((k**2 + l**2)**.5 - eyerad)**2 / (2 * gsslst[n][2]**2))
                pupscr = pupscr / pupwht
                iriscr = iriscr / iriwht
                if (pupscr - iriscr) / pupscr > locscr:
                    locscr = (pupscr - iriscr) / pupscr
                    loccrd = [bscord[0] + u, bscord[1] + v]
        if locscr > .25:
            fnllst.append([loccrd[0], loccrd[1], 2**compressionfactor * gsslst[n][2]])
    return fnllst

def subsectioncomparison(image, period, wavelengths = 4, minoverlap = 0, noisy = False):
    windowsize = int(numpy.ceil(wavelengths * period))
    windowgrid = [int(numpy.ceil((image.shape[1] - windowsize) / (windowsize - minoverlap))), int(numpy.ceil((image.shape[0] - windowsize) / (windowsize - minoverlap)))]
    stepsize = [(image.shape[1] - windowsize) / windowgrid[0], (image.shape[0] - windowsize) / windowgrid[1]]
    imagefft = numpy.fft.fft2(image)
    readablefft = numpy.log10((numpy.real(imagefft)**2 + numpy.imag(imagefft)**2)**.5)
    if noisy:
        mpl.figure()
        mpl.imshow(image)
        #mpl.figure()
        #mpl.imshow(readablefft)
    for j in range(windowgrid[1] + 1):
        for i in range(windowgrid[0] + 1):
            smallslice = image[int(numpy.floor(j * stepsize[1])):(int(numpy.floor(j * stepsize[1])) + windowsize), int(numpy.floor(i * stepsize[0])):(int(numpy.floor(i * stepsize[0])) + windowsize)]
            smallfft = numpy.fft.fft2(smallslice)
            smallread = numpy.log10((numpy.real(smallfft)**2 + numpy.real(smallfft)**2)**.5)
            if noisy:
                mpl.figure()
                mpl.imshow(smallslice)
                #mpl.figure()
                #mpl.imshow(smallread)

def poolimage(orgimg, twopower = 1):
    smlimg = numpy.zeros((int(orgimg.shape[0] / 2**twopower), int(orgimg.shape[1] / 2**twopower)))
    for j in range(smlimg.shape[0]):
        for i in range(smlimg.shape[1]):
            compvl = 0
            for l in range(2**twopower):
                for k in range(2**twopower):
                    compvl = compvl + orgimg[j * 2**twopower + l][i * 2**twopower + k]
            smlimg[j][i] = compvl / (2**twopower * 2**twopower)
    return smlimg

def getsobel(orgimg, convolve = True):
    xderiv = numpy.zeros((orgimg.shape[0] - 2, orgimg.shape[1] - 2))
    yderiv = numpy.zeros((orgimg.shape[0] - 2, orgimg.shape[1] - 2))
    for j in range(1, orgimg.shape[0] - 1):
        for i in range(1, orgimg.shape[1] - 1):
            xderiv[j - 1][i - 1] = orgimg[j + 1][i + 1] + 2 * orgimg[j][i + 1] + orgimg[j - 1][i + 1] - orgimg[j + 1][i - 1] - 2 * orgimg[j][i - 1] - orgimg[j - 1][i - 1]
            yderiv[j - 1][i - 1] = orgimg[j + 1][i + 1] + 2 * orgimg[j + 1][i] + orgimg[j + 1][i - 1] - orgimg[j - 1][i + 1] - 2 * orgimg[j - 1][i] - orgimg[j - 1][i - 1]
    if convolve:
        outarr = (xderiv**2 + yderiv**2)**.5
    else:
        outarr = numpy.zeros((2, xderiv.shape[0], xderiv.shape[1]))
        outarr[0, :, :] = xderiv
        outarr[1, :, :] = yderiv
    return outarr

if __name__ == "__main__":
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    orgimg = numpy.genfromtxt(flname, delimiter = ',')
    #schrad = findcompressionradius(orgimg, noisy = True)
    orgfft = numpy.fft.fft2(orgimg)
    nrmfft = numpy.log10((numpy.real(orgfft)**2 + numpy.imag(orgfft)**2)**.5)
    revfft = numpy.zeros(nrmfft.shape)
    revfft[0:int(revfft.shape[0] / 2), 0:int(revfft.shape[1] / 2)] = nrmfft[int(nrmfft.shape[0] / 2):nrmfft.shape[0], int(nrmfft.shape[1] / 2):nrmfft.shape[1]]
    revfft[int(revfft.shape[0] / 2):revfft.shape[0], 0:int(revfft.shape[1] / 2)] = nrmfft[0:int(nrmfft.shape[0] / 2), int(nrmfft.shape[1] / 2):nrmfft.shape[1]]
    revfft[0:int(revfft.shape[0] / 2), int(revfft.shape[1] / 2):revfft.shape[1]] = nrmfft[int(nrmfft.shape[0] / 2):nrmfft.shape[0], 0:int(nrmfft.shape[1] / 2)]
    revfft[int(revfft.shape[0] / 2):revfft.shape[0], int(revfft.shape[1] / 2):revfft.shape[1]] = nrmfft[0:int(nrmfft.shape[0] / 2), 0:int(nrmfft.shape[1] / 2)]
    print(findcharacteristicperiod(orgimg, noisereductionfactor = 2, mode = 'average', feedback = True))
    mpl.figure()
    mpl.imshow(orgimg)
    mpl.figure()
    mpl.imshow(revfft)
    #subsectioncomparison(orgimg, 60, noisy = True)
    # The way I see it, there are two methods: for each local maxima, find the 
    # decay function and determine if I(x) > I(x+1) for x up to schrad and compare 
    # scores for each point, or for each maxima, see if there is another, higher 
    # maxima less than schrad apart. Neither will be perfect. Can I combine them?
    # The second one first (easier to implement)
# =============================================================================
#     pntimg = numpy.zeros(orgimg.shape)
#     for j in range(orgimg.shape[0]):
#         for i in range(orgimg.shape[1]):
#             isbigg = True
#             if j > 0:
#                 if orgimg[j - 1][i] > orgimg[j][i]:
#                     isbigg = False
#             if i > 0:
#                 if orgimg[j][i - 1] > orgimg[j][i]:
#                     isbigg = False
#             if j < orgimg.shape[0] - 1:
#                 if orgimg[j + 1][i] > orgimg[j][i]:
#                     isbigg = False
#             if i < orgimg.shape[1] - 1:
#                 if orgimg[j][i + 1] > orgimg[j][i]:
#                     isbigg = False
#             if isbigg:
#                 for l in range(int(numpy.max([0, j - schrad])), int(numpy.min([orgimg.shape[0], j + schrad + 1]))):
#                     for k in range(int(numpy.max([0, i - numpy.ceil((schrad**2 - (j - l)**2)**.5)])), int(numpy.min([orgimg.shape[1], i + numpy.ceil((schrad**2 - (j - l)**2)**.5) + 1]))):
#                         if orgimg[j][i] < orgimg[l][k]:
#                             isbigg = False
#             if isbigg:
#                 pntimg[j][i] = 1
#     vizimg = numpy.zeros((orgimg.shape[0], orgimg.shape[1], 3))
#     vizimg[:, :, 0] = (orgimg - numpy.min(orgimg)) / (numpy.max(orgimg) - numpy.min(orgimg))
#     vizimg[:, :, 1] = pntimg
# =============================================================================
# =============================================================================
#     testimg = numpy.ones((512, 512))
#     for j in range(testimg.shape[0]):
#         for i in range(testimg.shape[1]):
#             testimg[j][i] = numpy.exp(-1 * ((i - 40 * numpy.round(i / 40))**2 + (j - 50 * numpy.round(j / 50))**2) / (2 * 10**2)) + 15
#     #vizimg = fftfilter(testimg, noisy = True) #bounds = [[0, 2.5, 0, 180], [29.5, 34.5, 25.5, 30.5], [27.5, 34.5, 85.5, 90.5], [27.5, 34.5, 145.5, 150.5]], [15.5, 20.5, -5, 5], [15.5, 20.5, 55, 65], [15.5, 20.5, 115, 125]
#     vizimg = numpy.fft.ifft2(numpy.fft.fft2(testimg))
#     mpl.figure()
#     mpl.imshow(testimg)
#     mpl.figure()
#     mpl.imshow((numpy.real(vizimg)**2 + numpy.imag(vizimg)**2)**.5)
# =============================================================================