# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:35:21 2020

@author: shini
"""

import numpy
import fftcompression
import rsicompression
from matplotlib import pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

def expectmaxlocalize(image, points, radius):
    # First set up the parameters for the points
    gaussparam = numpy.zeros((len(points), 4))
    baseline = 0
    baseweight = 0
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            baseline = baseline + ((numpy.max(image) - image[j][i]) / (numpy.max(image) - numpy.min(image))) * image[j][i]
            baseweight = baseweight + ((numpy.max(image) - image[j][i]) / (numpy.max(image) - numpy.min(image)))
    baseline = baseline / baseweight
    for i in range(len(points)):
        initpos = [int(numpy.round(points[i][0])), int(numpy.round(points[i][1]))]
        hguess = image[initpos[1]][initpos[0]]
        if initpos[0] > 0:
            hguess = hguess + image[initpos[1]][initpos[0] - 1]
        if initpos[0] < image.shape[1] - 1:
            hguess = hguess + image[initpos[1]][initpos[0] + 1]
        if initpos[1] > 0:
            hguess = hguess + image[initpos[1] - 1][initpos[0]]
        if initpos[1] < image.shape[0] - 1:
            hguess = hguess + image[initpos[1] + 1][initpos[0]]
        if initpos[0] > 0 and initpos[0] < image.shape[1] - 1:
            if initpos[1] > 0 and initpos[1] < image.shape[0] - 1:
                hguess = hguess / 5
            else:
                hguess = hguess / 4
        else:
            if initpos[1] > 0 and initpos[1] < image.shape[0] - 1:
                hguess = hguess / 4
            else:
                hguess = hguess / 3
        gaussparam[i][0] = hguess
        gaussparam[i][1] = points[i][0]
        gaussparam[i][2] = points[i][1]
        gaussparam[i][3] = baseline
    noisig = 0
    crscor = image[0:image.shape[0] - 1, :] - image[1:image.shape[0], :]
    noisvg = 0
    for j in range(crscor.shape[0]):
        for i in range(crscor.shape[1]):
            noisvg = noisvg + numpy.abs(crscor[j][i])
    noisvg = noisvg / ((image.shape[0] - 1) * image.shape[1])
    for j in range(crscor.shape[0]):
        for i in range(crscor.shape[1]):
            noisig = noisig + (crscor[j][i] - noisvg)**2
    noisig = noisig**.5 / ((image.shape[0] - 1) * image.shape[1])
    atomsig = radius / 3
    magprobs = numpy.ones((len(points))) / len(points)
    belongprobs = numpy.ones((image.shape[0], image.shape[1], len(points)))
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            for k in range(len(points)):
                fauxcenter = gaussparam[k][0] * numpy.exp(-1 * ((i - gaussparam[k][1])**2 + (j - gaussparam[k][2])**2) / (2 * atomsig**2)) + gaussparam[k][3]
                belongprobs[j][i][k] = magprobs[k] * numpy.exp(-1 * (image[j][i] - fauxcenter)**2 / (2 * noisig**2)) / (noisig * (2 * numpy.pi)**.5)
    likely = 0
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            likely = likely + numpy.sum(belongprobs[j][i])
            if numpy.sum(belongprobs[j][i]) != 0:
                belongprobs[j][i] = belongprobs[j][i] / numpy.sum(belongprobs[j][i])
            else:
                belongprobs[j][i] = numpy.ones((len(points))) / len(points)
    iterparam = 0
    while iterparam < 1:
        newparams = numpy.zeros(gaussparam.shape)
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                for k in range(gaussparam.shape[0]):
                    rvectr = [gaussparam[k][1] - i, gaussparam[k][2] - j]
                    radius = numpy.log((image[j][i] - gaussparam[k][3]) / gaussparam[k][0])
                    newparams[k][0] = newparams[k][0] + belongprobs[j][i][k] * (image[j][i] - gaussparam[k][3]) / numpy.exp(-1 * ((i - gaussparam[k][1])**2 + (j - gaussparam[k][2])**2) / (2 * atomsig**2))
                    

def generatesobel(image):
    # Let's start with a Sobel operator calculation to see if we can do 
    # anything with that
    # Either way, I'm constructing a box and trying to maximize the size and 
    # average value inside the box while discounting size beyond what is 
    # necessary
    sobelimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            dx = 0
            dy = 0
            if j == 0:
                if i == 0:
                    dx = (image[j + 1][i + 1] + 2 * image[j][i + 1] - 3 * image[j][i]) / 4
                    dy = (2 * image[j + 1][i] + image[j + 1][i + 1] - 3 * image[j][i]) / 4
                elif i == image.shape[0] - 1:
                    dx = (3 * image[j][i] - 2 * image[j][i - 1] - image[j + 1][i - 1]) / 4
                    dy = (image[j + 1][i - 1] + 2 * image[j + 1][i] - 3 * image[j][i]) / 4
                else:
                    dx = (image[j + 1][i + 1] + 2 * image[j][i + 1] - 2 * image[j][i - 1] - image[j + 1][i - 1]) / 4
                    dy = (image[j + 1][i - 1] + 2 * image[j + 1][i] + image[j + 1][i + 1] - 4 * image[j][i]) / 4
            elif j == image.shape[1] - 1:
                if i == 0:
                    dx = (2 * image[j][i + 1] + image[j - 1][i + 1] - 3 * image[j][i]) / 4
                    dy = (3 * image[j][i] - image[j - 1][i + 1] - 2 * image[j - 1][i]) / 4
                elif i == image.shape[0] - 1:
                    dx = (3 * image[j][i] - image[j - 1][i - 1] - 2 * image[j][i - 1]) / 4
                    dy = (3 * image[j][i] - 2 * image[j - 1][i] - image[j - 1][i - 1]) / 4
                else:
                    dx = (image[j - 1][i + 1] + 2 * image[j][i + 1] - 2 * image[j][i - 1] - image[j - 1][i - 1]) / 4
                    dy = (4 * image[j][i] - image[j - 1][i + 1] - 2 * image[j - 1][i] - image[j - 1][i - 1]) / 4
            else:
                if i == 0:
                    dx = (image[j + 1][i + 1] + 2 * image[j][i + 1] + image[j - 1][i + 1] - 4 * image[j][i]) / 4
                    dy = (2 * image[j + 1][i] + image[j + 1][i + 1] - image[j - 1][i + 1] - 2 * image[j - 1][i]) / 4
                elif i == image.shape[0] - 1:
                    dx = (4 * image[j][i] - image[j - 1][i - 1] - 2 * image[j][i - 1] - image[j + 1][i - 1]) / 4
                    dy = (image[j + 1][i - 1] + 2 * image[j + 1][i] - 2 * image[j - 1][i] - image[j - 1][i - 1]) / 4
                else:
                    dx = (image[j + 1][i + 1] + 2 * image[j][i + 1] + image[j - 1][i + 1] - image[j - 1][i - 1] - 2 * image[j][i - 1] - image[j + 1][i - 1]) / 4
                    dy = (image[j + 1][i - 1] + 2 * image[j + 1][i] + image[j + 1][i + 1] - image[j - 1][i + 1] - 2 * image[j - 1][i] - image[j - 1][i - 1]) / 4
            totalg = (dx**2 + dy**2)**.5
            sobelimg[j][i] = totalg
    return sobelimg

def sizeparticle(image, guess):
    # guess is a diameter/radius (whichever is more convenient) of the initial 
    # size to test whether there is a particle there.
    probimg = numpy.zeros(image.shape)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            ystart = int(numpy.max([0, j - numpy.floor(guess / 2)]))
            yend = int(numpy.min([image.shape[0], j + numpy.floor(guess / 2) + 1]))
            imgmax = image[j][i]
            imgmin = image[j][i]
            totcor = 0
            for l in range(ystart, yend):
                xstart = int(numpy.max([0, i - numpy.floor((numpy.floor(guess / 2)**2 - (j - l)**2)**.5)]))
                xend = int(numpy.min([image.shape[1], i + numpy.floor((numpy.floor(guess / 2)**2 - (j - l)**2)**.5) + 1]))
                for k in range(xstart, xend):
                    if image[l][k] > imgmax:
                        imgmax = image[l][k]
                    if image[l][k] < imgmin:
                        imgmin = image[l][k]
            for l in range(ystart, yend):
                xstart = int(numpy.max([0, i - numpy.floor((numpy.floor(guess / 2)**2 - (j - l)**2)**.5)]))
                xend = int(numpy.min([image.shape[1], i + numpy.floor((numpy.floor(guess / 2)**2 - (j - l)**2)**.5) + 1]))
                for k in range(xstart, xend):
                    totcor = totcor + (image[l][k] - imgmin) * ((guess / 2)**2 - (j - l)**2 - (i - k)**2)**.5
            probimg[j][i] = totcor / ((guess / 2) * (imgmax - imgmin))
    maxlst = []
    for j in range(probimg.shape[0]):
        for i in range(probimg.shape[1]):
            search = False
            if j > 0:
                if probimg[j][i] > probimg[j - 1][i]:
                    search = True
            if i > 0:
                if probimg[j][i] > probimg[j][i - 1]:
                    search = True
            if j < probimg.shape[0] - 1:
                if probimg[j][i] > probimg[j + 1][i]:
                    search = True
            if i < probimg.shape[1] - 1:
                if probimg[j][i] > probimg[j][i + 1]:
                    search = True
            if search:
                srchng = True
                nxtlst = []
                if j > 0:
                    nxtlst.append([i, j - 1])
                if i > 0:
                    nxtlst.append([i - 1, j])
                if j < probimg.shape[0] - 1:
                    nxtlst.append([i, j + 1])
                if i < probimg.shape[1] - 1:
                    nxtlst.append([i + 1, j])
                while len(nxtlst) > 0:
                    curpnt = nxtlst.pop(0)
                    if probimg[curpnt[1]][curpnt[0]] > probimg[j][i]:
                        srchng = False
                    if srchng:
                        if curpnt[0] <= i and curpnt[1] > j:
                            if curpnt[0] > 0 and ((curpnt[0] - 1 - i)**2 + (curpnt[1] - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0] - 1, curpnt[1]])
                            if curpnt[0] == i and curpnt[1] < probimg.shape[0] - 1 and ((curpnt[0] - i)**2 + (curpnt[1] + 1 - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0], curpnt[1] + 1])
                        if curpnt[0] < i and curpnt[1] <= j:
                            if curpnt[1] > 0 and ((curpnt[0] - i)**2 + (curpnt[1] - 1 - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0], curpnt[1] - 1])
                            if curpnt[1] == j and curpnt[0] > 0 and ((curpnt[0] - 1 - i)**2 + (curpnt[1] - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0] - 1, curpnt[1]])
                        if curpnt[0] >= i and curpnt[1] < j:
                            if curpnt[0] < probimg.shape[1] - 1 and ((curpnt[0] + 1 - i)**2 + (curpnt[1] - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0] + 1, curpnt[1]])
                            if curpnt[0] == i and curpnt[1] > 0 and ((curpnt[0] - i)**2 + (curpnt[1] - 1 - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0], curpnt[1] - 1])
                        if curpnt[0] > i and curpnt[1] >= j:
                            if curpnt[1] < probimg.shape[0] - 1 and ((curpnt[0] - i)**2 + (curpnt[1] + 1 - j)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0], curpnt[1] + 1])
                            if curpnt[1] == j and curpnt[0] < probimg.shape[0] - 1 and ((curpnt[0] + 1 - i)**2 + (curpnt[1] + 1 - i)**2)**.5 <= guess / 2:
                                nxtlst.append([curpnt[0] + 1, curpnt[1]])
                if srchng:
                    maxlst.append([i, j, image[j][i]])
    srtlst = []
    for pnt in maxlst:
        pvtpnt = len(srtlst) - 1
        pvtrng = len(srtlst) / 2
        while pvtrng >= 1:
            if srtlst[pvtpnt - int(numpy.floor(pvtrng))][2] <= pnt[2]:
                pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                pvtrng = numpy.ceil(pvtrng) / 2
            else:
                pvtrng = numpy.floor(pvtrng) / 2
        if len(srtlst) == 0:
            srtlst.append(pnt)
        elif pvtpnt == len(srtlst) - 1:
            if srtlst[pvtpnt][2] >= pnt[2]:
                srtlst.append(pnt)
            else:
                srtlst.insert(pvtpnt, pnt)
        else:
            srtlst.insert(pvtpnt, pnt)
    print(srtlst)
    pntimg = numpy.zeros(probimg.shape)
    for pnt in maxlst:
        pntimg[pnt[1]][pnt[0]] = 1
    return pntimg

def outlinefind(image, radius):
    borderimage = numpy.zeros((image.shape[0], image.shape[1], 3))
    colorwheel = numpy.array([[.25, .75, 0], [.4375, .5625, 0], [.625, .375, 0], [.8125, .1875, 0], [1, 0, 0], [.8125, 0, .1872], [.625, 0, .375], [.4375, 0, .5625], [.25, 0, .75], [.0625, 0, .9375], [0, .125, .875], [0, .3125, .6875], [0, .5, .5], [0, .6875, .3125], [0, .875, .125], [.0625, .9375, 0]])
    categimage = numpy.zeros((image.shape[0], image.shape[1], 2))
    magmax = 0
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            xstart = numpy.max([i - radius, 0])
            xfinis = numpy.min([i + radius + 2, image.shape[1]])
            ystart = numpy.max([j - radius, 0])
            yfinis = numpy.min([j + radius + 2, image.shape[0]])
            orient = numpy.zeros((8, 4))
            for l in range(ystart, yfinis):
                for k in range(xstart, xfinis):
                    deltax = k - i
                    deltay = l - j
                    if deltax**2 + deltay**2 <= radius:
                        if deltax < 0:
                            orient[0][0] = orient[0][0] + image[l][k]
                            orient[0][1] = orient[0][1] + 1
                        elif deltax > 0:
                            orient[0][2] = orient[0][2] + image[l][k]
                            orient[0][3] = orient[0][3] + 1
                        if deltay > numpy.tan(67.5) * deltax:
                            orient[1][0] = orient[1][0] + image[l][k]
                            orient[1][1] = orient[1][1] + 1
                        elif deltay < numpy.tan(67.5) * deltax:
                            orient[1][2] = orient[1][2] + image[l][k]
                            orient[1][3] = orient[1][3] + 1
                        if deltay > deltax:
                            orient[2][0] = orient[2][0] + image[l][k]
                            orient[2][1] = orient[2][1] + 1
                        elif deltay < deltax:
                            orient[2][2] = orient[2][2] + image[l][k]
                            orient[2][3] = orient[2][3] + 1
                        if deltay > numpy.tan(22.5) * deltax:
                            orient[3][0] = orient[3][0] + image[l][k]
                            orient[3][1] = orient[3][1] + 1
                        elif deltay < numpy.tan(22.5) * deltax:
                            orient[3][2] = orient[3][2] + image[l][k]
                            orient[3][3] = orient[3][3] + 1
                        if deltay > 0:
                            orient[4][0] = orient[4][0] + image[l][k]
                            orient[4][1] = orient[4][1] + 1
                        elif deltay < 0:
                            orient[4][2] = orient[4][2] + image[l][k]
                            orient[4][3] = orient[4][3] + 1
                        if deltay > -1 * numpy.tan(22.5) * deltax:
                            orient[5][0] = orient[5][0] + image[l][k]
                            orient[5][1] = orient[5][1] + 1
                        elif deltay < -1 * numpy.tan(22.5) * deltax:
                            orient[5][2] = orient[5][2] + image[l][k]
                            orient[5][3] = orient[5][3] + 1
                        if deltay > -1 * deltax:
                            orient[6][0] = orient[6][0] + image[l][k]
                            orient[6][1] = orient[6][1] + 1
                        elif deltay < -1 * deltax:
                            orient[6][2] = orient[6][2] + image[l][k]
                            orient[6][3] = orient[6][3] + 1
                        if deltay > -1 * numpy.tan(67.5) * deltax:
                            orient[7][0] = orient[7][0] + image[l][k]
                            orient[7][1] = orient[7][1] + 1
                        elif deltay < -1 * numpy.tan(67.5) * deltax:
                            orient[7][2] = orient[7][2] + image[l][k]
                            orient[7][3] = orient[7][3] + 1
            direct = 0
            magnit = 0
            for m in range(orient.shape[0]):
                if orient[m][1] > 0 and orient[m][3] > 0:
                    orient[m][0] = orient[m][0] / orient[m][1]
                    orient[m][2] = orient[m][2] / orient[m][3]
                    if numpy.abs(orient[m][0] - orient[m][2]) > magnit:
                        direct = m
                        magnit = numpy.abs(orient[m][0] - orient[m][2])
            if orient[direct][0] < orient[direct][2]:
                categimage[j][i][0] = direct
            else:
                categimage[j][i][0] = direct + 8
            categimage[j][i][1] = magnit
            if magnit > magmax:
                magmax = magnit
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            isamax = True
            if i > 0:
                if categimage[j][i - 1][1] > categimage[j][i][1]:
                    isamax = False
                if j > 0:
                    if categimage[j - 1][i - 1][1] > categimage[j][i][1]:
                        isamax = False
                if j < image.shape[0] - 1:
                    if categimage[j + 1][i - 1][1] > categimage[j][i][1]:
                        isamax = False
            if i < image.shape[1] - 1:
                if categimage[j][i + 1][1] > categimage[j][i][1]:
                    isamax = False
                if j > 0:
                    if categimage[j - 1][i + 1][1] > categimage[j][i][1]:
                        isamax = False
                if j < image.shape[0] - 1:
                    if categimage[j + 1][i + 1][1] > categimage[j][i][1]:
                        isamax = False
            if j > 0:
                if categimage[j - 1][i][1] > categimage[j][i][1]:
                    isamax = False
            if j < image.shape[1] - 1:
                if categimage[j + 1][i][1] > categimage[j][i][1]:
                    isamax = False
            if isamax:
                print(categimage[j][i])
                borderimage[j][i] = colorwheel[int(categimage[j][i][0])]
    #borderimage = borderimage / magmax
    return borderimage

# =============================================================================
# def curvetrace(self, image, borderlist):
#     curvelist = []
#     for i in range(borderlist.shape[0]):
#         mostprob = None
#         highprob = 0
#         for j in range(borderlist.shape[0]):
#             if i != j:
# =============================================================================

def axisproject(pointvectors, viewaxis):
    # The projection of a vector into a plane is the dot product of that 
    # vector with the two other vectors that define the plane. A plane that 
    # is perpendicular to vector [a, b, c] is defined by [c, 0, -a] and 
    # [-ab, a^2 + c^2, -bc], unless that vector is [0, 1, 0], in which case 
    # it uses the vectors [1, 0, 0] and [0, 0, -1]
    if viewaxis[0] != 0 or viewaxis[2] != 0:
        xtrans = numpy.array([viewaxis[2], 0, -1 * viewaxis[0]]) / (viewaxis[0]**2 + viewaxis[2]**2)**.5
        ytrans = numpy.array([-1 * viewaxis[0] * viewaxis[1], viewaxis[0]**2 + viewaxis[2]**2, -1 * viewaxis[1] * viewaxis[2]]) / ((viewaxis[0]**2 + viewaxis[2]**2) * (viewaxis[0]**2 + viewaxis[1]**2 + viewaxis[2]**2))**.5
        #ztrans = numpy.arrays([viewaxis[0], viewaxis[1], viewaxis[2]]) / (viewaxis[0]**2 + viewaxis[1]**2 + viewaxis[2]**2)**.5
    else:
        xtrans = numpy.array([1, 0, 0])
        ytrans = numpy.array([0, 0, -1])
        #ztrans = numpy.array([0, 1, 0])
    projections = []
    for point in pointvectors:
        projections.append([point[0] * xtrans[0] + point[1] * xtrans[1] + point[2] * xtrans[2], point[0] * ytrans[0] + point[1] * ytrans[1] + point[2] * ytrans[2]])
    return numpy.array(projections)

def determinealign(pointvectors, refbasis):
    # What I want to do is compare the projections of a lattice to the 
    # experimental STEM images. Since there are technically an infinite number 
    # of projections, this can be paired down by only looking at the axes that 
    # have at least two atoms aligned along it. First, I need to find those 
    # axes and avoid duplicates
    zonevectors = []
    for i in range(pointvectors.shape[0] - 1):
        for j in range(i + 1, pointvectors.shape[0]):
            zone = pointvectors[j] - pointvectors[i]
            zone = zone / (zone[0]**2 + zone[1]**2 + zone[2]**2)**.5
            pvtpnt = len(zonevectors) - 1
            pvtrng = len(zonevectors) / 2
            while pvtrng >= 1:
                if zonevectors[pvtpnt - int(numpy.floor(pvtrng))][0] < zone[0] and numpy.abs(zonevectors[pvtpnt - int(numpy.floor(pvtrng))][0] - zone[0]) > 10**-8:
                    pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                    pvtrng = numpy.ceil(pvtrng) / 2
                elif zonevectors[pvtpnt - int(numpy.floor(pvtrng))][0] == zone[0] or numpy.abs(zonevectors[pvtpnt - int(numpy.floor(pvtrng))][0] - zone[0]) <= 10**-8:
                    if zonevectors[pvtpnt - int(numpy.floor(pvtrng))][1] < zone[1] and numpy.abs(zonevectors[pvtpnt - int(numpy.floor(pvtrng))][1] - zone[1]) > 10**-8:
                        pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                        pvtrng = numpy.ceil(pvtrng) / 2
                    elif zonevectors[pvtpnt - int(numpy.floor(pvtrng))][1] == zone[1] or numpy.abs(zonevectors[pvtpnt - int(numpy.floor(pvtrng))][1] - zone[1]) <= 10**-8:
                        if (zonevectors[pvtpnt - int(numpy.floor(pvtrng))][2] - zone[2]) <= 10**-8:
                            pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                            pvtrng = numpy.ceil(pvtrng) / 2
                        else:
                            pvtrng = numpy.floor(pvtrng) / 2
                    else:
                        pvtrng = numpy.floor(pvtrng) / 2
                else:
                    pvtrng = numpy.floor(pvtrng) / 2
            if len(zonevectors) == 0:
                zonevectors.append(zone)
            elif pvtpnt == len(zonevectors) - 1:
                if zonevectors[pvtpnt][0] > zone[0] and numpy.abs(zonevectors[pvtpnt][0] - zone[0]) > 10**-8:
                    zonevectors.append(zone)
                elif zonevectors[pvtpnt][0] == zone[0] or numpy.abs(zonevectors[pvtpnt][0] - zone[0]) <= 10**-8:
                    if zonevectors[pvtpnt][1] > zone[1] and numpy.abs(zonevectors[pvtpnt][1] - zone[1]) > 10**-8:
                        zonevectors.append(zone)
                    elif zonevectors[pvtpnt][1] == zone[1] or numpy.abs(zonevectors[pvtpnt][1] - zone[1]) <= 10**-8:
                        if zonevectors[pvtpnt][2] > zone[2] and numpy.abs(zonevectors[pvtpnt][2] - zone[2]) > 10**-8:
                            zonevectors.append(zone)
                        elif zonevectors[pvtpnt][2] < zone[2] and numpy.abs(zonevectors[pvtpnt][2] - zone[2]) > 10**-8:
                            zonevectors.insert(pvtpnt, zone)
                    else:
                        zonevectors.insert(pvtpnt, zone)
                else:
                    zonevectors.insert(pvtpnt, zone)
            else:
                if zonevectors[pvtpnt][0] == zone[0] or numpy.abs(zonevectors[pvtpnt][0] - zone[0]) <= 10**-8:
                    if zonevectors[pvtpnt][1] == zone[1] or numpy.abs(zonevectors[pvtpnt][1] - zone[1]) <= 10**-8:
                        if zonevectors[pvtpnt][2] < zone[2] and numpy.abs(zonevectors[pvtpnt][2] - zone[2]) > 10**-8:
                            zonevectors.insert(pvtpnt, zone)
                    else:
                        zonevectors.insert(pvtpnt, zone)
                else:
                    zonevectors.insert(pvtpnt, zone)
    # Next, generate all the projections for comparison. It would also be 
    # beneficial for these to contain no duplicates, so those will be weeded 
    # out as well
    projections = []
    for zone in zonevectors:
        fullproject = axisproject(pointvectors, zone)
        cndsdprj = []
        for point in fullproject:
            pvtpnt = len(cndsdprj) - 1
            pvtrng = len(cndsdprj) / 2
            while pvtrng >= 1:
                if cndsdprj[pvtpnt - int(numpy.floor(pvtrng))][0] < point[0] and numpy.abs(cndsdprj[pvtpnt - int(numpy.floor(pvtrng))][0] - point[0]) > 10**-8:
                    pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                    pvtrng = numpy.ceil(pvtrng) / 2
                elif cndsdprj[pvtpnt - int(numpy.floor(pvtrng))][0] == point[0] or numpy.abs(cndsdprj[pvtpnt - int(numpy.floor(pvtrng))][0] - point[0]) <= 10**-8:
                    if (cndsdprj[pvtpnt - int(numpy.floor(pvtrng))][1] - point[1]) <= 10**-8:
                        pvtpnt = pvtpnt - int(numpy.floor(pvtrng))
                        pvtrng = numpy.ceil(pvtrng) / 2
                    else:
                        pvtrng = numpy.floor(pvtrng) / 2
                else:
                    pvtrng = numpy.floor(pvtrng) / 2
            if len(cndsdprj) == 0:
                cndsdprj.append(point)
            elif pvtpnt == len(cndsdprj) - 1:
                if cndsdprj[pvtpnt][0] > point[0] and numpy.abs(cndsdprj[pvtpnt][0] - point[0]) > 10**-8:
                    cndsdprj.append(point)
                elif cndsdprj[pvtpnt][0] == point[0] or numpy.abs(cndsdprj[pvtpnt][0] - point[0]) <= 10**-8:
                    if cndsdprj[pvtpnt][1] > point[1] and numpy.abs(cndsdprj[pvtpnt][1] - point[1]) > 10**-8:
                        cndsdprj.append(point)
                    elif cndsdprj[pvtpnt][1] < point[1] and numpy.abs(cndsdprj[pvtpnt][1] - point[1]) > 10**-8:
                        cndsdprj.insert(pvtpnt, point)
                else:
                    cndsdprj.insert(pvtpnt, point)
            else:
                if cndsdprj[pvtpnt][0] == point[0] or numpy.abs(cndsdprj[pvtpnt][0] - point[0]) <= 10**-8:
                    if cndsdprj[pvtpnt][1] < point[1] and numpy.abs(cndsdprj[pvtpnt][1] - point[1]) > 10**-8:
                        cndsdprj.insert(pvtpnt, point)
                else:
                    cndsdprj.insert(pvtpnt, point)
        projections.append([zone, numpy.array(cndsdprj)])
    # Finally, compare the different projections for how well they compare to 
    # to a reference pattern. This will consist of finding a central point -  
    # or several - finding the rotation that causes the least error, 
    # then calculating some error function (distance from lattice points) and 
    # ranking them. I'm not trying to optimize anything, so I can just sweep 
    # through 180 degrees in 1 degree increments
    axisscores = []
    print(len(projections))
    count = 0
    for project in projections:
        print(100 * count / len(projections))
        count = count + 1
        axiserror = -1
        #optang = None
        for rot in range(180):
            compvecs = numpy.zeros(refbasis.shape)
            for i in range(compvecs.shape[0]):
                compvecs[i][0] = refbasis[i][0] * numpy.cos(rot * numpy.pi / 180) - refbasis[i][1] * numpy.sin(rot * numpy.pi / 180)
                compvecs[i][1] = refbasis[i][0] * numpy.sin(rot * numpy.pi / 180) + refbasis[i][1] * numpy.cos(rot * numpy.pi / 180)
            totalerror = 0
            for j in range(project[1].shape[0]):
                for vec in compvecs:
                    vecerr = ((project[1][0][0] - project[1][j][0]) - vec[0])**2 + ((project[1][0][1] - project[1][j][1]) - vec[1])**2
                    for i in range(1, project[1].shape[0]):
                        if ((project[1][i][0] - project[1][j][0]) - vec[0])**2 + ((project[1][i][1] - project[1][j][1]) - vec[1])**2 < vecerr:
                            vecerr = ((project[1][i][0] - project[1][j][0]) - vec[0])**2 + ((project[1][i][1] - project[1][j][1]) - vec[1])**2
                    totalerror = totalerror + vecerr
            if axiserror < 0 or totalerror < axiserror:
                axiserror = totalerror
                #optang = compvecs
        axisscores.append([project[0][0], project[0][1], project[0][2], axiserror / project[1].shape[0]])
        #mpl.figure()
        #mpl.scatter(project[1][:, 0], project[1][:, 1])
        #mpl.scatter(optang[:, 0], optang[:, 1])
    return axisscores

def findslices(basis, avec, bvec, cvec, zonevect, area, relative = True, minslice = 10**-6):
    # This will take in a basis and a set of vectors to define a unit cell 
    # and will extrapolate that structure such that it fills a volume as tall 
    # as the new projection and as wide as some defined area, centered on the 
    # original origin. It will return the structure in terms of slices, where 
    # a slice is all atoms between some z-range (lower bound inclusive) 
    # bounded by the z positions of the atoms of the projected basis.
    # First, convert from relative to (x, y, z) if necessary
    corrbasis = []
    if relative:
        if len(basis.shape) == 1:
            corrbasis.append(numpy.array([avec[0] * basis[0] + bvec[0] * basis[1] + cvec[0] * basis[2], avec[1] * basis[0] + bvec[1] * basis[1] + cvec[1] * basis[2], avec[2] * basis[0] + bvec[2] * basis[1] + cvec[2] * basis[2]]))
        else:
            for point in basis:
                corrbasis.append([avec[0] * point[0] + bvec[0] * point[1] + cvec[0] * point[2], avec[1] * point[0] + bvec[1] * point[1] + cvec[1] * point[2], avec[2] * point[0] + bvec[2] * point[1] + cvec[2] * point[2]])
        viewaxis = numpy.array([avec[0] * zonevect[0] + bvec[0] * zonevect[1] + cvec[0] * zonevect[2], avec[1] * zonevect[0] + bvec[1] * zonevect[1] + cvec[1] * zonevect[2], avec[2] * zonevect[0] + bvec[2] * zonevect[1] + cvec[2] * zonevect[2]])
    else:
        if len(basis.shape) == 1:
            corrbasis.append(numpy.array([basis[0], basis[1], basis[2]]))
        else:
            for point in basis:
                corrbasis.append([point[0], point[1], point[2]])
        viewaxis = zonevect
    # Create the projection, then transform the original basis and the lattice 
    # vectors
    if viewaxis[0] != 0 or viewaxis[2] != 0:
        xtrans = numpy.array([viewaxis[2], 0, -1 * viewaxis[0]]) / (viewaxis[0]**2 + viewaxis[2]**2)**.5
        ytrans = numpy.array([-1 * viewaxis[0] * viewaxis[1], viewaxis[0]**2 + viewaxis[2]**2, -1 * viewaxis[1] * viewaxis[2]]) / ((viewaxis[0]**2 + viewaxis[2]**2) * (viewaxis[0]**2 + viewaxis[1]**2 + viewaxis[2]**2))**.5
        ztrans = numpy.array([viewaxis[0], viewaxis[1], viewaxis[2]]) / (viewaxis[0]**2 + viewaxis[1]**2 + viewaxis[2]**2)**.5
    else:
        xtrans = numpy.array([1, 0, 0])
        ytrans = numpy.array([0, 0, -1])
        ztrans = numpy.array([0, 1, 0])
    projections = []
    for point in corrbasis:
        projections.append([point[0] * xtrans[0] + point[1] * xtrans[1] + point[2] * xtrans[2], point[0] * ytrans[0] + point[1] * ytrans[1] + point[2] * ytrans[2], point[0] * ztrans[0] + point[1] * ztrans[1] + point[2] * ztrans[2]])
    projlvecs = numpy.zeros((3, 3))
    projlvecs[0, :] = numpy.array([avec[0] * xtrans[0] + avec[1] * xtrans[1] + avec[2] * xtrans[2], avec[0] * ytrans[0] + avec[1] * ytrans[1] + avec[2] * ytrans[2], avec[0] * ztrans[0] + avec[1] * ztrans[1] + avec[2] * ztrans[2]])
    projlvecs[1, :] = numpy.array([bvec[0] * xtrans[0] + bvec[1] * xtrans[1] + bvec[2] * xtrans[2], bvec[0] * ytrans[0] + bvec[1] * ytrans[1] + bvec[2] * ytrans[2], bvec[0] * ztrans[0] + bvec[1] * ztrans[1] + bvec[2] * ztrans[2]])
    projlvecs[2, :] = numpy.array([cvec[0] * xtrans[0] + cvec[1] * xtrans[1] + cvec[2] * xtrans[2], cvec[0] * ytrans[0] + cvec[1] * ytrans[1] + cvec[2] * ytrans[2], cvec[0] * ztrans[0] + cvec[1] * ztrans[1] + cvec[2] * ztrans[2]])
    cornerlist = [projlvecs[0], projlvecs[1], projlvecs[2], projlvecs[0] + projlvecs[1], projlvecs[1] + projlvecs[2], projlvecs[2] + projlvecs[0], projlvecs[0] + projlvecs[1] + projlvecs[2]]
    orgbounds = numpy.zeros((3, 2))
    for corner in cornerlist:
        if corner[0] < orgbounds[0][0]:
            orgbounds[0][0] = corner[0]
        if corner[0] > orgbounds[0][1]:
            orgbounds[0][1] = corner[0]
        if corner[1] < orgbounds[1][0]:
            orgbounds[1][0] = corner[1]
        if corner[1] > orgbounds[1][1]:
            orgbounds[1][1] = corner[1]
        if corner[2] < orgbounds[2][0]:
            orgbounds[2][0] = corner[2]
        if corner[2] > orgbounds[2][1]:
            orgbounds[2][1] = corner[2]
    volsize = numpy.zeros((3, 2))
    # Define the dimensions of the slices
    if relative:
        if area[0] > 1:
            volsize[0][0] = ((orgbounds[0][0] + orgbounds[0][1]) / 2) - area[0] * ((orgbounds[0][1] - orgbounds[0][0]) / 2)
            volsize[0][1] = ((orgbounds[0][0] + orgbounds[0][1]) / 2) + area[0] * ((orgbounds[0][1] - orgbounds[0][0]) / 2)
        else:
            volsize[0][0] = orgbounds[0][0]
            volsize[0][1] = orgbounds[0][1]
        if area[1] > 1:
            volsize[1][0] = ((orgbounds[1][0] + orgbounds[1][1]) / 2) - area[1] * ((orgbounds[1][1] - orgbounds[1][0]) / 2)
            volsize[1][1] = ((orgbounds[1][0] + orgbounds[1][1]) / 2) + area[1] * ((orgbounds[1][1] - orgbounds[1][0]) / 2)
        else:
            volsize[1][0] = orgbounds[1][0]
            volsize[1][1] = orgbounds[1][1]
        volsize[2][0] = orgbounds[2][0]
        volsize[2][1] = orgbounds[2][1]
    else:
        if area[0] > (orgbounds[0][1] - orgbounds[0][0]):
            volsize[0][0] = ((orgbounds[0][0] + orgbounds[0][1]) / 2) - area[0] / 2
            volsize[0][1] = ((orgbounds[0][0] + orgbounds[0][1]) / 2) + area[0] / 2
        else:
            volsize[0][0] = orgbounds[0][0]
            volsize[0][1] = orgbounds[0][1]
        if area[1] > (orgbounds[1][1] - orgbounds[1][0]):
            volsize[1][0] = ((orgbounds[1][0] + orgbounds[1][1]) / 2) - area[1] / 2
            volsize[1][1] = ((orgbounds[1][0] + orgbounds[1][1]) / 2) + area[1] / 2
        else:
            volsize[1][0] = orgbounds[1][0]
            volsize[1][1] = orgbounds[1][1]
        volsize[2][0] = orgbounds[2][0]
        volsize[2][1] = orgbounds[2][1]
    planeslices = []
    for point in projections:
        if len(planeslices) == 0:
            planeslices.append([point[2], []])
        else:
            i = 0
            srchng = True
            while i < len(planeslices) and srchng:
                if point[2] - planeslices[i][0] > minslice:
                    i = i + 1
                elif point[2] - planeslices[i][0] <= minslice:
                    srchng = False
            if srchng:
                planeslices.append([point[2], []])
            else:
                if point[2] - planeslices[i][0] < -1 * minslice:
                    planeslices.insert(i, [point[2], []])
# =============================================================================
#     for corner in cornerlist:
#         i = 0
#         srchng = True
#         while i < len(planeslices) and srchng:
#             if corner[2] - planeslices[i][0] > minslice:
#                 i = i + 1
#             elif corner[2] - planeslices[i][0] <= minslice:
#                 srchng = False
#         if srchng:
#             planeslices.append([corner[2], []])
#         else:
#             if corner[2] - planeslices[i][0] < -1 * minslice:
#                 planeslices.insert(i, [corner[2], []])
# =============================================================================
    # Find the different offset coordinates needed to fill the volume
    ijklims = [[-1, 1], [-1, 1], [-1, 1]]
    for xb in range(2):
        for yb in range(2):
            for zb in range(2):
                prjb = [volsize[0][xb] * projlvecs[0][0] + volsize[1][yb] * projlvecs[0][1] + volsize[2][zb] * projlvecs[0][2], volsize[0][xb] * projlvecs[1][0] + volsize[1][yb] * projlvecs[1][1] + volsize[2][zb] * projlvecs[1][2], volsize[0][xb] * projlvecs[2][0] + volsize[1][yb] * projlvecs[2][1] + volsize[2][zb] * projlvecs[2][2]]
                if numpy.floor(prjb[0]) < ijklims[0][0]:
                    ijklims[0][0] = int(numpy.floor(prjb[0]))
                elif numpy.ceil(prjb[0]) > ijklims[0][1]:
                    ijklims[0][1] = int(numpy.ceil(prjb[0]))
                if numpy.floor(prjb[1]) < ijklims[1][0]:
                    ijklims[1][0] = int(numpy.floor(prjb[1]))
                elif numpy.ceil(prjb[1]) > ijklims[1][1]:
                    ijklims[1][1] = int(numpy.ceil(prjb[1]))
                if numpy.floor(prjb[2]) < ijklims[2][0]:
                    ijklims[2][0] = int(numpy.floor(prjb[2]))
                elif numpy.ceil(prjb[2]) > ijklims[2][1]:
                    ijklims[2][1] = int(numpy.ceil(prjb[2]))
    for i in range(ijklims[0][0], ijklims[0][1] + 1):
        for j in range(ijklims[1][0], ijklims[1][1] + 1):
            for k in range(ijklims[2][0], ijklims[2][1] + 1):
                for point in projections:
                    shftdpnt = numpy.array([point[0] + i * projlvecs[0][0] + j * projlvecs[1][0] + k * projlvecs[2][0], point[1] + i * projlvecs[0][1] + j * projlvecs[1][1] + k * projlvecs[2][1], point[2] + i * projlvecs[0][2] + j * projlvecs[1][2] + k * projlvecs[2][2]])
                    if (shftdpnt[0] - volsize[0][0] >= -1 * minslice) and (shftdpnt[0] - volsize[0][1] <= minslice):
                        if (shftdpnt[1] - volsize[1][0] >= -1 * minslice) and (shftdpnt[1] - volsize[1][1] <= minslice):
                            if (shftdpnt[2] - volsize[2][0] >= -1 * minslice) and (shftdpnt[2] - volsize[2][1] <= minslice):
                                n = 0
                                srtng = True
                                while n < len(planeslices) and srtng:
                                    if shftdpnt[2] - planeslices[n][0] <= minslice:
                                        srtng = False
                                    else:
                                        n = n + 1
                                if srtng:
                                    planeslices.append([shftdpnt[2], []])
                                    planeslices[n][1].append(shftdpnt)
                                else:
                                    if numpy.abs(shftdpnt[2] - planeslices[n][0]) <= minslice:
                                        planeslices[n][1].append(shftdpnt)
                                    else:
                                        planeslices.insert(n, [shftdpnt[2], []])
                                        planeslices[n][1].append(shftdpnt)
    for plane in planeslices:
        grphsa = numpy.zeros((len(plane[1]), 3))
        for m in range(grphsa.shape[0]):
            grphsa[m][0] = plane[1][m][0]
            grphsa[m][1] = plane[1][m][1]
            grphsa[m][2] = plane[1][m][2]
        plane[1] = grphsa
    return planeslices

def visualizeprojection(projection, scale):
    # scale is in angstroms per pixel
    xwidth = int(numpy.ceil((numpy.max(projection[:, 0]) - numpy.min(projection[:, 0])) / scale))
    ywidth = int(numpy.ceil((numpy.max(projection[:, 1]) - numpy.min(projection[:, 1])) / scale))
    xstart = numpy.min(projection[:, 0])
    ystart = numpy.min(projection[:, 1])
    retimg = numpy.zeros((ywidth + 18, xwidth + 18))
    for point in projection:
        xcentr = int(numpy.floor((point[0] - xstart) / scale))
        ycentr = int(numpy.floor((point[1] - ystart) / scale))
        xoffst = ((point[0] - xstart) / scale) - xcentr
        yoffst = ((point[1] - ystart) / scale) - ycentr
        for j in range(-3, 4):
            for i in range(-3, 4):
                retimg[ycentr + j + 9][xcentr + i + 9] = retimg[ycentr + j + 9][xcentr + i + 9] + numpy.exp(-1 * ((xoffst + i)**2 + (yoffst + j)**2) / 4.5)
    return retimg

def sparsecrosscorr(refbasis, compbasis, basvecs, increments):
    # For each increment from (0, 1 - increments^-1) in both x and y, find the 
    # lowest overall spacing for each point in the basis to compare
    scrmap = numpy.zeros((increments, increments))
    for i in range(increments):
        for j in range(increments):
            #pltbsx = []
            #pltbsy = []
            #pltrfx = []
            #pltrfy = []
            # For each point in the basis to compare, find their representation in 
            # (u, v) coordinates, then find the closest point in the reference basis 
            # within +-1 the (u, v) cell to be thorough
            totscr = 0
            for compnt in compbasis:
                offpnt = [compnt[0] + i * basvecs[0][0] / increments + j * basvecs[1][0] / increments, compnt[1] + i * basvecs[0][1] / increments + j * basvecs[1][1] / increments]
                #pltbsx.append(offpnt[0])
                #pltbsy.append(offpnt[1])
                ucoord = int(numpy.floor((offpnt[0] * basvecs[1][1] - offpnt[1] * basvecs[1][0]) / (basvecs[0][0] * basvecs[1][1] - basvecs[0][1] * basvecs[1][0])))
                vcoord = int(numpy.floor((offpnt[1] * basvecs[0][0] - offpnt[0] * basvecs[0][1]) / (basvecs[0][0] * basvecs[1][1] - basvecs[0][1] * basvecs[1][0])))
                #clscor = [0, 0]
                mindst = basvecs[0][0]**2 + basvecs[0][1]**2 + basvecs[1][0]**2 + basvecs[1][1]**2
                for k in range(ucoord - 1, ucoord + 2):
                    for l in range(vcoord - 1, vcoord + 2):
                        for refpnt in refbasis:
                            #pltrfx.append(refpnt[0] + k * basvecs[0][0] + l * basvecs[1][0])
                            #pltrfy.append(refpnt[1] + k * basvecs[0][1] + l * basvecs[1][1])
                            if ((offpnt[0] - (refpnt[0] + k * basvecs[0][0] + l * basvecs[1][0]))**2 + (offpnt[1] - (refpnt[1] + k * basvecs[0][1] + l * basvecs[1][1]))**2)**.5 < mindst:
                                mindst = ((offpnt[0] - (refpnt[0] + k * basvecs[0][0] + l * basvecs[1][0]))**2 + (offpnt[1] - (refpnt[1] + k * basvecs[0][1] + l * basvecs[1][1]))**2)**.5
                                #clscor = [refpnt[0] + k * basvecs[0][0] + l * basvecs[1][0], refpnt[1] + k * basvecs[0][1] + l * basvecs[1][1]]
                totscr = totscr + mindst
            #mpl.figure()
            #mpl.scatter(numpy.array(pltbsx), numpy.array(pltbsy))
            #mpl.scatter(numpy.array(pltrfx), numpy.array(pltrfy))
            scrmap[j][i] = totscr / len(compbasis)
    return scrmap

def allangscc(refbasis, compbasis, basvecs, increments):
    # A wrapper function for sparsecrosscorr that will run sparsecrosscorr on 
    # compbasis rotated 360 degrees in 1 degree increments. It will return the 
    # minimum score of the output map for sparsecrosscorr for each angle
    scrarr = []
    for theta in range(360):
        trnbas = []
        for pnt in compbasis:
            trnbas.append([pnt[0] * numpy.cos(theta * numpy.pi / 180) - pnt[1] * numpy.sin(theta * numpy.pi / 180), pnt[0] * numpy.sin(theta * numpy.pi / 180) + pnt[1] * numpy.cos(theta * numpy.pi / 180)])
        curmap = sparsecrosscorr(refbasis, trnbas, basvecs, increments)
        scrarr.append(numpy.min(curmap))
    return scrarr

if __name__ == '__main__':
    #flname = input('Please enter the name of your file: ')
    #[fname, ftype] = flname.split('.')
    #if ftype == 'csv':
    #    imgdat = numpy.genfromtxt(flname, delimiter = ',')
    #elif ftype == 'jpg':
    #    imgdat = imageio.imread(flname)
    #imprad = fftcompression.findcompressionradius(imgdat)
    #newdims = (int(imgdat.shape[0] / imprad), int(imgdat.shape[1] / imprad))
    #iddimg = rsicompression.resize(imgdat, newdims)
    #mpl.figure()
    #mpl.imshow(iddimg)
    #sobel = generatesobel(iddimg)
    #analimg = sizeparticle(iddimg, 25)
    #outimg = outlinefind(iddimg, 30)
# =============================================================================
#     #basis = numpy.array([[0, 0, 1.67], [1.45, .66, 0], [5.5, 1.905, 0], [4.05, 2.57, 1.67], [6.95, 1.24, 1.67], [9.55, 3.15, 0]])
#     #basis = numpy.genfromtxt(f'C:\\Users\\shini\\Documents\\Research\\Deborah\\TEM\\Processed images\\PdCl2b basis.csv', delimiter = ',')
#     #basis = numpy.array([[0, 0, 0], [1.802, 1.802, 0], [0, 1.802, 1.802], [1.802, 0, 1.802]])
#     basis = numpy.array([[0, 0, .5], [.132025, .174079754601227, 0], [.367974549310711, .674079754601227, .5], [.5, .5, 0], [.63202545068929, .325920245398773, .5], [.86797454931071, .825920245398773, 0]])
#     avect = numpy.array([11, 0, 0])
#     bvect = numpy.array([0, 3.81, 0])
#     cvect = numpy.array([0, 0, 3.34])
#     xpoints = []
#     ypoints = []
#     zpoints = []
#     particle = []
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 for base in basis:
#                     #xpoints.append(base[0] + 13.047 * i - 13.047 * k * numpy.cos(numpy.pi / 3))
#                     #ypoints.append(base[1] + 8.602 * j)
#                     #zpoints.append(base[2] + 13.047 * k * numpy.sin(numpy.pi / 3))
#                     #particle.append([base[0] + 13.047 * i - 13.047 * k * numpy.cos(numpy.pi / 3), base[1] + 8.602 * j, base[2] + 13.047 * k * numpy.sin(numpy.pi / 3)])
#                     particle.append((base[0] * avect + base[1] * bvect + base[2] * cvect) + i * avect + j * bvect + k* cvect)
#     particle = numpy.array(particle)
#     compare = numpy.array([[2.3, 0], [1.21, 1.95], [-1.14, 1.05], [-2.3, 0], [-1.21, -1.95], [1.14, -1.05]])
#     #xaxis = axisproject(particle, numpy.array([1, 0, 0]))
#     #yaxis = axisproject(particle, numpy.array([0, 1, 0]))
#     #zaxis = axisproject(particle, numpy.array([0, 0, 1]))
#     #allaxs = determinealign(particle, compare)
#     #numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Deborah\\TEM\\Processed images\\Cu 3x3x3.csv', allaxs, delimiter = ',')
#     #print(numpy.array(allaxs))
#     pjcton = axisproject(particle, numpy.array([1, 0, 0]))
#     orange = findslices(basis, avect, bvect, cvect, numpy.array([3, 3, 1]), (5, 5))
#     #print(orange)
#     fulldatlst = []
#     for i in range(len(orange)):
#         fulldatlst.append([orange[i][0], 0, 0])
#         for point in orange[i][1]:
#             datrow = []
#             for coord in point:
#                 datrow.append(coord)
#             fulldatlst.append(datrow)
#     #print(fulldatlst)
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Deborah\\TEM\\Processed images\\Lattice projections\\PdCl2a 331 5x5.csv', numpy.array(fulldatlst), delimiter = ',')
#     #mpl.figure()
#     #mpl.imshow(sobel)
#     fig = mpl.figure()
#     ax = fig.add_subplot(111, projection = '3d')
#     #ax.scatter(xpoints, ypoints, zpoints)
#     ax.scatter(particle[:, 0], particle[:, 1], particle[:, 2])
# # =============================================================================
# #     mpl.figure()
# #     mpl.scatter(xaxis[:, 0], xaxis[:, 1])
# #     #mpl.scatter(compare[:, 0], compare[:, 1])
# #     mpl.figure()
# #     mpl.scatter(yaxis[:, 0], yaxis[:, 1])
# #     #mpl.scatter(compare[:, 0], compare[:, 1])
# #     mpl.figure()
# #     mpl.scatter(zaxis[:, 0], zaxis[:, 1])
# #     #mpl.scatter(compare[:, 0], compare[:, 1])
# # =============================================================================
#     mpl.figure()
#     mpl.scatter(pjcton[:, 0], pjcton[:, 1])
#     mpl.figure()
#     mpl.imshow(visualizeprojection(pjcton, .3125))
#     mpl.figure()
#     for level in orange:
#         mpl.scatter(level[1][:, 0], level[1][:, 1])
# =============================================================================
    atopbas = [[0, 0], [.142 * numpy.cos(numpy.pi / 6), .142 * numpy.sin(numpy.pi / 6)]]
    bridbas = [[.071 * numpy.cos(numpy.pi / 6), .071 * numpy.sin(numpy.pi / 6)], [.142 * numpy.cos(numpy.pi / 6), .142], [.213 * numpy.cos(numpy.pi / 6), .071 * numpy.sin(numpy.pi / 6)]]
    bothbas = [[0, 0], [.071 * numpy.cos(numpy.pi / 6), .071 * numpy.sin(numpy.pi / 6)], [.142 * numpy.cos(numpy.pi / 6), .142 * numpy.sin(numpy.pi / 6)], [.142 * numpy.cos(numpy.pi / 6), .142], [.213 * numpy.cos(numpy.pi / 6), .071 * numpy.sin(numpy.pi / 6)]]
    #compas = [[0, 0], [.275, 0], [0, .406], [.275, .406]] #, [.55, 0], [.55, .389], [0, .778], [.275, .778], [.55, .778]
    #compas = [[0, 0], [.23 * 2 * numpy.sin(numpy.pi / 3), 0], [.23 * 2 * numpy.sin(numpy.pi / 3) * numpy.cos(numpy.pi / 3), .23 * 2 * numpy.sin(numpy.pi / 3) * numpy.sin(numpy.pi / 3)]]
    #compas = [[0, 0], [.275, 0], [0, .41], [.275, .41]]
    #compas = [[0, 0], [.275, 0], [.55, 0], [0, .41], [.275, .41], [.55, .41], [0, .82], [.275, .82], [.55, .82]]
    #compas = [[0, 0], [.275, 0], [.55, 0], [.825, 0], [0, .41], [.275, .41], [.55, .41], [.825, .41], [0, .82], [.275, .82], [.55, .82], [.825, .82], [0, 1.23], [.275, 1.23], [.55, 1.23], [.825, 1.23]]
    # Need a 7x5 grid to get a 2nm x 2nm particle
    compas = []
    for i in range(8):
        for j in range(5):
            compas.append([.39951657 * i, .691983 * j])
    for i in range(7):
        for j in range(4):
            compas.append([.199758286 + .39951657 * i, .3459915 + .691983 * j])
    #atopmap = sparsecrosscorr(atopbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    atopspc = allangscc(atopbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    print('Atop done')
    #bridmap = sparsecrosscorr(bridbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    bridspc = allangscc(bridbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    print('Bridge done')
    #bothmap = sparsecrosscorr(bothbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    bothspc = allangscc(bothbas, compas, [[.284 * numpy.cos(numpy.pi / 6), 0], [0.284 * numpy.cos(numpy.pi / 6) * numpy.cos(numpy.pi / 3), .284 * numpy.cos(numpy.pi / 6) * numpy.sin(numpy.pi / 3)]], 50)
    print('Both done')
    mpl.figure()
    mpl.plot(numpy.array(range(360)), atopspc)
    mpl.plot(numpy.array(range(360)), bridspc)
    mpl.plot(numpy.array(range(360)), bothspc)
    numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Deborah\\Figures\\Registration comparison film 7x4.csv', numpy.array([atopspc, bridspc, bothspc]), delimiter = ',')
    #mpl.imshow(atopmap)
    #mpl.figure()
    #mpl.imshow(bridmap)
    #mpl.figure()
    #mpl.imshow(bothmap)