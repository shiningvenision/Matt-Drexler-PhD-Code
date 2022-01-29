# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:50:50 2021

@author: shini
"""

import numpy
import imageio
import time
import scipy.special
import matplotlib.pyplot as mpl

def sobeloperator(image, noisy = False):
    sobelimage = numpy.zeros((image.shape[0], image.shape[1], 2))
    for j in range(sobelimage.shape[0]):
        for i in range(sobelimage.shape[1]):
            if j == 0:
                if i == 0:
                    sobelimage[j][i][0] = image[j + 1][i + 1] + 2 * image[j][i + 1] - 3 * image[j][i]
                    sobelimage[j][i][1] = image[j + 1][i + 1] + 2 * image[j + 1][i] - 3 * image[j][i]
                elif i == sobelimage.shape[1] - 1:
                    sobelimage[j][i][0] = 3 * image[j][i] - image[j + 1][i - 1] - 2 * image[j][i - 1]
                    sobelimage[j][i][1] = 2 * image[j + 1][i] + image[j + 1][i - 1] - 3 * image[j][i]
                else:
                    sobelimage[j][i][0] = image[j + 1][i + 1] + 2 * image[j][i + 1] - image[j + 1][i - 1] - 2 * image[j][i - 1]
                    sobelimage[j][i][1] = image[j + 1][i + 1] + 2 * image[j + 1][i] + image[j + 1][i - 1] - 4 * image[j][i]
            elif j == sobelimage.shape[0] - 1:
                if i == 0:
                    sobelimage[j][i][0] = 2 * image[j][i + 1] + image[j - 1][i + 1] - 3 * image[j][i]
                    sobelimage[j][i][1] = 3 * image[j][i] - image[j - 1][i + 1] - 2 * image[j - 1][i]
                elif i == sobelimage.shape[1] - 1:
                    sobelimage[j][i][0] = 3 * image[j][i] - 2 * image[j][i - 1] - image[j - 1][i - 1]
                    sobelimage[j][i][1] = 3 * image[j][i] - 2 * image[j - 1][i] - image[j - 1][i - 1]
                else:
                    sobelimage[j][i][0] = 2 * image[j][i + 1] + image[j - 1][i + 1] - 2 * image[j][i - 1] - image[j - 1][i - 1]
                    sobelimage[j][i][1] = 4 * image[j][i] - image[j - 1][i + 1] - 2 * image[j - 1][i] - image[j - 1][i - 1]
            else:
                if i == 0:
                    sobelimage[j][i][0] = image[j + 1][i + 1] + 2 * image[j][i + 1] + image[j - 1][i + 1] - 4 * image[j][i]
                    sobelimage[j][i][1] = image[j + 1][i + 1] + 2 * image[j + 1][i] - image[j - 1][i + 1] - 2 * image[j - 1][i]
                elif i == sobelimage.shape[1] - 1:
                    sobelimage[j][i][0] = 4 * image[j][i] - image[j + 1][i - 1] - 2 * image[j][i - 1] - image[j - 1][i - 1]
                    sobelimage[j][i][1] = 2 * image[j + 1][i] + image[j + 1][i - 1] - 2 * image[j - 1][i] - image[j - 1][i - 1]
                else:
                    sobelimage[j][i][0] = image[j + 1][i + 1] + 2 * image[j][i + 1] + image[j - 1][i + 1] - image[j + 1][i - 1] - 2 * image[j][i - 1] - image[j - 1][i - 1]
                    sobelimage[j][i][1] = image[j + 1][i + 1] + 2 * image[j + 1][i] + image[j + 1][i - 1] - image[j - 1][i + 1] - 2 * image[j - 1][i] - image[j - 1][i - 1]
    if noisy:
        mpl.figure()
        mpl.imshow((sobelimage[:, :, 0]**2 + sobelimage[:, :, 1]**2)**.5)
    return sobelimage

def findparticles(image, noisy = False):
    gradient = sobeloperator(image, noisy = noisy)
    averagegradient = numpy.average((gradient[:, :, 0]**2 + gradient[:, :, 1]**2)**.5)
    print(averagegradient)

def doniachsunjic(values, asymmetry, width, center):
    return numpy.cos((numpy.pi * asymmetry / 2) + (1 - asymmetry) * numpy.arctan((values - center) / width)) / (width**2 + (values - center)**2)**((1 - asymmetry) / 2)

# This is used to model the oxygen reduction reaction with diffusion 
# limitations
def orrscansim(timestep, estart, estop, scanspeed, kreact, alpha, activop, inithto, initot, inithp, diffhto, diffot, diffhp, area, maxiterations = 100000):
    # Set up some constants
    nfprt = 4 * 96485.322 / (8.314 * 298.15)
    iarpaf = inithto**2 / (initot * inithp**4)
    erev = 1.22 - nfprt**-1 * numpy.log(iarpaf)
    # Initialize the parameters
    # The stats are:
    # [0] E
    # [1] [H2O]
    # [2] [O2]
    # [3] [H+]
    # [4] R(for) - R(rev)
    # [5] H2O resupply
    # [6] O2 resupply
    # [7] H+ resupply
    # [8] d[H2O]/dt
    # [9] d[O2]/dt
    # [10] d[H+]/dt
    # [11] I
    prevstats = numpy.zeros((12))
    curstats = numpy.zeros((12))
    curstats[0] = estart
    curstats[1] = inithto
    curstats[2] = initot
    curstats[3] = inithp
    curstats[4] = kreact * (curstats[2] * curstats[3]**4 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + curstats[0] - erev)) - curstats[1]**2 * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + curstats[0] - erev)))
    curstats[5] = diffhto * (inithto - curstats[1])
    curstats[6] = diffot * (initot - curstats[2])
    curstats[7] = diffhp * (inithp - curstats[3])
    curstats[8] = 2 * curstats[5] + curstats[6]
    curstats[9] = -1 * curstats[5] + curstats[7]
    curstats[10] = -4 * curstats[5] + curstats[8]
    curstats[11] = -4 * 96485.322 * area * curstats[4]
    nextstats = numpy.zeros((12))
    nextstats[0] = curstats[0] - scanspeed * timestep
    nextstats[1] = curstats[1] + curstats[8] * timestep
    nextstats[2] = curstats[2] + curstats[9] * timestep
    nextstats[3] = curstats[3] + curstats[10] * timestep
    nextstats[4] = kreact * (nextstats[2] * nextstats[3]**4 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + nextstats[0] - erev)) - nextstats[1]**2 * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + nextstats[0] - erev)))
    nextstats[5] = diffhto * (inithto - nextstats[1])
    nextstats[6] = diffot * (initot - nextstats[2])
    nextstats[7] = diffhp * (inithp - nextstats[3])
    nextstats[8] = 2 * nextstats[4] + nextstats[5]
    nextstats[9] = -1 * nextstats[4] + nextstats[6]
    nextstats[10] = -4 * nextstats[4] + nextstats[7]
    nextstats[11] = -4 * 96485.322 * area * nextstats[4]
    lsvlog = []
    iteration = 0
    while curstats[0] >= estop and iteration < maxiterations:
        prevstats = curstats
        curstats = nextstats
        nextstats = numpy.zeros((12))
        lsvlog.append(prevstats)
        currenttimestep = timestep
        # Prevent positive feedback loops by reducing the time step if the concentration derivative of any chemical suddenly changes sign
        if curstats[8] * prevstats[8] < 0 or curstats[9] * prevstats[9] < 0 or curstats[10] * prevstats[10] < 0:
            if (curstats[1] > prevstats[1] and (curstats[1] + curstats[8] * currenttimestep) < prevstats[1]) or (curstats[1] < prevstats[1] and (curstats[1] + curstats[8] * currenttimestep) > prevstats[1]):
                currenttimestep = (prevstats[1] - curstats[1]) / curstats[8]
            if (curstats[2] > prevstats[2] and (curstats[2] + curstats[9] * currenttimestep) < prevstats[2]) or (curstats[2] < prevstats[2] and (curstats[2] + curstats[9] * currenttimestep) > prevstats[2]):
                currenttimestep = (prevstats[2] - curstats[2]) / curstats[9]
            if (curstats[3] > prevstats[3] and (curstats[3] + curstats[10] * currenttimestep) < prevstats[3]) or (curstats[3] < prevstats[3] and (curstats[3] + curstats[10] * currenttimestep) > prevstats[3]):
                currenttimestep = (prevstats[3] - curstats[3]) / curstats[10]
        # Prevent nonsensical values by making sure that the concentrations can never be negative
        if curstats[1] + curstats[8] * currenttimestep < 0 or curstats[2] + curstats[9] * currenttimestep < 0 or curstats[3] + curstats[10] * currenttimestep < 0:
            if curstats[8] < 0 and -1 * curstats[1] / curstats[8] < currenttimestep:
                currenttimestep = -.1 * curstats[1] / curstats[8]
            if curstats[9] < 0 and -1 * curstats[2] / curstats[9] < currenttimestep:
                currenttimestep = -.1 * curstats[2] / curstats[9]
            if curstats[10] < 0 and -1 * curstats[3] / curstats[10] < currenttimestep:
                currenttimestep = -.1 * curstats[3] / curstats[10]
        nextstats[0] = curstats[0] - scanspeed * currenttimestep
        nextstats[1] = curstats[1] + curstats[8] * currenttimestep
        nextstats[2] = curstats[2] + curstats[9] * currenttimestep
        nextstats[3] = curstats[3] + curstats[10] * currenttimestep
        nextstats[4] = kreact * (nextstats[2] * nextstats[3]**4 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + nextstats[0] - erev)) - nextstats[1]**2 * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + nextstats[0] - erev)))
        nextstats[5] = diffhto * (inithto - nextstats[1])
        nextstats[6] = diffot * (initot - nextstats[2])
        nextstats[7] = diffhp * (inithp - nextstats[3])
        nextstats[8] = 2 * nextstats[4] + nextstats[5]
        nextstats[9] = -1 * nextstats[4] + nextstats[6]
        nextstats[10] = -4 * nextstats[4] + nextstats[7]
        nextstats[11] = -4 * 96485.322 * area * nextstats[4]
        iteration = iteration + 1
    # The following reduces the size of the output data to be approximately equal to what would be expected if there had been no step resizing during the run
    onset = True
    inflection = True
    datapoint = 1
    onsetpot = lsvlog[1][0]
    while (onset or inflection) and datapoint < len(lsvlog) - 2:
        if lsvlog[datapoint][11] < -.00001 and onset:
            print(f'Onset potential: {lsvlog[datapoint][0]}')
            print(f'Current: {lsvlog[datapoint][11]}')
            onset = False
            onsetpot = lsvlog[datapoint][0]
        sdone = (((lsvlog[datapoint + 1][11] - lsvlog[datapoint][11]) / (lsvlog[datapoint + 1][0] - lsvlog[datapoint][0])) - ((lsvlog[datapoint][11] - lsvlog[datapoint - 1][11]) / (lsvlog[datapoint][0] - lsvlog[datapoint - 1][0]))) / ((lsvlog[datapoint + 1][0] - lsvlog[datapoint - 1][0]) / 2)
        sdtwo = (((lsvlog[datapoint + 2][11] - lsvlog[datapoint + 1][11]) / (lsvlog[datapoint + 2][0] - lsvlog[datapoint + 1][0])) - ((lsvlog[datapoint + 1][11] - lsvlog[datapoint][11]) / (lsvlog[datapoint + 1][0] - lsvlog[datapoint][0]))) / ((lsvlog[datapoint + 2][0] - lsvlog[datapoint][0]) / 2)
        if sdone <= 0 and sdtwo > 0 and inflection and lsvlog[datapoint][0] < onsetpot:
            print(f'Inflection point: {lsvlog[datapoint][0]}')
            print(f'Current: {lsvlog[datapoint][11]}')
            inflection = False
        datapoint = datapoint + 1
    return numpy.array(lsvlog)

def hercvsim(timestep, cycles, estart, eupper, elower, scanspeed, kreact, alpha, activop, initht, inithp, diffht, diffhp, area, maxiterations = 100000):
    # Set up some constants
    nfprt = 2 * 96485.322 / (8.314 * 298.15)
    iarpaf = initht / inithp**2
    erev = 0 - nfprt**-1 * numpy.log(iarpaf)
    # Initialize the parameters
    # The stats are:
    # [0] E
    # [1] [H2]
    # [2] [H+]
    # [3] R(for) - R(rev)
    # [4] H2 resupply
    # [5] H+ resupply
    # [6] d[H2]/dt
    # [7] d[H+]/dt
    # [8] I
    prevstats = numpy.zeros((9))
    curstats = numpy.zeros((9))
    curstats[0] = estart
    curstats[1] = initht
    curstats[2] = inithp
    curstats[3] = kreact * (curstats[2]**2 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + curstats[0] - erev)) - curstats[1] * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + curstats[0] - erev)))
    curstats[4] = diffht * (initht - curstats[1])
    curstats[5] = diffhp * (inithp - curstats[2])
    curstats[6] = curstats[3] + curstats[4]
    curstats[7] = -2 * curstats[3] + curstats[5]
    curstats[8] = -2 * 96485.322 * area * curstats[3]
    nextstats = numpy.zeros((9))
    nextstats[0] = curstats[0] - scanspeed * timestep
    nextstats[1] = curstats[1] + curstats[6] * timestep
    nextstats[2] = curstats[2] + curstats[7] * timestep
    nextstats[3] = kreact * (nextstats[2]**2 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + nextstats[0] - erev)) - nextstats[1] * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + nextstats[0] - erev)))
    nextstats[4] = diffht * (initht - nextstats[1])
    nextstats[5] = diffhp * (inithp - nextstats[3])
    nextstats[6] = nextstats[3] + nextstats[4]
    nextstats[7] = -2 * nextstats[3] + nextstats[5]
    nextstats[8] = -2 * 96485.322 * area * nextstats[3]
    cvlog = []
    iteration = 0
    cyclenumber = 0
    direction = -1
    while (cyclenumber < cycles or (cyclenumber == cycles and curstats[0] > estart)) and iteration < maxiterations:
        prevstats = curstats
        curstats = nextstats
        nextstats = numpy.zeros((9))
        cvlog.append(prevstats)
        currenttimestep = timestep
        # Prevent positive feedback loops by reducing the time step if the concentration derivative of any chemical suddenly changes sign
        if curstats[6] * prevstats[6] < 0 or curstats[7] * prevstats[7] < 0:
            if (curstats[1] > prevstats[1] and (curstats[1] + curstats[6] * currenttimestep) < prevstats[1]) or (curstats[1] < prevstats[1] and (curstats[1] + curstats[6] * currenttimestep) > prevstats[1]):
                currenttimestep = (prevstats[1] - curstats[1]) / curstats[6]
            if (curstats[2] > prevstats[2] and (curstats[2] + curstats[7] * currenttimestep) < prevstats[2]) or (curstats[2] < prevstats[2] and (curstats[2] + curstats[7] * currenttimestep) > prevstats[2]):
                currenttimestep = (prevstats[2] - curstats[2]) / curstats[7]
        # Prevent nonsensical values by making sure that the concentrations can never be negative
        if curstats[1] + curstats[6] * currenttimestep < 0 or curstats[2] + curstats[7] * currenttimestep < 0:
            if curstats[6] < 0 and -1 * curstats[1] / curstats[6] < currenttimestep:
                currenttimestep = -.1 * curstats[1] / curstats[6]
            if curstats[7] < 0 and -1 * curstats[2] / curstats[7] < currenttimestep:
                currenttimestep = -.1 * curstats[2] / curstats[7]
        nextstats[0] = curstats[0] + direction * scanspeed * currenttimestep
        nextstats[1] = curstats[1] + curstats[6] * currenttimestep
        nextstats[2] = curstats[2] + curstats[7] * currenttimestep
        nextstats[3] = kreact * (nextstats[2]**2 * iarpaf**alpha * numpy.exp(-1 * alpha * nfprt * (activop + nextstats[0] - erev)) - nextstats[1] * iarpaf**(-1 * (1 - alpha)) * numpy.exp((1 - alpha) * nfprt * (-1 * activop + nextstats[0] - erev)))
        nextstats[4] = diffht * (initht - nextstats[1])
        nextstats[5] = diffhp * (inithp - nextstats[2])
        nextstats[6] = nextstats[3] + nextstats[4]
        nextstats[7] = -2 * nextstats[3] + nextstats[5]
        nextstats[8] = -2 * 96485.322 * area * nextstats[4]
        # Hydrogen will form a gaseous phase if above a certain concentration, so the concentration will be capped at a certain amount
        if nextstats[1] > .00079365:
            nextstats[1] = .00079365
        # If an endpoint has been reached, switch directions
        if direction < 0 and nextstats[0] < elower:
            direction = 1
            cyclenumber = cyclenumber + .5
        elif direction > 0 and nextstats[0] > eupper:
            direction = -1
            cyclenumber = cyclenumber + .5
        iteration = iteration + 1
    # The following reduces the size of the output data to be approximately equal to what would be expected if there had been no step resizing during the run
    outlog = []
    outlog.append(cvlog[0])
    for i in range(1, len(cvlog)):
        if numpy.abs(cvlog[i][0] - outlog[1][0]) > .75 * timestep * scanspeed:
            outlog.append(cvlog[i])
    return numpy.array(outlog)

# This is used to simulate a graphene HAADF STEM image
def graphenehaadfstemimage(imagesize, scale, orientation):
    outimage = numpy.random.random(imagesize)
    actualorientation = (orientation - (numpy.pi / 6)) / (numpy.pi / 3)
    if actualorientation >= 0:
        actualorientation = (numpy.pi / 3) * (actualorientation - numpy.floor(actualorientation))
    else:
        actualorientation = (numpy.pi / 3) * (actualorientation - numpy.ceil(actualorientation))
    avector = numpy.array([.246 * numpy.cos(actualorientation) * scale, .246 * numpy.sin(actualorientation) * scale])
    bvector = numpy.array([.246 * numpy.cos(actualorientation + (numpy.pi / 3)) * scale, .246 * numpy.sin(actualorientation + (numpy.pi / 3)) * scale])
    offset = numpy.array([.142 * numpy.cos(orientation) * scale, .142 * numpy.sin(orientation) * scale])
    origin = numpy.array([imagesize[0] / 2, imagesize[1] / 2])
    pointlst = []
    if avector[0] == 0:
        highx = numpy.ceil(imagesize[1] / 2)
        lowx = numpy.floor(-1 * imagesize[1] / 2)
    elif avector[0] + bvector[0] >= 0 and avector[1] + bvector[1] >= 0:
        highx = numpy.ceil(numpy.max([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif avector[0] + bvector[0] >= 0 and avector[1] + bvector[1] < 0:
        highx = numpy.ceil(numpy.max([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif avector[0] + bvector[0] < 0 and avector[1] + bvector[1] >= 0:
        highx = numpy.ceil(numpy.max([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif avector[0] + bvector[0] < 0 and avector[1] + bvector[1] < 0:
        highx = numpy.ceil(numpy.max([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    print([lowx, highx])
    for i in range(int(lowx), int(highx)):
        srchy = True
        j = 0
        while srchy:
            baspnt = origin + i * avector + j * bvector
            offpnt = origin + i * avector + j * bvector + offset
            pointlst.append(baspnt)
            pointlst.append(offpnt)
            if baspnt[1] >= 0 and baspnt[1] <= outimage.shape[0] - 1 and offpnt[1] >= 0 and offpnt[1] <= outimage.shape[0] - 1:
                j = j + 1
            else:
                srchy = False
        srchy = True
        j = -1
        while srchy:
            baspnt = origin + i * avector + j * bvector
            offpnt = origin + i * avector + j * bvector + offset
            pointlst.append(baspnt)
            pointlst.append(offpnt)
            if baspnt[1] >= 0 and baspnt[1] <= outimage.shape[0] - 1 and offpnt[1] >= 0 and offpnt[1] <= outimage.shape[0] - 1:
                j = j - 1
            else:
                srchy = False
    for i in range(outimage.shape[0]):
        for j in range(outimage.shape[1]):
            for point in pointlst:
                outimage[j][i] = outimage[j][i] + numpy.exp(-1 * ((i - point[0])**2 + (j - point[1])**2) / (.426 * scale))
    outfft = numpy.fft.fft2(outimage)
    outfft = numpy.log10((outfft.real**2 + outfft.imag**2)**.5)
    rearrfft = numpy.zeros(imagesize)
    rearrfft[0:int(imagesize[1] / 2), :] = outfft[int(imagesize[1] / 2):imagesize[1], :]
    rearrfft[int(imagesize[1] / 2):imagesize[1], :] = outfft[0:int(imagesize[1] / 2), :]
    outfft[:, 0:int(imagesize[0] / 2)] = rearrfft[:, int(imagesize[0] / 2):imagesize[0]]
    outfft[:, int(imagesize[0] / 2):imagesize[0]] = rearrfft[:, 0:int(imagesize[0] / 2)]
    mpl.figure()
    mpl.imshow(outimage)
    mpl.figure()
    mpl.imshow(outfft)

# This is used to simulate a HAADF STEM image for a material of the specified 
# lattice parameters
def metalhaadfstemimage(imagesize, latticea, latticeb, orientation, basis = [[0, 0]]):
    outimage = numpy.random.random(imagesize)
    if orientation >= 0:
        actualorientation = orientation - (numpy.pi) * numpy.floor(orientation / (numpy.pi))
    else:
        actualorientation = orientation - (numpy.pi) * numpy.ceil(orientation / (numpy.pi))
    avector = numpy.array([latticea * numpy.cos(actualorientation), latticea * numpy.sin(actualorientation)])
    bvector = numpy.array([latticeb * numpy.cos(actualorientation + (numpy.pi / 2)), latticeb * numpy.sin(actualorientation + (numpy.pi / 2))])
    origin = numpy.array([imagesize[0] / 2, imagesize[1] / 2])
    width = numpy.min([latticea, latticeb])
    for i in range(1, len(basis)):
        if ((latticea * basis[i][0])**2 + (latticeb * basis[i][1])**2)**.5 < width:
            width = ((latticea * basis[i][0])**2 + (latticeb * basis[i][1])**2)**.5
    pointlst = []
    if avector[0] == 0:
        highx = numpy.ceil(imagesize[0] / (2 * latticea))
        lowx = numpy.floor(-1 * imagesize[0] / (2 * latticea))
    elif avector[1] == 0:
        highx = numpy.ceil(imagesize[1] / (2 * latticea))
        lowx = numpy.floor(-1 * imagesize[1] / (2 * latticea))
    elif bvector[0] >= 0 and bvector[1] >= 0:
        highx = numpy.ceil(numpy.max([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif bvector[0] >= 0 and bvector[1] < 0:
        highx = numpy.ceil(numpy.max([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif bvector[0] < 0 and bvector[1] >= 0:
        highx = numpy.ceil(numpy.max([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((-1 * imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    elif bvector[0] < 0 and bvector[1] < 0:
        highx = numpy.ceil(numpy.max([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
        lowx = numpy.floor(numpy.min([((-1 * imagesize[1] / 2) * bvector[1] - (imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1]), ((imagesize[1] / 2) * bvector[1] - (-1 * imagesize[0] / 2) * bvector[0]) / (avector[0] * bvector[1] - bvector[0] * avector[1])]))
    print([lowx, highx])
    for i in range(int(lowx), int(highx)):
        srchy = True
        j = 0
        while srchy:
            for point in basis:
                basepoint = origin + (i + point[0]) * avector + (j + point[1]) * bvector
                pointlst.append(basepoint)
                if basepoint[1] < 0 or basepoint[1] > outimage.shape[0] - 1:
                    srchy = False
            if srchy:
                j = j + 1
        srchy = True
        j = -1
        while srchy:
            for point in basis:
                basepoint = origin + (i + point[0]) * avector + (j + point[1]) * bvector
                pointlst.append(basepoint)
                if basepoint[1] < 0 or basepoint[1] > outimage.shape[0] - 1:
                    srchy = False
            if srchy:
                j = j - 1
    for i in range(outimage.shape[0]):
        for j in range(outimage.shape[1]):
            for point in pointlst:
                outimage[j][i] = outimage[j][i] + numpy.exp(-1 * ((i - point[0])**2 + (j - point[1])**2) / width)
    outfft = numpy.fft.fft2(outimage)
    outfft = numpy.log10((outfft.real**2 + outfft.imag**2)**.5)
    rearrfft = numpy.zeros(imagesize)
    rearrfft[0:int(imagesize[1] / 2), :] = outfft[int(imagesize[1] / 2):imagesize[1], :]
    rearrfft[int(imagesize[1] / 2):imagesize[1], :] = outfft[0:int(imagesize[1] / 2), :]
    outfft[:, 0:int(imagesize[0] / 2)] = rearrfft[:, int(imagesize[0] / 2):imagesize[0]]
    outfft[:, int(imagesize[0] / 2):imagesize[0]] = rearrfft[:, 0:int(imagesize[0] / 2)]
    mpl.figure()
    mpl.imshow(outimage)
    mpl.figure()
    mpl.imshow(outfft)

# This is used to get the general shape of an XPS curve for a specific element 
# by finding the energies at which the curve reaches specific heights relative 
# to the peak
def xpscurvetracing(spectrum, threshold = .05):
    peaklist = []
    troughlist = []
    trace = []
    peakwidth = int(numpy.ceil(threshold * spectrum.shape[0] / 2))
    for i in range(spectrum.shape[0]):
        ispeak = True
        istrough = True
        j = int(numpy.max([-1 * peakwidth, -1 * i]))
        while j < peakwidth and i + j < spectrum.shape[0]:
            if spectrum[i][1] < spectrum[i + j][1]:
                ispeak = False
            elif spectrum[i][1] > spectrum[i + j][1]:
                istrough = False
            j = j + 1
        if ispeak:
            peaklist.append(i)
        if istrough:
            troughlist.append(i)
    for peak in peaklist:
        print(f'Peak: {spectrum[peak]}')
    for trough in troughlist:
        print(f'Trough: {spectrum[trough]}')
    currenttrough = troughlist.pop(0)
    while len(peaklist) > 0 and len(troughlist) > 0:
        currentpeak = peaklist.pop(0)
        while currentpeak < currenttrough:
            currentpeak = peaklist.pop(0)
        traceheight = (spectrum[currentpeak][1] - spectrum[currenttrough][1]) / 4
        trace.append(spectrum[currenttrough])
        for i in range(currenttrough, currentpeak):
            if spectrum[i][1] > spectrum[currenttrough][1] + traceheight:
                trace.append([spectrum[i][0] - (((spectrum[i][1] - (spectrum[currenttrough][1] + traceheight)) / (spectrum[i][1] - spectrum[i - 1][1])) * (spectrum[i][0] - spectrum[i - 1][0])), spectrum[currenttrough][1] + traceheight])
                traceheight = traceheight + (spectrum[currentpeak][1] - spectrum[currenttrough][1]) / 4
        currenttrough = troughlist.pop(0)
        while currenttrough < currentpeak:
            currenttrough = troughlist.pop(0)
        traceheight = (spectrum[currentpeak][1] - spectrum[currenttrough][1]) / 4
        trace.append(spectrum[currentpeak])
        for i in range(currentpeak, currenttrough):
            if spectrum[i][1] < spectrum[currentpeak][1] - traceheight:
                trace.append([spectrum[i][0] - (((spectrum[i][1] - (spectrum[currentpeak][1] - traceheight)) / (spectrum[i][1] - spectrum[i - 1][1])) * (spectrum[i][0] - spectrum[i - 1][0])), spectrum[currentpeak][1] - traceheight])
                traceheight = traceheight + (spectrum[currentpeak][1] - spectrum[currenttrough][1]) / 4
    trace.append(spectrum[currenttrough])
    return numpy.array(trace)

# This is used to fix the two heights of an XPS spectrum for a specific 
# element with two peaks (e.g. Pt, Pd, Au) at a height of 1 and the lowest 
# values of the spectrum outside and between those features at a height of 0. 
# This is for easier comparison between different spectra.
def xpscurvenormalize(spectrum, threshold = .05):
    peaklist = []
    troughlist = []
    trace = numpy.ones(spectrum.shape)
    trace[:, 0] = spectrum[:, 0]
    peakwidth = int(numpy.ceil(threshold * spectrum.shape[0] / 2))
    for i in range(spectrum.shape[0]):
        ispeak = True
        istrough = True
        j = int(numpy.max([-1 * peakwidth, -1 * i]))
        while j < peakwidth and i + j < spectrum.shape[0]:
            if spectrum[i][1] < spectrum[i + j][1]:
                ispeak = False
            elif spectrum[i][1] > spectrum[i + j][1]:
                istrough = False
            j = j + 1
        if ispeak:
            peaklist.append(i)
        if istrough:
            troughlist.append(i)
    for peak in peaklist:
        print(f'Peak: {spectrum[peak]}')
    for trough in troughlist:
        print(f'Trough: {spectrum[trough]}')
    currenttrough = troughlist.pop(0)
    while len(peaklist) > 0 and len(troughlist) > 0:
        currentpeak = peaklist.pop(0)
        while currentpeak < currenttrough:
            currentpeak = peaklist.pop(0)
        trace[currenttrough:currentpeak, 1] = (spectrum[currenttrough:currentpeak, 1] - spectrum[currenttrough][1]) / (spectrum[currentpeak][1] - spectrum[currenttrough][1])
        currenttrough = troughlist.pop(0)
        while currenttrough < currentpeak:
            currenttrough = troughlist.pop(0)
        trace[currentpeak:currenttrough, 1] = (spectrum[currentpeak:currenttrough, 1] - spectrum[currenttrough][1]) / (spectrum[currentpeak][1] - spectrum[currenttrough][1])
    return trace

def iterativeconvolution(image, radius, iterations):
    convolution = numpy.zeros(image.shape)
    gaussfilter = numpy.zeros((2 * radius + 1, 2 * radius + 1))
    for l in range(-1 * radius, radius + 1):
        for k in range(-1 * numpy.round(radius**2 - numpy.abs(l - .5)**2), numpy.round(radius**2 - numpy.abs(l - .5)**2) + 1):
            gaussfilter[radius + l][radius + k] = numpy.exp(-1 * (l**2 + k**2) / (2))
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            convolvescore = 0
            for l in range(-1 * radius, radius + 1):
                for k in range(-1 * numpy.round(radius**2 - numpy.abs(l - .5)**2), numpy.round(radius**2 - numpy.abs(l - .5)**2) + 1):
                    convolvescore = convolvescore

def middlebias(bottom, top, median, deviation):
    probdistfunc = numpy.zeros((499, 2))
    orgprobdist = numpy.zeros((501, 3))
    stepsize = (top - bottom) / 500
    for i in range(501):
        orgprobdist[i][0] = i * stepsize
        orgprobdist[i][1] = numpy.exp(-1 * (i * stepsize - median)**2 / (2 * deviation**2)) / (deviation * (2 * numpy.pi)**.5)
        orgprobdist[i][2] = .5 * (scipy.special.erf((i * stepsize - median) / (deviation * 2**.5)) + 1)
    for i in range(1, 500):
        probdistfunc[i - 1][0] = i * stepsize
        for j in range(i, 501):
            for k in range(0, i):
                probdistfunc[i - 1][1] = probdistfunc[i - 1][1] + stepsize**2 * (orgprobdist[i][1] * orgprobdist[j][1] * orgprobdist[k][1]) / (orgprobdist[j][2] * (orgprobdist[j][2] - orgprobdist[k][2]))
        for j in range(0, i):
            for k in range(i, 501):
                probdistfunc[i - 1][1] = probdistfunc[i - 1][1] + stepsize**2 * (orgprobdist[i][1] * orgprobdist[j][1] * orgprobdist[k][1]) / ((1 - orgprobdist[j][2]) * (orgprobdist[k][2] - orgprobdist[j][2]))
        probdistfunc[i - 1][1] = probdistfunc[i - 1][1] / 2
    mpl.figure()
    mpl.plot(orgprobdist[:, 0], orgprobdist[:, 1])
    mpl.plot(probdistfunc[:, 0], probdistfunc[:, 1])
    numpy.savetxt('C:\\Users\\shini\\Documents\\Thesis\\Dissertation assets\\middlebiasedselection.csv', probdistfunc, delimiter = ',')

if __name__ == "__main__":
    #flname = input('Please enter the name of your file: ')
    #[fname, ftype] = flname.split('.')
    #if ftype == 'csv' or ftype == 'txt':
    #    imgdat = numpy.genfromtxt(flname, delimiter = ',')
    #else:
    #    imgdat = imageio.imread(flname)
    #smallimage = densitymap(clstda, (32, 32))
    #smallgradient = sobeloperator(smallimage)
    #correlatederivatives(smallgradient, error = 1.5)
    #start = time.time()
    #sobeloperator(imgdat, noisy = True)
    #mpl.figure()
    #mpl.imshow(imgdat)
    #end = time.time()
    #print(f'Clustering took {end - start} s')
    # orrscansim(timestep, estart, estop, scanspeed, kreact, alpha, activop, inithto, initot, inithp, diffhto, diffot, diffhp, area)
    # orrscansim(       s,      V,     V,       V/s, dm**?/s,     -,       V,   mol/L,  mol/L,  mol/L,      Hz,     Hz,     Hz, m**2)
# =============================================================================
#     orrlsv = orrscansim(1 * 10**-3, 1.22, -.1, .02, 1 * 10**-9, .5, 0, 55.5, 1.22 * 10**-3, 10**-1, .5, .5, .5, .00002)
#     mpl.figure()
#     mpl.plot(orrlsv[:, 0], orrlsv[:, 11])
#     newarr = []
#     stepcount = 0
#     for i in range(orrlsv.shape[0]):
#         if orrlsv[i][0] < orrlsv[0][0] - stepcount * .001:
#             newarr.append(orrlsv[i, :])
#             stepcount = stepcount + 1
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Carmen\\Model_Ap2.csv', numpy.array(newarr), delimiter = ',')
#     orrlsv = orrscansim(1 * 10**-3, 1.22, -.1, .02, 1 * 10**-9, .5, 0, 55.5, 1.22 * 10**-3, 10**-1, .5, .5, .5, .00004)
#     mpl.plot(orrlsv[:, 0], orrlsv[:, 11])
#     newarr = []
#     stepcount = 0
#     for i in range(orrlsv.shape[0]):
#         if orrlsv[i][0] < orrlsv[0][0] - stepcount * .001:
#             newarr.append(orrlsv[i, :])
#             stepcount = stepcount + 1
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Carmen\\Model_Ap4.csv', numpy.array(newarr), delimiter = ',')
#     orrlsv = orrscansim(1 * 10**-3, 1.22, -.1, .02, 1 * 10**-9, .5, 0, 55.5, 1.22 * 10**-3, 10**-1, .5, .5, .5, .00006)
#     mpl.plot(orrlsv[:, 0], orrlsv[:, 11])
#     newarr = []
#     stepcount = 0
#     for i in range(orrlsv.shape[0]):
#         if orrlsv[i][0] < orrlsv[0][0] - stepcount * .001:
#             newarr.append(orrlsv[i, :])
#             stepcount = stepcount + 1
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Carmen\\Model_Ap6.csv', numpy.array(newarr), delimiter = ',')
#     orrlsv = orrscansim(1 * 10**-3, 1.22, -.1, .02, 1 * 10**-9, .5, 0, 55.5, 1.22 * 10**-3, 10**-1, .5, .5, .5, .00008)
#     mpl.plot(orrlsv[:, 0], orrlsv[:, 11])
#     newarr = []
#     stepcount = 0
#     for i in range(orrlsv.shape[0]):
#         if orrlsv[i][0] < orrlsv[0][0] - stepcount * .001:
#             newarr.append(orrlsv[i, :])
#             stepcount = stepcount + 1
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Carmen\\Model_Ap8.csv', numpy.array(newarr), delimiter = ',')
#     orrlsv = orrscansim(1 * 10**-3, 1.22, -.1, .02, 1 * 10**-9, .5, 0, 55.5, 1.22 * 10**-3, 10**-1, .5, .5, .5, .0001)
#     mpl.plot(orrlsv[:, 0], orrlsv[:, 11])
#     newarr = []
#     stepcount = 0
#     for i in range(orrlsv.shape[0]):
#         if orrlsv[i][0] < orrlsv[0][0] - stepcount * .001:
#             newarr.append(orrlsv[i, :])
#             stepcount = stepcount + 1
#     numpy.savetxt(f'C:\\Users\\shini\\Documents\\Research\\Carmen\\Model_A1.csv', numpy.array(newarr), delimiter = ',')
# =============================================================================
    #graphenehaadfstemimage((128, 128), 50, numpy.pi / 3)
    #metalhaadfstemimage((128, 128), 13.95615, 20.3234, 1.277464, basis = [[0, 0], [.510845, .500815]])
    middlebias(0, 5, 2.5, .5)
# =============================================================================
#     trace = xpscurvenormalize(imgdat, threshold = .33)
#     numpy.savetxt(f'{fname}_normalized.csv', trace, delimiter = ',')
#     mpl.figure()
#     mpl.plot(imgdat[:, 0], imgdat[:, 1])
#     mpl.figure()
#     mpl.plot(trace[:, 0], trace[:, 1])
# =============================================================================
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], 2 * orrlsv[:, 4])
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 5])
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], -1 * orrlsv[:, 4])
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 6])
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], -4 * orrlsv[:, 4])
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 7])
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 8])
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 9])
    #mpl.figure()
    #mpl.plot(orrlsv[:, 0], orrlsv[:, 10])