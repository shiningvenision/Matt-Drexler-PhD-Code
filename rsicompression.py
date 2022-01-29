# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:08:18 2020

@author: shini
"""

import numpy
import fftcompression
import imageio
import matplotlib.pyplot as mpl
from scipy import stats

def imageevaluation(image, radius):
    categi = numpy.zeros((image.shape[0], image.shape[1], 3))
    #viwwin = numpy.zeros((2 * comprd + 1, 2 * comprd + 1))
    # No feature
    catega = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # Edge
    categb = numpy.array([[[1, 1, 1], [1, 1, 1], [.4, .4, .4]], [[1, 1, 1], [1, 1, .4], [1, .4, .4]], [[1, 1, .4], [1, 1, .4], [1, 1, .4]], [[1, .4, .4], [1, 1, .4], [1, 1, 1]], [[.4, .4, .4], [1, 1, 1], [1, 1, 1]], [[.4, .4, 1], [.4, 1, 1], [1, 1, 1]], [[.4, 1, 1], [.4, 1, 1], [.4, 1, 1]], [[1, 1, 1], [.4, 1, 1], [.4, .4, 1]]])
    # Ridge
    categc = numpy.array([[[.4, .4, .4], [1, 1, 1], [.4, .4, .4]], [[.4, .4, 1], [.4, 1, .4], [1, .4, .4]], [[.4, 1, .4], [.4, 1, .4], [.4, 1, .4]], [[1, .4, .4], [.4, 1, .4], [.4, .4, 1]]])
    # Peninsula
    categd = numpy.array([[[1, 1, 1], [.4, 1, .4], [.4, .4, .4]], [[1, 1, .4], [1, 1, .4], [.4, .4, .4]], [[1, .4, .4], [1, 1, .4], [1, .4, .4]], [[.4, .4, .4], [1, 1, .4], [1, 1, .4]], [[.4, .4, .4], [.4, 1, .4], [1, 1, 1]], [[.4, .4, .4], [.4, 1, 1], [.4, 1, 1]], [[.4, .4, 1], [.4, 1, 1], [.4, .4, 1]], [[.4, 1, 1], [.4, 1, 1], [.4, .4, .4]]])
    # Peak
    catege = numpy.array([[.4, .4, .4], [.4, 1, .4], [.4, .4, .4]])
    #j = int(orgimg.shape[0] * numpy.random.rand())
    #i = int(orgimg.shape[1] * numpy.random.rand())
    #print([i, j])
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            compix = numpy.zeros((3, 3, 2))
            for l in range(int(numpy.max([0, j - radius])), int(numpy.min([image.shape[0], j + radius + 1]))):
                if j == l:
                    highx = radius
                else:
                    highx = int(numpy.round((radius**2 - (numpy.abs(j - l) - .5)**2)**.5))
                for k in range(int(numpy.max([0, i - highx])), int(numpy.min([image.shape[1], i + highx + 1]))):
                    #viwwin[comprd - (j - l)][comprd - (i - k)] = orgimg[l][k]
                    mnrdst = ((i - k)**2 + (j - l)**2)**.5
                    if mnrdst <= radius / 2:
                        compix[1][1][0] = compix[1][1][0] + image[l][k]
                        compix[1][1][1] = compix[1][1][1] + 1
                    if mnrdst >= radius / 2:
                        pntsin = (l - j) / mnrdst
                        pntcos = (k - i) / mnrdst
                        if pntcos == 1: # Border of sectors (1, 0) and (2, 0)
                            compix[0][2][0] = compix[0][2][0] + image[l][k]
                            compix[1][2][0] = compix[1][2][0] + image[l][k]
                            compix[2][2][0] = compix[2][2][0] + image[l][k]
                            compix[0][2][1] = compix[0][2][1] + 1
                            compix[1][2][1] = compix[1][2][1] + 1
                            compix[2][2][1] = compix[2][2][1] + 1
                        elif pntcos > 1 / 2**.5:
                            if pntsin > 0: # Sector (1, 0)
                                compix[1][2][0] = compix[1][2][0] + image[l][k]
                                compix[2][2][0] = compix[2][2][0] + image[l][k]
                                compix[1][2][1] = compix[1][2][1] + 1
                                compix[2][2][1] = compix[2][2][1] + 1
                            else: # Sector (2, 0)
                                compix[1][2][0] = compix[1][2][0] + image[l][k]
                                compix[0][2][0] = compix[0][2][0] + image[l][k]
                                compix[1][2][1] = compix[1][2][1] + 1
                                compix[0][2][1] = compix[0][2][1] + 1
                        elif pntcos == 1 / 2**.5:
                            if pntsin > 0: # Border of sectors (0, 1) and (1, 0)
                                compix[1][2][0] = compix[1][2][0] + image[l][k]
                                compix[2][2][0] = compix[2][2][0] + image[l][k]
                                compix[2][1][0] = compix[2][1][0] + image[l][k]
                                compix[1][2][1] = compix[1][2][1] + 1
                                compix[2][2][1] = compix[2][2][1] + 1
                                compix[2][1][1] = compix[2][1][1] + 1
                            else: # Border of sectors (3, 1) and (2, 0)
                                compix[1][2][0] = compix[1][2][0] + image[l][k]
                                compix[0][2][0] = compix[0][2][0] + image[l][k]
                                compix[0][1][0] = compix[0][1][0] + image[l][k]
                                compix[1][2][1] = compix[1][2][1] + 1
                                compix[0][2][1] = compix[0][2][1] + 1
                                compix[0][1][1] = compix[0][1][1] + 1
                        elif pntcos > 0:
                            if pntsin > 0: # Sector (0, 1)
                                compix[2][2][0] = compix[2][2][0] + image[l][k]
                                compix[2][1][0] = compix[2][1][0] + image[l][k]
                                compix[2][2][1] = compix[2][2][1] + 1
                                compix[2][1][1] = compix[2][1][1] + 1
                            else: # Sector (3, 1)
                                compix[0][2][0] = compix[0][2][0] + image[l][k]
                                compix[0][1][0] = compix[0][1][0] + image[l][k]
                                compix[0][2][1] = compix[0][2][1] + 1
                                compix[0][1][1] = compix[0][1][1] + 1
                        elif pntcos == 0:
                            if pntsin > 0: # Border of sectors (0, 1) and (0, 2)
                                compix[2][2][0] = compix[2][2][0] + image[l][k]
                                compix[2][1][0] = compix[2][1][0] + image[l][k]
                                compix[2][0][0] = compix[2][0][0] + image[l][k]
                                compix[2][2][1] = compix[2][2][1] + 1
                                compix[2][1][1] = compix[2][1][1] + 1
                                compix[2][0][1] = compix[2][0][1] + 1
                            else: # Border of sectors (3, 1) and (3, 2)
                                compix[0][2][0] = compix[0][2][0] + image[l][k]
                                compix[0][1][0] = compix[0][1][0] + image[l][k]
                                compix[0][0][0] = compix[0][0][0] + image[l][k]
                                compix[0][2][1] = compix[0][2][1] + 1
                                compix[0][1][1] = compix[0][1][1] + 1
                                compix[0][0][1] = compix[0][0][1] + 1
                        elif pntcos > -1 / 2**.5:
                            if pntsin > 0: # Sector (0, 2)
                                compix[2][1][0] = compix[2][1][0] + image[l][k]
                                compix[2][0][0] = compix[2][0][0] + image[l][k]
                                compix[2][1][1] = compix[2][1][1] + 1
                                compix[2][0][1] = compix[2][0][1] + 1
                            else: # Sector (3, 2)
                                compix[0][1][0] = compix[0][1][0] + image[l][k]
                                compix[0][0][0] = compix[0][0][0] + image[l][k]
                                compix[0][1][1] = compix[0][1][1] + 1
                                compix[0][0][1] = compix[0][0][1] + 1
                        elif pntcos == -1 / 2**.5:
                            if pntsin > 0: # Border of sectors (0, 2) and (1, 3)
                                compix[2][1][0] = compix[2][1][0] + image[l][k]
                                compix[2][0][0] = compix[2][0][0] + image[l][k]
                                compix[1][0][0] = compix[1][0][0] + image[l][k]
                                compix[2][1][1] = compix[2][1][1] + 1
                                compix[2][0][1] = compix[2][0][1] + 1
                                compix[1][0][1] = compix[1][0][1] + 1
                            else: # Border of sectors (3, 2) and (2, 3)
                                compix[0][1][0] = compix[0][1][0] + image[l][k]
                                compix[0][0][0] = compix[0][0][0] + image[l][k]
                                compix[1][0][0] = compix[1][0][0] + image[l][k]
                                compix[0][1][1] = compix[0][1][1] + 1
                                compix[0][0][1] = compix[0][0][1] + 1
                                compix[1][0][1] = compix[1][0][1] + 1
                        elif pntcos > -1:
                            if pntsin > 0: # Sector (1, 3)
                                compix[2][0][0] = compix[2][0][0] + image[l][k]
                                compix[1][0][0] = compix[1][0][0] + image[l][k]
                                compix[2][0][1] = compix[2][0][1] + 1
                                compix[1][0][1] = compix[1][0][1] + 1
                            else: # Sector (2, 3)
                                compix[0][0][0] = compix[0][0][0] + image[l][k]
                                compix[1][0][0] = compix[1][0][0] + image[l][k]
                                compix[0][0][1] = compix[0][0][1] + 1
                                compix[1][0][1] = compix[1][0][1] + 1
                        elif pntcos == -1: # Border of sectors (1, 3) and (2, 3)
                            compix[2][0][0] = compix[2][0][0] + image[l][k]
                            compix[1][0][0] = compix[1][0][0] + image[l][k]
                            compix[0][0][0] = compix[0][0][0] + image[l][k]
                            compix[2][0][1] = compix[2][0][1] + 1
                            compix[1][0][1] = compix[1][0][1] + 1
                            compix[0][0][1] = compix[0][0][1] + 1
            minhgt = compix[1][1][0]
            for n in range(compix.shape[0]):
                for m in range(compix.shape[1]):
                    if compix[n][m][1] > 0:
                        compix[n][m][0] = compix[n][m][0] / compix[n][m][1]
                        if compix[n][m][0] < minhgt:
                            minhgt = compix[n][m][0]
            if minhgt == compix[1][1][0]:
                nrmpix = compix[:, :, 0] / minhgt
            else:
                nrmpix = (compix[:, :, 0] - minhgt) / (compix[1][1][0] - minhgt)
            mtchsc = 0
            categ = 0
            for m in range(compix.shape[0]):
                for n in range(compix.shape[1]):
                    if compix[m][n][1] > 0:
                        mtchsc = mtchsc + (nrmpix[m][n] - catega[m][n])**2
            for u in range(categb.shape[0]):
                cscore = 0
                for m in range(compix.shape[0]):
                    for n in range(compix.shape[1]):
                        if compix[m][n][1] > 0:
                            cscore = cscore + (nrmpix[m][n] - categb[u][m][n])**2
                if cscore < mtchsc:
                    categ = 1
                    mtchsc = cscore
            for u in range(categc.shape[0]):
                cscore = 0
                for m in range(compix.shape[0]):
                    for n in range(compix.shape[1]):
                        if compix[m][n][1] > 0:
                            cscore = cscore + (nrmpix[m][n] - categc[u][m][n])**2
                if cscore < mtchsc:
                    categ = 2
                    mtchsc = cscore
            for u in range(categd.shape[0]):
                cscore = 0
                for m in range(compix.shape[0]):
                    for n in range(compix.shape[1]):
                        if compix[m][n][1] > 0:
                            cscore = cscore + (nrmpix[m][n] - categd[u][m][n])**2
                if cscore < mtchsc:
                    categ = 3
                    mtchsc = cscore
            cscore = 0
            for m in range(compix.shape[0]):
                for n in range(compix.shape[1]):
                    if compix[m][n][1] > 0:
                        cscore = cscore + (nrmpix[m][n] - catege[m][n])**2
            if cscore < mtchsc:
                categ = 4
                mtchsc = cscore
            pekscr = [0, 0]
            for n in range(3):
                for m in range(3):
                    if compix[m][n][1] > 0:
                        pekscr[0] = pekscr[0] + (compix[1][1][0] - compix[m][n][0])
                        pekscr[1] = pekscr[1] + 1
            categi[j][i][0] = pekscr[0] / pekscr[1]
            categi[j][i][1] = categ
    return categi

def pointdetection(catimage, radius):
    pointlist = []
    aralst = []
    for l in range(-1 * radius, radius + 1):
        for k in range(-1 * int(numpy.round((radius**2 - (numpy.abs(l) - .5)**2)**.5)), int(numpy.round((radius**2 - (numpy.abs(l) - .5)**2)**.5)) + 1):
            m = 0
            srchng = True
            while m < len(aralst) and srchng:
                if (k**2 + l**2)**.5 <= (aralst[m][0]**2 + aralst[m][1]**2)**.5:
                    srchng = False
                else:
                    m = m + 1
            if srchng:
                aralst.append([k, l])
            else:
                aralst.insert(m, [k, l])
    aralst.pop(0)
    for j in range(catimage.shape[0]):
        for i in range(catimage.shape[1]):
            if catimage[j][i][1] > 0:
                m = 0
                srchng = True
                while m < len(aralst) and srchng:
                    if j + aralst[m][1] >= 0 and j + aralst[m][1] < catimage.shape[0] and i + aralst[m][0] >= 0 and i + aralst[m][0] < catimage.shape[1]:
                        if catimage[j + aralst[m][1]][i + aralst[m][0]][1] > catimage[j][i][1] or (catimage[j + aralst[m][1]][i + aralst[m][0]][0] > catimage[j][i][0] and catimage[j + aralst[m][1]][i + aralst[m][0]][1] == catimage[j][i][1]):
                            srchng = False
                        else:
                            m = m + 1
                    else:
                        m = m + 1
                if srchng:
                    pointlist.append([i, j])
    return pointlist

def visualizecategoryimage(categoryimage):
    outputimage = numpy.zeros(categoryimage.shape)
    scoremax = numpy.max(categoryimage[:, :, 0])
    scoremin = numpy.min(categoryimage[:, :, 0])
    for j in range(categoryimage.shape[0]):
        for i in range(categoryimage.shape[1]):
            outputimage[j][i][0] = (categoryimage[j][i][0] - scoremin) / (scoremax - scoremin)
            outputimage[j][i][1] = categoryimage[j][i][1] / 4
            outputimage[j][i][2] = (4 - categoryimage[j][i][1]) / 4
    return outputimage

def markpositions(image, points):
    markedimage = numpy.ones((image.shape[0], image.shape[1], 3))
    markedimage[:, :, 0] = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    markedimage[:, :, 1] = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    markedimage[:, :, 2] = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    for point in points:
        markedimage[point[1]][point[0]][0] = 1
        markedimage[point[1]][point[0]][1] = 0
        markedimage[point[1]][point[0]][2] = 0
        if point[0] > 0:
            markedimage[point[1]][point[0] - 1][0] = 1
            markedimage[point[1]][point[0] - 1][1] = 0
            markedimage[point[1]][point[0] - 1][2] = 0
        if point[1] > 0:
            markedimage[point[1] - 1][point[0]][0] = 1
            markedimage[point[1] - 1][point[0]][1] = 0
            markedimage[point[1] - 1][point[0]][2] = 0
        if point[0] < image.shape[1] - 1:
            markedimage[point[1]][point[0] + 1][0] = 1
            markedimage[point[1]][point[0] + 1][1] = 0
            markedimage[point[1]][point[0] + 1][2] = 0
        if point[1] < image.shape[0] - 1:
            markedimage[point[1] + 1][point[0]][0] = 1
            markedimage[point[1] + 1][point[0]][1] = 0
            markedimage[point[1] + 1][point[0]][2] = 0
    return markedimage

def testpointvsnoise(size = (21, 21), height = .75, noise = .25, breadth = 0):
    tstimg = numpy.ones(size)
    if breadth > 0:
        width = breadth**2 / (-1 * numpy.log(.1))
        radius = breadth
    else:
        width = ((numpy.min(size) - 1) / 4)**2 / (-1 * numpy.log(.1))
        radius = int((numpy.min(size) - 1) / 4)
    midpnt = [int((size[0] - 1) / 2), int((size[1] - 1) / 2)]
    for j in range(tstimg.shape[0]):
        for i in range(tstimg.shape[1]):
            tstimg[j][i] = height * numpy.exp(-1 * ((j - midpnt[0])**2 + (i - midpnt[1])**2) / width) + noise * numpy.random.rand()
    compix = numpy.zeros((3, 3, 2))
    for l in range(midpnt[0] - radius, midpnt[0] + radius + 1):
        if l == midpnt[0]:
            highx = radius
        else:
            highx = int(numpy.round((radius**2 - (numpy.abs(midpnt[0] - l) - .5)**2)**.5))
        for k in range(int(midpnt[1] - highx), int(midpnt[1] + highx + 1)):
            #viwwin[comprd - (j - l)][comprd - (i - k)] = orgimg[l][k]
            mnrdst = ((midpnt[1] - k)**2 + (midpnt[0] - l)**2)**.5
            if mnrdst <= radius / 2:
                compix[1][1][0] = compix[1][1][0] + tstimg[l][k]
                compix[1][1][1] = compix[1][1][1] + 1
            if mnrdst >= radius / 2:
                pntsin = (l - midpnt[0]) / mnrdst
                pntcos = (k - midpnt[1]) / mnrdst
                if pntcos == 1: # Border of sectors (1, 0) and (2, 0)
                    compix[0][2][0] = compix[0][2][0] + tstimg[l][k]
                    compix[1][2][0] = compix[1][2][0] + tstimg[l][k]
                    compix[2][2][0] = compix[2][2][0] + tstimg[l][k]
                    compix[0][2][1] = compix[0][2][1] + 1
                    compix[1][2][1] = compix[1][2][1] + 1
                    compix[2][2][1] = compix[2][2][1] + 1
                elif pntcos > 1 / 2**.5:
                    if pntsin > 0: # Sector (1, 0)
                        compix[1][2][0] = compix[1][2][0] + tstimg[l][k]
                        compix[2][2][0] = compix[2][2][0] + tstimg[l][k]
                        compix[1][2][1] = compix[1][2][1] + 1
                        compix[2][2][1] = compix[2][2][1] + 1
                    else: # Sector (2, 0)
                        compix[1][2][0] = compix[1][2][0] + tstimg[l][k]
                        compix[0][2][0] = compix[0][2][0] + tstimg[l][k]
                        compix[1][2][1] = compix[1][2][1] + 1
                        compix[0][2][1] = compix[0][2][1] + 1
                elif pntcos == 1 / 2**.5:
                    if pntsin > 0: # Border of sectors (0, 1) and (1, 0)
                        compix[1][2][0] = compix[1][2][0] + tstimg[l][k]
                        compix[2][2][0] = compix[2][2][0] + tstimg[l][k]
                        compix[2][1][0] = compix[2][1][0] + tstimg[l][k]
                        compix[1][2][1] = compix[1][2][1] + 1
                        compix[2][2][1] = compix[2][2][1] + 1
                        compix[2][1][1] = compix[2][1][1] + 1
                    else: # Border of sectors (3, 1) and (2, 0)
                        compix[1][2][0] = compix[1][2][0] + tstimg[l][k]
                        compix[0][2][0] = compix[0][2][0] + tstimg[l][k]
                        compix[0][1][0] = compix[0][1][0] + tstimg[l][k]
                        compix[1][2][1] = compix[1][2][1] + 1
                        compix[0][2][1] = compix[0][2][1] + 1
                        compix[0][1][1] = compix[0][1][1] + 1
                elif pntcos > 0:
                    if pntsin > 0: # Sector (0, 1)
                        compix[2][2][0] = compix[2][2][0] + tstimg[l][k]
                        compix[2][1][0] = compix[2][1][0] + tstimg[l][k]
                        compix[2][2][1] = compix[2][2][1] + 1
                        compix[2][1][1] = compix[2][1][1] + 1
                    else: # Sector (3, 1)
                        compix[0][2][0] = compix[0][2][0] + tstimg[l][k]
                        compix[0][1][0] = compix[0][1][0] + tstimg[l][k]
                        compix[0][2][1] = compix[0][2][1] + 1
                        compix[0][1][1] = compix[0][1][1] + 1
                elif pntcos == 0:
                    if pntsin > 0: # Border of sectors (0, 1) and (0, 2)
                        compix[2][2][0] = compix[2][2][0] + tstimg[l][k]
                        compix[2][1][0] = compix[2][1][0] + tstimg[l][k]
                        compix[2][0][0] = compix[2][0][0] + tstimg[l][k]
                        compix[2][2][1] = compix[2][2][1] + 1
                        compix[2][1][1] = compix[2][1][1] + 1
                        compix[2][0][1] = compix[2][0][1] + 1
                    else: # Border of sectors (3, 1) and (3, 2)
                        compix[0][2][0] = compix[0][2][0] + tstimg[l][k]
                        compix[0][1][0] = compix[0][1][0] + tstimg[l][k]
                        compix[0][0][0] = compix[0][0][0] + tstimg[l][k]
                        compix[0][2][1] = compix[0][2][1] + 1
                        compix[0][1][1] = compix[0][1][1] + 1
                        compix[0][0][1] = compix[0][0][1] + 1
                elif pntcos > -1 / 2**.5:
                    if pntsin > 0: # Sector (0, 2)
                        compix[2][1][0] = compix[2][1][0] + tstimg[l][k]
                        compix[2][0][0] = compix[2][0][0] + tstimg[l][k]
                        compix[2][1][1] = compix[2][1][1] + 1
                        compix[2][0][1] = compix[2][0][1] + 1
                    else: # Sector (3, 2)
                        compix[0][1][0] = compix[0][1][0] + tstimg[l][k]
                        compix[0][0][0] = compix[0][0][0] + tstimg[l][k]
                        compix[0][1][1] = compix[0][1][1] + 1
                        compix[0][0][1] = compix[0][0][1] + 1
                elif pntcos == -1 / 2**.5:
                    if pntsin > 0: # Border of sectors (0, 2) and (1, 3)
                        compix[2][1][0] = compix[2][1][0] + tstimg[l][k]
                        compix[2][0][0] = compix[2][0][0] + tstimg[l][k]
                        compix[1][0][0] = compix[1][0][0] + tstimg[l][k]
                        compix[2][1][1] = compix[2][1][1] + 1
                        compix[2][0][1] = compix[2][0][1] + 1
                        compix[1][0][1] = compix[1][0][1] + 1
                    else: # Border of sectors (3, 2) and (2, 3)
                        compix[0][1][0] = compix[0][1][0] + tstimg[l][k]
                        compix[0][0][0] = compix[0][0][0] + tstimg[l][k]
                        compix[1][0][0] = compix[1][0][0] + tstimg[l][k]
                        compix[0][1][1] = compix[0][1][1] + 1
                        compix[0][0][1] = compix[0][0][1] + 1
                        compix[1][0][1] = compix[1][0][1] + 1
                elif pntcos > -1:
                    if pntsin > 0: # Sector (1, 3)
                        compix[2][0][0] = compix[2][0][0] + tstimg[l][k]
                        compix[1][0][0] = compix[1][0][0] + tstimg[l][k]
                        compix[2][0][1] = compix[2][0][1] + 1
                        compix[1][0][1] = compix[1][0][1] + 1
                    else: # Sector (2, 3)
                        compix[0][0][0] = compix[0][0][0] + tstimg[l][k]
                        compix[1][0][0] = compix[1][0][0] + tstimg[l][k]
                        compix[0][0][1] = compix[0][0][1] + 1
                        compix[1][0][1] = compix[1][0][1] + 1
                elif pntcos == -1: # Border of sectors (1, 3) and (2, 3)
                    compix[2][0][0] = compix[2][0][0] + tstimg[l][k]
                    compix[1][0][0] = compix[1][0][0] + tstimg[l][k]
                    compix[0][0][0] = compix[0][0][0] + tstimg[l][k]
                    compix[2][0][1] = compix[2][0][1] + 1
                    compix[1][0][1] = compix[1][0][1] + 1
                    compix[0][0][1] = compix[0][0][1] + 1
    minhgt = compix[1][1][0]
    for n in range(compix.shape[0]):
        for m in range(compix.shape[1]):
            if compix[n][m][1] > 0:
                compix[n][m][0] = compix[n][m][0] / compix[n][m][1]
                if compix[n][m][0] < minhgt:
                    minhgt = compix[n][m][0]
    if minhgt == compix[1][1][0]:
        nrmpix = compix[:, :, 0] / minhgt
    else:
        nrmpix = (compix[:, :, 0] - minhgt) / (compix[1][1][0] - minhgt)
    print(f'Height, noise, width, radius, value 1, value 2, value 3, value 4, value 5, value 6, value 7, value 8')
    print([height, noise, width, radius, nrmpix[0][0], nrmpix[0][1], nrmpix[0][2], nrmpix[1][0], nrmpix[1][2], nrmpix[2][0], nrmpix[2][1], nrmpix[2][2]])
    fig, ax = mpl.subplots(1, 2)
    ax[0].imshow(tstimg)
    ax[1].imshow(nrmpix)

def reevaluatepoints(image, points, radius):
    evllst = []
    #viwwin = numpy.zeros((2 * comprd + 1, 2 * comprd + 1))
    # No feature
    catega = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # Edge
    categb = numpy.array([[[1, 1, 1], [1, 1, 1], [.4, .4, .4]], [[1, 1, 1], [1, 1, .4], [1, .4, .4]], [[1, 1, .4], [1, 1, .4], [1, 1, .4]], [[1, .4, .4], [1, 1, .4], [1, 1, 1]], [[.4, .4, .4], [1, 1, 1], [1, 1, 1]], [[.4, .4, 1], [.4, 1, 1], [1, 1, 1]], [[.4, 1, 1], [.4, 1, 1], [.4, 1, 1]], [[1, 1, 1], [.4, 1, 1], [.4, .4, 1]]])
    # Ridge
    categc = numpy.array([[[.4, .4, .4], [1, 1, 1], [.4, .4, .4]], [[.4, .4, 1], [.4, 1, .4], [1, .4, .4]], [[.4, 1, .4], [.4, 1, .4], [.4, 1, .4]], [[1, .4, .4], [.4, 1, .4], [.4, .4, 1]]])
    # Peninsula
    categd = numpy.array([[[1, 1, 1], [.4, 1, .4], [.4, .4, .4]], [[1, 1, .4], [1, 1, .4], [.4, .4, .4]], [[1, .4, .4], [1, 1, .4], [1, .4, .4]], [[.4, .4, .4], [1, 1, .4], [1, 1, .4]], [[.4, .4, .4], [.4, 1, .4], [1, 1, 1]], [[.4, .4, .4], [.4, 1, 1], [.4, 1, 1]], [[.4, .4, 1], [.4, 1, 1], [.4, .4, 1]], [[.4, 1, 1], [.4, 1, 1], [.4, .4, .4]]])
    # Peak
    catege = numpy.array([[.4, .4, .4], [.4, 1, .4], [.4, .4, .4]])
    #j = int(orgimg.shape[0] * numpy.random.rand())
    #i = int(orgimg.shape[1] * numpy.random.rand())
    #print([i, j])
    for pnt in points:
        i = pnt[0]
        j = pnt[1]
        compix = numpy.zeros((3, 3, 2))
        for l in range(int(numpy.max([0, j - radius])), int(numpy.min([image.shape[0], j + radius + 1]))):
            if j == l:
                highx = radius
            else:
                highx = int(numpy.round((radius**2 - (numpy.abs(j - l) - .5)**2)**.5))
            for k in range(int(numpy.max([0, i - highx])), int(numpy.min([image.shape[1], i + highx + 1]))):
                #viwwin[comprd - (j - l)][comprd - (i - k)] = orgimg[l][k]
                mnrdst = ((i - k)**2 + (j - l)**2)**.5
                if mnrdst <= radius / 2:
                    compix[1][1][0] = compix[1][1][0] + image[l][k]
                    compix[1][1][1] = compix[1][1][1] + 1
                if mnrdst >= radius / 2:
                    pntsin = (l - j) / mnrdst
                    pntcos = (k - i) / mnrdst
                    if pntcos == 1: # Border of sectors (1, 0) and (2, 0)
                        compix[0][2][0] = compix[0][2][0] + image[l][k]
                        compix[1][2][0] = compix[1][2][0] + image[l][k]
                        compix[2][2][0] = compix[2][2][0] + image[l][k]
                        compix[0][2][1] = compix[0][2][1] + 1
                        compix[1][2][1] = compix[1][2][1] + 1
                        compix[2][2][1] = compix[2][2][1] + 1
                    elif pntcos > 1 / 2**.5:
                        if pntsin > 0: # Sector (1, 0)
                            compix[1][2][0] = compix[1][2][0] + image[l][k]
                            compix[2][2][0] = compix[2][2][0] + image[l][k]
                            compix[1][2][1] = compix[1][2][1] + 1
                            compix[2][2][1] = compix[2][2][1] + 1
                        else: # Sector (2, 0)
                            compix[1][2][0] = compix[1][2][0] + image[l][k]
                            compix[0][2][0] = compix[0][2][0] + image[l][k]
                            compix[1][2][1] = compix[1][2][1] + 1
                            compix[0][2][1] = compix[0][2][1] + 1
                    elif pntcos == 1 / 2**.5:
                        if pntsin > 0: # Border of sectors (0, 1) and (1, 0)
                            compix[1][2][0] = compix[1][2][0] + image[l][k]
                            compix[2][2][0] = compix[2][2][0] + image[l][k]
                            compix[2][1][0] = compix[2][1][0] + image[l][k]
                            compix[1][2][1] = compix[1][2][1] + 1
                            compix[2][2][1] = compix[2][2][1] + 1
                            compix[2][1][1] = compix[2][1][1] + 1
                        else: # Border of sectors (3, 1) and (2, 0)
                            compix[1][2][0] = compix[1][2][0] + image[l][k]
                            compix[0][2][0] = compix[0][2][0] + image[l][k]
                            compix[0][1][0] = compix[0][1][0] + image[l][k]
                            compix[1][2][1] = compix[1][2][1] + 1
                            compix[0][2][1] = compix[0][2][1] + 1
                            compix[0][1][1] = compix[0][1][1] + 1
                    elif pntcos > 0:
                        if pntsin > 0: # Sector (0, 1)
                            compix[2][2][0] = compix[2][2][0] + image[l][k]
                            compix[2][1][0] = compix[2][1][0] + image[l][k]
                            compix[2][2][1] = compix[2][2][1] + 1
                            compix[2][1][1] = compix[2][1][1] + 1
                        else: # Sector (3, 1)
                            compix[0][2][0] = compix[0][2][0] + image[l][k]
                            compix[0][1][0] = compix[0][1][0] + image[l][k]
                            compix[0][2][1] = compix[0][2][1] + 1
                            compix[0][1][1] = compix[0][1][1] + 1
                    elif pntcos == 0:
                        if pntsin > 0: # Border of sectors (0, 1) and (0, 2)
                            compix[2][2][0] = compix[2][2][0] + image[l][k]
                            compix[2][1][0] = compix[2][1][0] + image[l][k]
                            compix[2][0][0] = compix[2][0][0] + image[l][k]
                            compix[2][2][1] = compix[2][2][1] + 1
                            compix[2][1][1] = compix[2][1][1] + 1
                            compix[2][0][1] = compix[2][0][1] + 1
                        else: # Border of sectors (3, 1) and (3, 2)
                            compix[0][2][0] = compix[0][2][0] + image[l][k]
                            compix[0][1][0] = compix[0][1][0] + image[l][k]
                            compix[0][0][0] = compix[0][0][0] + image[l][k]
                            compix[0][2][1] = compix[0][2][1] + 1
                            compix[0][1][1] = compix[0][1][1] + 1
                            compix[0][0][1] = compix[0][0][1] + 1
                    elif pntcos > -1 / 2**.5:
                        if pntsin > 0: # Sector (0, 2)
                            compix[2][1][0] = compix[2][1][0] + image[l][k]
                            compix[2][0][0] = compix[2][0][0] + image[l][k]
                            compix[2][1][1] = compix[2][1][1] + 1
                            compix[2][0][1] = compix[2][0][1] + 1
                        else: # Sector (3, 2)
                            compix[0][1][0] = compix[0][1][0] + image[l][k]
                            compix[0][0][0] = compix[0][0][0] + image[l][k]
                            compix[0][1][1] = compix[0][1][1] + 1
                            compix[0][0][1] = compix[0][0][1] + 1
                    elif pntcos == -1 / 2**.5:
                        if pntsin > 0: # Border of sectors (0, 2) and (1, 3)
                            compix[2][1][0] = compix[2][1][0] + image[l][k]
                            compix[2][0][0] = compix[2][0][0] + image[l][k]
                            compix[1][0][0] = compix[1][0][0] + image[l][k]
                            compix[2][1][1] = compix[2][1][1] + 1
                            compix[2][0][1] = compix[2][0][1] + 1
                            compix[1][0][1] = compix[1][0][1] + 1
                        else: # Border of sectors (3, 2) and (2, 3)
                            compix[0][1][0] = compix[0][1][0] + image[l][k]
                            compix[0][0][0] = compix[0][0][0] + image[l][k]
                            compix[1][0][0] = compix[1][0][0] + image[l][k]
                            compix[0][1][1] = compix[0][1][1] + 1
                            compix[0][0][1] = compix[0][0][1] + 1
                            compix[1][0][1] = compix[1][0][1] + 1
                    elif pntcos > -1:
                        if pntsin > 0: # Sector (1, 3)
                            compix[2][0][0] = compix[2][0][0] + image[l][k]
                            compix[1][0][0] = compix[1][0][0] + image[l][k]
                            compix[2][0][1] = compix[2][0][1] + 1
                            compix[1][0][1] = compix[1][0][1] + 1
                        else: # Sector (2, 3)
                            compix[0][0][0] = compix[0][0][0] + image[l][k]
                            compix[1][0][0] = compix[1][0][0] + image[l][k]
                            compix[0][0][1] = compix[0][0][1] + 1
                            compix[1][0][1] = compix[1][0][1] + 1
                    elif pntcos == -1: # Border of sectors (1, 3) and (2, 3)
                        compix[2][0][0] = compix[2][0][0] + image[l][k]
                        compix[1][0][0] = compix[1][0][0] + image[l][k]
                        compix[0][0][0] = compix[0][0][0] + image[l][k]
                        compix[2][0][1] = compix[2][0][1] + 1
                        compix[1][0][1] = compix[1][0][1] + 1
                        compix[0][0][1] = compix[0][0][1] + 1
        minhgt = compix[1][1][0]
        for n in range(compix.shape[0]):
            for m in range(compix.shape[1]):
                if compix[n][m][1] > 0:
                    compix[n][m][0] = compix[n][m][0] / compix[n][m][1]
                    if compix[n][m][0] < minhgt:
                        minhgt = compix[n][m][0]
        if minhgt == compix[1][1][0]:
            nrmpix = compix[:, :, 0] / minhgt
        else:
            nrmpix = (compix[:, :, 0] - minhgt) / (compix[1][1][0] - minhgt)
        mtchsc = 0
        categ = 0
        for m in range(compix.shape[0]):
            for n in range(compix.shape[1]):
                if compix[m][n][1] > 0:
                    mtchsc = mtchsc + (nrmpix[m][n] - catega[m][n])**2
        for u in range(categb.shape[0]):
            cscore = 0
            for m in range(compix.shape[0]):
                for n in range(compix.shape[1]):
                    if compix[m][n][1] > 0:
                        cscore = cscore + (nrmpix[m][n] - categb[u][m][n])**2
            if cscore < mtchsc:
                categ = 1
                mtchsc = cscore
        for u in range(categc.shape[0]):
            cscore = 0
            for m in range(compix.shape[0]):
                for n in range(compix.shape[1]):
                    if compix[m][n][1] > 0:
                        cscore = cscore + (nrmpix[m][n] - categc[u][m][n])**2
            if cscore < mtchsc:
                categ = 2
                mtchsc = cscore
        for u in range(categd.shape[0]):
            cscore = 0
            for m in range(compix.shape[0]):
                for n in range(compix.shape[1]):
                    if compix[m][n][1] > 0:
                        cscore = cscore + (nrmpix[m][n] - categd[u][m][n])**2
            if cscore < mtchsc:
                categ = 3
                mtchsc = cscore
        cscore = 0
        for m in range(compix.shape[0]):
            for n in range(compix.shape[1]):
                if compix[m][n][1] > 0:
                    cscore = cscore + (nrmpix[m][n] - catege[m][n])**2
        if cscore < mtchsc:
            categ = 4
            mtchsc = cscore
        #pekscr = [0, 0]
        #for n in range(3):
        #    for m in range(3):
        #        if compix[m][n][1] > 0:
        #            pekscr[0] = pekscr[0] + (compix[1][1][0] - compix[m][n][0])
        #            pekscr[1] = pekscr[1] + 1
        evllst.append([i, j, categ, nrmpix[0][0], nrmpix[0][1], nrmpix[0][2], nrmpix[1][2], nrmpix[2][2], nrmpix[2][1], nrmpix[2][0], nrmpix[1][0]])
    return evllst

def ttestfilter(image, radius, noisy = False):
    indexarray = [[], [], [], [], [], [], [], [], []]
    for j in range(-1 * radius, radius + 1):
        for i in range(-1 * numpy.round(radius**2 - (numpy.abs(j) - .5)**2)**.5, numpy.round(radius**2 - (numpy.abs(j) - .5)**2)**.5 + 1):
            if (i**2 + j**2)**.5 <= radius / 2:
                indexarray[0].append([i, j])
            else:
                pixcos = i / (i**2 + j**2)**.5
                pixsin = j / (i**2 + j**2)**.5
                if pixcos <= 0 and pixsin >= 0:
                    indexarray[1].append([i, j])
                if pixcos >= -.7071 and pixcos <= .7071 and pixsin > 0:
                    indexarray[2].append([i, j])
                if pixcos >= 0 and pixsin >= 0:
                    indexarray[3].append([i, j])
                if pixcos > 0 and pixsin >= -.7071 and pixsin <= .7071:
                    indexarray[4].append([i, j])
                if pixcos >= 0 and pixsin <= 0:
                    indexarray[5].append([i, j])
                if pixcos >= -.7071 and pixcos <= .7071 and pixsin < 0:
                    indexarray[6].append([i, j])
                if pixcos <= 0 and pixsin <= 0:
                    indexarray[7].append([i, j])
                if pixcos < 0 and pixsin >= -.7071 and pixsin <= -.7071:
                    indexarray[8].append([i, j])
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            samplearray = [[], [], [], [], [], [], [], [], []]
            for u in range(len(indexarray)):
                for v in range(len(indexarray[u])):
                    if i + indexarray[u][v][0] >= 0 and i + indexarray[u][v][0] < image.shape[1] and j + indexarray[u][v][1] >= 0 and j + indexarray[u][v][1] < image.shape[0]:
                        samplearray[u][v].append(image[j + indexarray[u][v][1]][i + indexarray[u][v][0]])
            statarray = numpy.zeros((9, 2))
            for u in range(len(samplearray)):
                for v in range(len(samplearray[u])):
                    statarray[u][0] = statarray[u][0] + samplearray[u][v]
                if len(samplearray[u]) > 0:
                    statarray[u][0] = statarray[u][0] / len(samplearray[u])
            for u in range(len(samplearray)):
                for v in range(len(samplearray[u])):
                    statarray[u][1] = statarray[u][1] + (samplearray[u][v] - statarray[u][0])**2
            ttestarray = numpy.zeros((8, 2))
            if len(samplearray[1]) > 0:
                if statarray[0][1] / statarray[1][1] >= .5 and statarray[0][1] / statarray[1][1] <= 2:
                    ttestarray[0] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[1]), equal_var = True)
                else:
                    ttestarray[0] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[1]), equal_var = False)
            if len(samplearray[2]) > 0:
                if statarray[0][1] / statarray[2][1] >= .5 and statarray[0][1] / statarray[2][1] <= 2:
                    ttestarray[1] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[2]), equal_var = True)
                else:
                    ttestarray[1] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[2]), equal_var = False)
            if len(samplearray[3]) > 0:
                if statarray[0][1] / statarray[3][1] >= .5 and statarray[0][1] / statarray[3][1] <= 2:
                    ttestarray[2] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[3]), equal_var = True)
                else:
                    ttestarray[2] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[3]), equal_var = False)
            if len(samplearray[4]) > 0:
                if statarray[0][1] / statarray[4][1] >= .5 and statarray[0][1] / statarray[4][1] <= 2:
                    ttestarray[3] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[4]), equal_var = True)
                else:
                    ttestarray[3] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[4]), equal_var = False)
            if len(samplearray[5]) > 0:
                if statarray[0][1] / statarray[5][1] >= .5 and statarray[0][1] / statarray[5][1] <= 2:
                    ttestarray[4] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[5]), equal_var = True)
                else:
                    ttestarray[4] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[5]), equal_var = False)
            if len(samplearray[6]) > 0:
                if statarray[0][1] / statarray[6][1] >= .5 and statarray[0][1] / statarray[6][1] <= 2:
                    ttestarray[5] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[6]), equal_var = True)
                else:
                    ttestarray[5] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[6]), equal_var = False)
            if len(samplearray[7]) > 0:
                if statarray[0][1] / statarray[7][1] >= .5 and statarray[0][1] / statarray[7][1] <= 2:
                    ttestarray[6] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[7]), equal_var = True)
                else:
                    ttestarray[6] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[7]), equal_var = False)
            if len(samplearray[8]) > 0:
                if statarray[0][1] / statarray[8][1] >= .5 and statarray[0][1] / statarray[8][1] <= 2:
                    ttestarray[7] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[8]), equal_var = True)
                else:
                    ttestarray[7] = stats.ttest(numpy.array(samplearray[0]), numpy.array(samplearray[8]), equal_var = False)
            

def spotatoms(image, radius, noisy = False):
    normalizedimage = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    if noisy:
        mpl.figure()
        mpl.imshow(normalizedimage)
    filteredimage = eyefilter(normalizedimage, radius)
    combinedimage = filteredimage[:, :, 0] - filteredimage[:, :, 1]
    if noisy:
        #displayimage = numpy.zeros((filteredimage.shape[0], filteredimage.shape[1], 3))
        #displayimage[:, :, 0] = (filteredimage[:, :, 0] - numpy.min(filteredimage[:, :, 0])) / (numpy.max(filteredimage[:, :, 0]) - numpy.min(filteredimage[:, :, 0]))
        #displayimage[:, :, 1] = (filteredimage[:, :, 1] - numpy.min(filteredimage[:, :, 1])) / (numpy.max(filteredimage[:, :, 1]) - numpy.min(filteredimage[:, :, 1]))
        mpl.figure()
        mpl.imshow(combinedimage)
    #combinedaverage = numpy.sum(combinedimage) / (combinedimage.shape[0] * combinedimage.shape[1])
    #combinedstd = (numpy.sum((combinedimage - combinedaverage)**2) / (combinedimage.shape[0] * combinedimage.shape[1]))**.5
    threshold = characterizenoise(normalizedimage)
    print(threshold)
    pointlist = []
    for j in range(combinedimage.shape[0]):
        for i in range(combinedimage.shape[1]):
            if combinedimage[j][i] > .5 * threshold:
                searching = True
                l = -1 * radius
                while l < radius + 1 and searching:
                    k = int(-1 * numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5)
                    while k < numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5 + 1 and searching:
                        if i + k > 0 and i + k < combinedimage.shape[1] - 1 and j + l > 0 and j + l < combinedimage.shape[0] - 1:
                            if combinedimage[j][i] < combinedimage[j + l][i + k]:
                                searching = False
                            else:
                                k = k + 1
                        else:
                            k = k + 1
                    l = l + 1
                if searching:
                    pointlist.append([i, j])
                    print([i, j, filteredimage[j][i][0], filteredimage[j][i][1]])
    if noisy:
        spotimage = numpy.zeros((combinedimage.shape[0], combinedimage.shape[1], 3))
        spotimage[:, :, 0] = normalizedimage
        spotimage[:, :, 1] = normalizedimage
        spotimage[:, :, 2] = normalizedimage
        for point in pointlist:
            spotimage[point[1]][point[0]][0] = 1
            spotimage[point[1]][point[0]][1] = 0
            spotimage[point[1]][point[0]][2] = 0
        mpl.figure()
        mpl.imshow(spotimage)
    return pointlist

def eyefilter(image, radius, noisy = False):
    eyefilter = numpy.zeros((int(2 * radius + 1), int(2 * radius + 1), 2))
    for l in range(-1 * radius, radius + 1):
        for k in range(-1 * int(numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5), int(numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5) + 1):
            eyefilter[l][k][0] = numpy.exp(-1 * (l**2 + k**2) / (2 * (radius / 3)**2))
            eyefilter[l][k][1] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2))
    if noisy:
        eyeimage = numpy.zeros((eyefilter.shape[0], eyefilter.shape[1], 3))
        eyeimage[:, :, 0] = eyefilter[:, :, 0]
        eyeimage[:, :, 2] = eyefilter[:, :, 1]
        mpl.figure()
        mpl.imshow(eyeimage)
    filteredimage = numpy.zeros((image.shape[0], image.shape[1], 2))
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            pupilscore = 0
            irisscore = 0
            pupilweight = 0
            irisweight = 0
            for l in range(-1 * radius, radius + 1):
                for k in range(-1 * radius, radius + 1):
                    if i + k >= 0 and i + k < image.shape[1] and j + l >= 0 and j + l < image.shape[0]:
                        pupilscore = pupilscore + image[j + l][i + k] * eyefilter[l][k][0]
                        irisscore = irisscore + image[j + l][i + k] * eyefilter[l][k][1]
                        pupilweight = pupilweight + eyefilter[l][k][0]
                        irisweight = irisweight + eyefilter[l][k][1]
            filteredimage[j][i][0] = pupilscore / pupilweight
            filteredimage[j][i][1] = irisscore / irisweight
    return filteredimage

def buddhistfilter(image, radius, noisy = False):
    budfilter = numpy.zeros((int(2 * radius + 1), int(2 * radius + 1), 9))
    for l in range(-1 * radius, radius + 1):
        for k in range(-1 * int(numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5), int(numpy.round(radius**2 - (numpy.abs(l) - .5)**2)**.5) + 1):
            budfilter[l][k][0] = numpy.exp(-1 * (l**2 + k**2) / (2 * (radius / 3)**2))
            budfilter[l][k][1] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * ((k / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][2] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * (((k * 2**-.5 + l * 2**-.5) / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][3] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * ((l / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][4] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * (((k * -2**-.5 + l * 2**-.5) / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][5] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * ((-1 * k / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][6] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * (((k * -2**-.5 + l * -2**-.5) / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][7] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * ((-1 * l / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
            budfilter[l][k][8] = numpy.exp(-1 * ((l**2 + k**2)**.5 - radius)**2 / (2 * (radius / 3)**2)) * numpy.exp(-1 * (((k * 2**-.5 + l * -2**-.5) / (k**2 + l**2)**.5) - 1)**2 / (2 * .1**2))
    budweight = numpy.array([numpy.sum(budfilter[:, :, 0]), numpy.sum(budfilter[:, :, 1]), numpy.sum(budfilter[:, :, 2]), numpy.sum(budfilter[:, :, 3]), numpy.sum(budfilter[:, :, 4]), numpy.sum(budfilter[:, :, 5]), numpy.sum(budfilter[:, :, 6]), numpy.sum(budfilter[:, :, 7]), numpy.sum(budfilter[:, :, 8])])
    if noisy:
        budimage = numpy.zeros((budfilter.shape[0], budfilter.shape[1], 3))
        budimage[:, :, 0] = budfilter[:, :, 0] + budfilter[:, :, 1] + .6 * budfilter[:, :, 2] + .3 * budfilter[:, :, 3] + .3 * budfilter[:, :, 7] + .6 * budfilter[:, :, 8]
        budimage[:, :, 0] = budfilter[:, :, 0] + .3 * budfilter[:, :, 2] + .6 * budfilter[:, :, 3] + budfilter[:, :, 4] + .5 * budfilter[:, :, 5]
        budimage[:, :, 0] = budfilter[:, :, 0] + .5 * budfilter[:, :, 5] + budfilter[:, :, 6] + .6 * budfilter[:, :, 7] + .3 * budfilter[:, :, 8]
        mpl.figure()
        mpl.imshow(budimage / numpy.max(budimage))
    filteredimage = numpy.zeros((image.shape[0], image.shape[1], 9))
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            scorearray = numpy.zeros((9))
            lessweight = numpy.zeros((9))
            for l in range(-1 * radius, radius + 1):
                for k in range(-1 * radius, radius + 1):
                    if i + k >= 0 and i + k < image.shape[1] and j + l >= 0 and j + l < image.shape[0]:
                        scorearray = scorearray + image[j + l][i + k] * budfilter[l][k]
                    else:
                        lessweight = lessweight + budfilter[l][k]
            filteredimage[j][i] = scorearray / (budweight - lessweight)
    return filteredimage

def characterizenoise(image, noisy = False):
    #xdisplacement = numpy.abs(image[:, 1:image.shape[1]] - image[:, 0:(image.shape[1] - 1)])
    #ydisplacement = numpy.abs(image[1:image.shape[0], :] - image[0:(image.shape[0] - 1), :])
    averagechange = 0
    for j in range(image.shape[0] - 1):
        for i in range(image.shape[1] - 1):
            averagechange = averagechange + numpy.abs(image[j][i + 1] - image[j][i]) + numpy.abs(image[j + 1][i] - image[j][i])
    #averagechange = (numpy.sum(xdisplacement) + numpy.sum(ydisplacement)) / (xdisplacement.shape[0] * xdisplacement.shape[1] + ydisplacement.shape[0] * ydisplacement.shape[1])
    averagechange = averagechange / (2 * (image.shape[0] - 1) * (image.shape[1] - 1))
    if noisy:
        contourimage = numpy.zeros(image.shape)
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                contourimage[j][i] = numpy.floor((image[j][i] - numpy.min(image)) / averagechange)
        mpl.figure()
        mpl.imshow(image)
        mpl.figure()
        mpl.imshow(contourimage)
    return averagechange

def resize(image, dimensions):
    xfactor = image.shape[1] / dimensions[0]
    yfactor = image.shape[0] / dimensions[1]
    newimage = numpy.zeros(dimensions)
    countimage = numpy.zeros(dimensions)
    for j in range(dimensions[1]):
        for i in range(dimensions[0]):
            origx = int(numpy.floor(i * xfactor))
            origy = int(numpy.floor(j * yfactor))
            for l in range(int(numpy.ceil(xfactor))):
                for k in range(int(numpy.ceil(yfactor))):
                    newimage[j][i] = newimage[j][i] + image[origy + l][origx + k]
                    countimage[j][i] = countimage[j][i] + 1
    newimage = newimage / countimage
    return newimage

if __name__ == '__main__':
# =============================================================================
#     testpointvsnoise(height = .15 * numpy.random.rand() + .1)
#     testpointvsnoise(height = .15 * numpy.random.rand() + .1)
#     testpointvsnoise(height = .15 * numpy.random.rand() + .1)
#     testpointvsnoise(height = .15 * numpy.random.rand() + .1)
#     testpointvsnoise(height = .15 * numpy.random.rand() + .1)
# =============================================================================
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    orgimg = numpy.genfromtxt(flname, delimiter = ',')
    #comprd = fftcompression.findcompressionradius(orgimg)
    comprd = int(numpy.ceil(fftcompression.findcharacteristicperiod(orgimg, mode = 'average', feedback = True) / 2))
# =============================================================================
#     orgfft = numpy.fft.fft2(orgimg)
#     nrmfft = numpy.log10((numpy.real(orgfft)**2 + numpy.imag(orgfft)**2)**.5)
#     revfft = numpy.zeros(nrmfft.shape)
#     revfft[0:int(revfft.shape[0] / 2), 0:int(revfft.shape[1] / 2)] = nrmfft[int(nrmfft.shape[0] / 2):nrmfft.shape[0], int(nrmfft.shape[1] / 2):nrmfft.shape[1]]
#     revfft[int(revfft.shape[0] / 2):revfft.shape[0], 0:int(revfft.shape[1] / 2)] = nrmfft[0:int(nrmfft.shape[0] / 2), int(nrmfft.shape[1] / 2):nrmfft.shape[1]]
#     revfft[0:int(revfft.shape[0] / 2), int(revfft.shape[1] / 2):revfft.shape[1]] = nrmfft[int(nrmfft.shape[0] / 2):nrmfft.shape[0], 0:int(nrmfft.shape[1] / 2)]
#     revfft[int(revfft.shape[0] / 2):revfft.shape[0], int(revfft.shape[1] / 2):revfft.shape[1]] = nrmfft[0:int(nrmfft.shape[0] / 2), 0:int(nrmfft.shape[1] / 2)]
#     comprd = 6
#     iddimg = imageevaluation(revfft, comprd)
#     atomlocations = pointdetection(iddimg, comprd)
#     locatedimage = markpositions(revfft, atomlocations)
#     #numpy.savetxt(f'{fname} position evaluation {comprd} revised.csv', numpy.array(reevaluatepoints(orgimg, atomlocations, comprd)), delimiter = ',')
#     mpl.figure()
#     mpl.imshow(revfft)
#     mpl.figure()
#     mpl.imshow(visualizecategoryimage(iddimg))
#     mpl.figure()
#     mpl.imshow(locatedimage)
# =============================================================================
    #spotatoms(orgimg, comprd, noisy = True)
    iddimg = imageevaluation(orgimg, comprd)
    atmpos = pointdetection(iddimg, comprd)
    #print(characterizenoise(orgimg, noisy = True))
    locatedimage = markpositions(orgimg, atmpos)
    print(comprd)
    mpl.figure()
    mpl.imshow(locatedimage)
    #mpl.imshow(eyefilter(orgimg, 5))
    #imageio.imwrite(f'{fname} located {comprd}.png', locatedimage)
    #imageio.imwrite(f'{fname} original.png', (orgimg - numpy.min(orgimg)) / (numpy.max(orgimg) - numpy.min(orgimg)))
# =============================================================================
#     fig, ax = mpl.subplots(1, 3)
#     ax[0].imshow(viwwin)
#     ax[1].imshow(compix[:, :, 0])
#     if categ[0] == 0:
#         ax[2].imshow(catega)
#     elif categ[0] == 1:
#         ax[2].imshow(categb[categ[1]])
#     elif categ[0] == 2:
#         ax[2].imshow(categc[categ[1]])
#     elif categ[0] == 3:
#         ax[2].imshow(categd[categ[1]])
#     else:
#         ax[2].imshow(catege)
# =============================================================================