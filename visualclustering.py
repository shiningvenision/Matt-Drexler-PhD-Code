# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:12:24 2021

@author: shini
"""

import numpy
import time
import matplotlib.pyplot as mpl

def densitymap(data, size):
    themap = numpy.zeros(size)
    bounds = numpy.zeros((data.shape[1], 2))
    for point in data:
        for i in range(data.shape[1]):
            if point[i] < bounds[i][0]:
                bounds[i][0] = point[i]
            if point[i] > bounds[i][1]:
                bounds[i][1] = point[i]
    boxsizes = (bounds[:, 1] - bounds[:, 0]) / size
    for point in data:
        thebox = themap[int(numpy.min([(point[0] - bounds[0][0]) / boxsizes[0], size[0] - 1]))]
        for i in range(1, data.shape[1] - 1):
            thebox = thebox[int(numpy.min([(point[i] - bounds[i][0]) / boxsizes[i], size[i] - 1]))]
        thebox[int(numpy.min([(point[-1] - bounds[-1][0]) / boxsizes[-1], size[-1] - 1]))] = thebox[int(numpy.min([(point[-1] - bounds[-1][0]) / boxsizes[-1], size[-1] - 1]))] + 1
    mpl.figure()
    mpl.imshow(themap)
    return themap

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

def correlatederivatives(gradient, error = 1):
    averagederivative = numpy.sum((gradient[:, :, 0]**2 + gradient[:, :, 1]**2)**.5) / (gradient.shape[0] * gradient.shape[1])
    stddevderivative = 0
    for j in range(gradient.shape[0]):
        for i in range(gradient.shape[1]):
            stddevderivative = stddevderivative + ((gradient[j][i][0]**2 + gradient[j][i][1]**2)**.5 - averagederivative)**2
    stddevderivative = (stddevderivative / (gradient.shape[0] * gradient.shape[1] - 1))**.5
    notableedges = []
    for j in range(gradient.shape[0]):
        for i in range(gradient.shape[1]):
            if (gradient[j][i][0]**2 + gradient[j][i][1]**2)**.5 > averagederivative + stddevderivative:
                notableedges.append([i, j])
    clusterlist = []
    while len(notableedges) > 0:
        currentcluster = [notableedges.pop(0)]
        m = 0
        while m < len(currentcluster):
            n = 0
            while n < len(notableedges):
                if gradient[currentcluster[m][1]][currentcluster[m][0]][0] * gradient[notableedges[n][1]][notableedges[n][0]][0] + gradient[currentcluster[m][1]][currentcluster[m][0]][1] * gradient[notableedges[n][1]][notableedges[n][0]][1] > 0 and ((currentcluster[m][0] - notableedges[n][0])**2 + (currentcluster[m][1] - notableedges[n][1])**2)**.5 <= error:
                    currentcluster.append(notableedges.pop(n))
                else:
                    n = n + 1
            m = m + 1
        clusterlist.append(currentcluster)
    displaymap = numpy.zeros((gradient.shape[0], gradient.shape[1], 3))
    colorlist = numpy.random.rand(len(clusterlist), 3)
    for cluster in range(len(clusterlist)):
        for pixel in clusterlist[cluster]:
                displaymap[pixel[1]][pixel[0]][0] = colorlist[cluster][0]
                displaymap[pixel[1]][pixel[0]][1] = colorlist[cluster][1]
                displaymap[pixel[1]][pixel[0]][2] = colorlist[cluster][2]
    mpl.figure()
    mpl.imshow(displaymap)

def visualclustering(data, size, error = 1, verbose = False):
    densityimage = numpy.zeros(size)
    bounds = numpy.zeros((data.shape[1], 2))
    for point in data:
        for i in range(data.shape[1]):
            if point[i] < bounds[i][0]:
                bounds[i][0] = point[i]
            if point[i] > bounds[i][1]:
                bounds[i][1] = point[i]
    boxsizes = (bounds[:, 1] - bounds[:, 0]) / size
    for point in data:
        thebox = [int(numpy.min([(point[0] - bounds[0][0]) / boxsizes[0], size[0] - 1])), int(numpy.min([(point[1] - bounds[1][0]) / boxsizes[1], size[1] - 1]))]
        densityimage[thebox[1]][thebox[0]] = densityimage[thebox[1]][thebox[0]] + 1
    if verbose:
        mpl.figure()
        mpl.imshow(densityimage)
    gradient = sobeloperator(densityimage, noisy = verbose)
    averagederivative = numpy.sum((gradient[:, :, 0]**2 + gradient[:, :, 1]**2)**.5) / (gradient.shape[0] * gradient.shape[1])
    stddevderivative = 0
    for j in range(gradient.shape[0]):
        for i in range(gradient.shape[1]):
            stddevderivative = stddevderivative + ((gradient[j][i][0]**2 + gradient[j][i][1]**2)**.5 - averagederivative)**2
    stddevderivative = (stddevderivative / (gradient.shape[0] * gradient.shape[1] - 1))**.5
    notableedges = []
    for j in range(gradient.shape[0]):
        for i in range(gradient.shape[1]):
            if (gradient[j][i][0]**2 + gradient[j][i][1]**2)**.5 > averagederivative + stddevderivative:
                notableedges.append([i, j])
    clusterlist = []
    while len(notableedges) > 0:
        currentcluster = [notableedges.pop(0)]
        m = 0
        while m < len(currentcluster):
            n = 0
            while n < len(notableedges):
                if gradient[currentcluster[m][1]][currentcluster[m][0]][0] * gradient[notableedges[n][1]][notableedges[n][0]][0] + gradient[currentcluster[m][1]][currentcluster[m][0]][1] * gradient[notableedges[n][1]][notableedges[n][0]][1] > 0 and ((currentcluster[m][0] - notableedges[n][0])**2 + (currentcluster[m][1] - notableedges[n][1])**2)**.5 <= error:
                    currentcluster.append(notableedges.pop(n))
                else:
                    n = n + 1
            m = m + 1
        clusterlist.append(currentcluster)
    centerlist = []
    plotlists = []
    for m in range(len(clusterlist)):
        plotlists.append([])
        for box in clusterlist[m]:
            centerlist.append([bounds[0][0] + boxsizes[0] * (box[0] + .5), bounds[1][0] + boxsizes[1] * (box[1] + .5), m])
    labellist = []
    for point in data:
        label = -1
        lowdistance = numpy.sum((bounds[:, 1] - bounds[:, 0])**2)**.5
        for center in centerlist:
            if ((point[0] - center[0])**2 + (point[1] - center[1])**2)**.5 < lowdistance:
                lowdistance = ((point[0] - center[0])**2 + (point[1] - center[1])**2)**.5
                label = center[2]
        labellist.append(label)
    if verbose:
        for n in range(data.shape[0]):
            plotlists[labellist[n]].append(data[n])
        mpl.figure()
        for m in range(len(plotlists)):
            if len(plotlists[m]) > 0:
                outarray = numpy.array(plotlists[m])
                mpl.scatter(outarray[:, 0], outarray[:, 1])

if __name__ == "__main__":
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    clstda = numpy.genfromtxt(flname, delimiter = ',')
    #smallimage = densitymap(clstda, (32, 32))
    #smallgradient = sobeloperator(smallimage)
    #correlatederivatives(smallgradient, error = 1.5)
    start = time.time()
    visualclustering(clstda, (32, 32), error = 1.5, verbose = True)
    end = time.time()
    print(f'Clustering took {end - start} s')