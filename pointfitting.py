# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:27:42 2019

@author: shini
"""

import numpy
import imageio
import matplotlib.pyplot as mpl
import scipy.optimize as sciopt
import scipy.interpolate as interp
import levelgraph
import sandwhichsmooth

class densidist:
    
    def __init__(self, numdiv, bounds):
        # Note: this is not for n-dimensional data. This is for nxm parameter 
        # estimation
        self.size = (bounds.shape[0], bounds.shape[1], numdiv)
        self.prbmap = numpy.ones(self.size) / numdiv
        self.valmap = numpy.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.valmap[i][j] = numpy.linspace(bounds[i][j][0], bounds[i][j][1], num = numdiv)
    
    def drawsamp(self):
        rndnum = numpy.random.random((self.size[0], self.size[1]))
        sample = numpy.zeros((self.size[0], self.size[1]))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                k = -1
                while rndnum[i][j] > 0:
                    k = k + 1
                    rndnum[i][j] = rndnum[i][j] - self.prbmap[i][j][k]
                sample[i][j] = self.valmap[i][j][k]
        return sample
    
    def estimate(self, indata):
        # indata will be a list with the first axis representing each point 
        # and the second axis representing data points from the samples drawn
        # earlier. This may not be a rectangular array, so don't write it to 
        # assume so
        #self.prbmap = numpy.ones(self.size)
        stparr = self.valmap[:, :, 1] - self.valmap[:, :, 0]
        for i in range(self.size[0]):
            for n in range(len(indata[i])):
                indarr = (indata[i][n] - self.valmap[i, :, 0]) / stparr[i]
                for j in range(self.size[1]):
                    self.prbmap[i][j][int(indarr[j])] = self.prbmap[i][j][int(indarr[j])] + (1 - n / len(indata[i]))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.prbmap[i][j] = self.prbmap[i][j] / numpy.sum(self.prbmap[i][j])
    
    def getavg(self):
        avgarr = numpy.zeros((self.size[0], self.size[1]))
        for k in range(self.size[2]):
            avgarr = avgarr + self.prbmap[:, :, k] * self.valmap[:, :, k]
        return avgarr
    
    def readprob(self):
        for i in range(self.size[0]):
            print(f'i: {i}')
            for j in range(self.size[1]):
                print(f'v: {self.valmap[i][j]}')
                print(f'p: {self.prbmap[i][j]}')

def emforsize(haught, tstnum = 30):
    # This will estimate the maximum probability distribution using the 
    # expectation maximization algorithm outlined in that book referenced in 
    # that paper. This will need to iteratively test higher and higher numbers 
    # of distributions until a good answer can be found.
    stpnum = int(numpy.ceil(tstnum / 2))
    scores = []
    profis = []
    # First, perform the expectation maximization procedure for half the 
    # evaluation range
    for i in range(stpnum):
        # Start by initializing the parameters. There should be 3 per (i + 1)
        params = numpy.ones((3, i + 1))
        # The alphas (row one) should be equal, presumably
        params[0] = params[0] / (i + 1)
        # The averages (row two) should divide the sample data into equal 
        # parts
        stpsze = (numpy.max(haught) - numpy.min(haught)) / (i + 2)
        for j in range(i + 1):
            params[1][j] = numpy.min(haught) + (j + 1) * stpsze
        # The spreads (row three) should all be the same, and they should be 
        # such that the distributions have 1% of their max probability halfway 
        # between the averages
        params[2] = params[2] * ((stpsze / 2)**2 / (-2 * numpy.log(.01)))**.5
        # Initialize the probability and belonging matrix
        belong = numpy.ones((haught.shape[0], i + 1))
        probas = numpy.ones((haught.shape[0], i + 1))
        for j in range(probas.shape[0]):
            for k in range(probas.shape[1]):
                probas[j][k] = params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
        # Set up for optimization by calculating the log-likelihood
        #print(params)
        #print(probas)
        loglik = 0
        for height in haught:
            smpsum = 0
            for j in range(params.shape[1]):
                smpsum = smpsum + params[0][j] * numpy.exp(-1 * (height - params[1][j])**2 / (2 * params[2][j]**2)) / (params[2][j] * 2**.5 * numpy.pi**.5)
            loglik = loglik + numpy.log10(smpsum)
        preill = 2 * loglik
        while numpy.abs((preill - loglik) / preill) > .005:
            #print(loglik)
            # Calculate membership probabilities
            for j in range(belong.shape[0]):
                for k in range(belong.shape[1]):
                    belong[j][k] = probas[j][k] / numpy.sum(probas[j])
            # Calculate the new alphas
            for j in range(params.shape[1]):
                params[0][j] = numpy.sum(belong[:, j]) / belong.shape[0]
            # Calculate the new medians
            params[1] = numpy.zeros((params.shape[1]))
            for j in range(params.shape[1]):
                for k in range(belong.shape[0]):
                    params[1][j] = params[1][j] + belong[k][j] * haught[k]
                params[1][j] = params[1][j] / numpy.sum(belong[:, j])
            # Calculate the new spreads
            params[2] = numpy.zeros((params.shape[1]))
            for j in range(params.shape[1]):
                for k in range(belong.shape[0]):
                    params[2][j] = params[2][j] + belong[k][j] * (haught[k] - params[1][j])**2
                    #sigmas = sigmas + belong[k][j] * (haught[k] - params[1][j])**2
                if params[2][j] == 0:
                    params[2][j] = 10**-8 # I wonder if making this absolute rather than relative will come back to haunt me
                else:
                    params[2][j] = (params[2][j] / numpy.sum(belong[:, j]))**.5
            sigmas = numpy.sum(params[2])
            params[2] = numpy.ones((params.shape[1])) * sigmas / params.shape[1]
            # Calculate the log-likelihood
            preill = loglik
            loglik = 0
            for j in range(haught.shape[0]):
                smpsum = 0
                for k in range(params.shape[1]):
                    probas[j][k] = params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
                    smpsum = smpsum + params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
                loglik = loglik + numpy.log10(smpsum)
        # Record results
        entrop = 0
        for j in range(belong.shape[0]):
            for k in range(belong.shape[1]):
                if belong[j][k] != 0:
                    entrop = entrop + belong[j][k] * numpy.log10(belong[j][k])
        scores.append(-2 * loglik - 2 * entrop + 2 * (i + 1) * numpy.log10(haught.shape[0]))
        profis.append(params)
    # Next, repeat this process for the next higher half-range and evaluate 
    # whether that full range contains the appropriate value. If so, terminate 
    # and return the results, otherwise, repeat
    stlsch = True
    evlind = 0
    while stlsch:
        for i in range((evlind + 1) * stpnum, (evlind + 2) * stpnum):
            # Start by initializing the parameters. There should be 3 per (i + 1)
            params = numpy.ones((3, i + 1))
            # The alphas (row one) should be equal, presumably
            params[0] = params[0] / (i + 1)
            # The averages (row two) should divide the sample data into equal 
            # parts
            stpsze = (numpy.max(haught) - numpy.min(haught)) / (i + 2)
            for j in range(i + 1):
                params[1][j] = numpy.min(haught) + (j + 1) * stpsze
            # The spreads (row three) should all be the same, and they should be 
            # such that the distributions have 1% of their max probability halfway 
            # between the averages
            params[2] = params[2] * ((stpsze / 2)**2 / (-2 * numpy.log(.01)))**.5
            # Initialize the probability and belonging matrix
            belong = numpy.ones((haught.shape[0], i + 1))
            probas = numpy.ones((haught.shape[0], i + 1))
            for j in range(probas.shape[0]):
                for k in range(probas.shape[1]):
                    probas[j][k] = params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
            # Set up for optimization by calculating the log-likelihood
            #print(params)
            #print(probas)
            loglik = 0
            for height in haught:
                smpsum = 0
                for j in range(params.shape[1]):
                    smpsum = smpsum + params[0][j] * numpy.exp(-1 * (height - params[1][j])**2 / (2 * params[2][j]**2)) / (params[2][j] * 2**.5 * numpy.pi**.5)
                loglik = loglik + numpy.log10(smpsum)
            preill = 2 * loglik
            while numpy.abs((preill - loglik) / preill) > .005:
                #print(loglik)
                # Calculate membership probabilities
                for j in range(belong.shape[0]):
                    for k in range(belong.shape[1]):
                        belong[j][k] = probas[j][k] / numpy.sum(probas[j])
                # Calculate the new alphas
                for j in range(params.shape[1]):
                    params[0][j] = numpy.sum(belong[:, j]) / belong.shape[0]
                # Calculate the new medians
                params[1] = numpy.zeros((params.shape[1]))
                for j in range(params.shape[1]):
                    for k in range(belong.shape[0]):
                        params[1][j] = params[1][j] + belong[k][j] * haught[k]
                    params[1][j] = params[1][j] / numpy.sum(belong[:, j])
                # Calculate the new spreads
                params[2] = numpy.zeros((params.shape[1]))
                for j in range(params.shape[1]):
                    for k in range(belong.shape[0]):
                        params[2][j] = params[2][j] + belong[k][j] * (haught[k] - params[1][j])**2
                        #sigmas = sigmas + belong[k][j] * (haught[k] - params[1][j])**2
                    if params[2][j] == 0:
                        params[2][j] = 10**-8 # I wonder if making this absolute rather than relative will come back to haunt me
                    else:
                        params[2][j] = (params[2][j] / numpy.sum(belong[:, j]))**.5
                sigmas = numpy.sum(params[2])
                params[2] = numpy.ones((params.shape[1])) * sigmas / params.shape[1]
                # Calculate the log-likelihood
                preill = loglik
                loglik = 0
                for j in range(haught.shape[0]):
                    smpsum = 0
                    for k in range(params.shape[1]):
                        probas[j][k] = params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
                        smpsum = smpsum + params[0][k] * numpy.exp(-1 * (haught[j] - params[1][k])**2 / (2 * params[2][k]**2)) / (params[2][k] * 2**.5 * numpy.pi**.5)
                    loglik = loglik + numpy.log10(smpsum)
            # Record results
            entrop = 0
            for j in range(belong.shape[0]):
                for k in range(belong.shape[1]):
                    if belong[j][k] != 0:
                        entrop = entrop + belong[j][k] * numpy.log10(belong[j][k])
            scores.append(-2 * loglik - 2 * entrop + 2 * (i + 1) * numpy.log10(haught.shape[0]))
            profis.append(params)
        # Next, curve fit a parabola to the current set of scores to evaluate. 
        # If the parabola is curved upward and the apex is within the window, 
        # terminate the loop, otherwise, repeat
        tstvls = []
        for u in range(evlind * stpnum, (evlind + 2) * stpnum):
            tstvls.append(scores[u])
        paropt, parcov = sciopt.curve_fit(quadraticcurve, numpy.array(range(evlind * stpnum, (evlind + 2) * stpnum)), numpy.array(tstvls))
        print(paropt)
        if paropt[0] > 0 and (-1 * paropt[1] / (2 * paropt[0])) <= ((evlind + 2) * stpnum):
            stlsch = False
        else:
            evlind = evlind + 1
    # Once identifying the region of the curve in which the best fit falls, 
    # determine the best value by finding the lowest value from among the 
    # evaluation range
    bstscr = [evlind * stpnum, scores[evlind * stpnum]]
    for i in range(evlind * stpnum + 1, (evlind + 2) * stpnum):
        if scores[i] < bstscr[1]:
            bstscr[0] = i
            bstscr[1] = scores[i]
    binsze = (1.01 * numpy.max(haught) - numpy.min(haught)) / 100
    xscatt = numpy.zeros((100))
    functs = numpy.zeros((100, 3 + bstscr[0]))
    for i in range(xscatt.shape[0]):
        xscatt[i] = numpy.min(haught) + i * binsze
    for hght in haught:
        functs[int(numpy.floor((hght - numpy.min(haught)) / binsze))][0] = functs[int(numpy.floor((hght - numpy.min(haught)) / binsze))][0] + 1
    functs[:, 0] = functs[:, 0] / (haught.shape[0] * binsze)
    for i in range(xscatt.shape[0]):
        for j in range(profis[bstscr[0]].shape[1]):
            functs[i][1] = functs[i][1] + profis[bstscr[0]][0][j] * numpy.exp(-1 * (xscatt[i] - profis[bstscr[0]][1][j])**2 / (2 * profis[bstscr[0]][2][j]**2)) / (profis[bstscr[0]][2][j] * (2 * numpy.pi)**.5)
            functs[i][j + 2] = functs[i][j + 2] + profis[bstscr[0]][0][j] * numpy.exp(-1 * (xscatt[i] - profis[bstscr[0]][1][j])**2 / (2 * profis[bstscr[0]][2][j]**2)) / (profis[bstscr[0]][2][j] * (2 * numpy.pi)**.5)
    mpl.figure()
    mpl.scatter(xscatt, functs[:, 0])
    for i in range(1, functs.shape[1]):
        mpl.plot(xscatt, functs[:, i])
    mpl.figure()
    mpl.scatter(range(len(scores)), scores)
    print(profis[bstscr[0]])
    return profis[bstscr[0]][1]

def quadraticcurve(xdata, axtwo, bxone, cxzer):
    return axtwo * xdata**2 + bxone * xdata + cxzer

# =============================================================================
# def estatmsze(hghts, maxnum, trshld, itrlim = 500):
#     # This will estimate a maximum probability distribution from a list of 
#     # numbers where the distribution is a sum of evenly spaced Gaussian 
#     # curves
#     scores = numpy.zeros((maxnum))
#     profis = numpy.zeros((maxnum, 3))
#     for i in range(maxnum):
#         vertxs = []
#         baslin = numpy.min(hghts)
#         atmhgt = (numpy.max(hghts) - numpy.min(hghts)) / (i + 1)
#         #sigmas = atmhgt / 2
#         aponts = []
#         aponts.append(numpy.array([baslin, atmhgt * .5, (1.5 * atmhgt / (-2 * numpy.log(.01)))**.5]))
#         aponts.append(numpy.array([baslin, atmhgt * 1.5, (.5 * atmhgt / (-2 * numpy.log(.01)))**.5]))
#         aponts.append(numpy.array([baslin + atmhgt, atmhgt * 1.5, (1.5 * atmhgt / (-2 * numpy.log(.01)))**.5]))
#         aponts.append(numpy.array([baslin + atmhgt, atmhgt * .5, (.5 * atmhgt / (-2 * numpy.log(.01)))**.5]))
#         centrd = numpy.zeros((3))
#         for pont in aponts:
#             vscore = numpy.sum(multiheightdist(hghts, i, pont[0], pont[1], pont[2]))
#             centrd = centrd + pont
#             srchng = True
#             k = 0
#             while k < len(vertxs) and srchng:
#                 if vscore > vertxs[k][0]:
#                     k = k + 1
#                 else:
#                     srchng = False
#             if srchng:
#                 vertxs.append([vscore, pont])
#             else:
#                 vertxs.insert(k, [vscore, pont])
#         centrd = centrd / 4
#         simsig = 0
#         for j in range(4):
#             simsig = simsig + numpy.sum((vertxs[j][1] - centrd)**2)
#         simsig = (simsig / 3)**.5
#         limit = simsig * .05
#         itrate = 0
#         # Start optimization
#         while simsig > limit and itrate < itrlim:
#             itrate = itrate + 1 
#             # Calculate the relfected point
#             rvectr = vertxs[3][1] - centrd
#             fpoint = centrd - rvectr
#             if fpoint[1] <= 0:
#                 rvectr = rvectr * (centrd[1] - 10**-8) / (centrd[1] - fpoint[1])
#                 fpoint = centrd - rvectr
#             if fpoint[2] <= 0:
#                 rvectr = rvectr * (centrd[2] - 10**-8) / (centrd[2] - fpoint[2])
#                 fpoint = centrd - rvectr
#             rscore = numpy.sum(multiheightdist(hghts, i, fpoint[0], fpoint[1], fpoint[2]))
#             # Determine if the simplex should expand, contract, or shrink
#             j = 0
#             srchng = True
#             while j < len(vertxs) and srchng:
#                 if rscore > vertxs[j][0]:
#                     j = j + 1
#                 else:
#                     srchng = False
#             # If this is the best so far, consider expansion
#             if j == 0:
#                 evectr = 2 * (fpoint - centrd)
#                 epoint = centrd + evectr
#                 if epoint[1] <= 0:
#                     evectr = evectr * (centrd[1] - 10**-8) / (centrd[1] - epoint[1])
#                     epoint = centrd + evectr
#                 if epoint[2] <= 0:
#                     evectr = evectr * (centrd[2] - 10**-8) / (centrd[2] - epoint[2])
#                     epoint = centrd + evectr
#                 escore = numpy.sum(multiheightdist(hghts, i, epoint[0], epoint[1], epoint[2]))
#                 if escore <= rscore:
#                     vertxs.insert(j, [escore, epoint])
#                     vertxs.pop()
#                 else:
#                     vertxs.insert(j, [rscore, fpoint])
#                     vertxs.pop()
#             # If the point is better than the second worst point, replace the 
#             # worst point with it
#             elif j < 2:
#                 vertxs.insert(j, [rscore, fpoint])
#                 vertxs.pop()
#             # If the point is worse than the second worst, consider 
#             # contraction
#             elif j >= 2:
#                 cpoint = centrd + .5 * (vertxs[3][1] - centrd)
#                 cscore = numpy.sum(multiheightdist(hghts, i, cpoint[0], cpoint[1], cpoint[2]))
#                 # If the contracted point is better than the worst point, 
#                 # contract, otherwise, shrink
#                 if cscore <= vertxs[3][0]:
#                     k = 0
#                     srchng = True
#                     while k < len(vertxs) and srchng:
#                         if cscore > vertxs[k][0]:
#                             k = k + 1
#                         else:
#                             srchng = False
#                     if srchng:
#                         vertxs.append([cscore, cpoint])
#                     else:
#                         vertxs.insert(k, [cscore, cpoint])
#                     vertxs.pop()
#                 else:
#                     newpts = []
#                     for k in range(1, len(vertxs)):
#                         newpts.append(vertxs[0][1] + .5 * (vertxs[k][1] - vertxs[0][1]))
#                     vertxs = [vertxs[0]]
#                     for pnt in newpts:
#                         newscr = numpy.sum(multiheightdist(hghts, i, pnt[0], pnt[1], pnt[2]))
#                         k = 0
#                         srchng = True
#                         while k < len(vertxs) and srchng:
#                             if newscr > vertxs[k][0]:
#                                 k = k + 1
#                             else:
#                                 srchng = False
#                         if srchng:
#                             vertxs.append([newscr, pnt])
#                         else:
#                             vertxs.insert(k, [newscr, pnt])
#             # Compute the new centroid and determine if the algorithm should 
#             # terminate
#             centrd = numpy.zeros((3))
#             for vertex in vertxs:
#                 centrd = centrd + vertex[1]
#             centrd = centrd / 4
#             simsig = 0
#             for vertex in vertxs:
#                 simsig = simsig + numpy.sum((vertex[1] - centrd)**2)
#             simsig = (simsig / 3)**.5
#         if itrate >= itrlim:
#             print(f'Too many iterations')
#         scores[i] = vertxs[0][0]
#         profis[i][0] = vertxs[0][1][0]
#         profis[i][1] = vertxs[0][1][1]
#         profis[i][2] = vertxs[0][1][2]
#     print(profis)
#     print(scores)
#     binsze = (1.01 * numpy.max(hghts) - numpy.min(hghts)) / 50
#     xscatt = numpy.zeros((50))
#     functs = numpy.zeros((50, maxnum + 1))
#     for i in range(xscatt.shape[0]):
#         xscatt[i] = numpy.min(hghts) + i * binsze
#     for hght in hghts:
#         functs[int(numpy.floor((hght - numpy.min(hghts)) / binsze))][0] = functs[int(numpy.floor((hght - numpy.min(hghts)) / binsze))][0] + 1
#     for i in range(xscatt.shape[0]):
#         for j in range(1, functs.shape[1]):
#             for k in range(j):
#                 functs[i][j] = functs[i][j] + numpy.exp(-1 * (xscatt[i] - (profis[j - 1][0] + k * profis[j - 1][1]))**2 / (2 * profis[j - 1][2]**2)) / (profis[j - 1][2] * (2 * numpy.pi)**.5)
#     fig = mpl.figure()
#     ax = fig.add_subplot(111)
#     mpl.scatter(xscatt, functs[:, 0])
#     for i in range(1, functs.shape[1]):
#         mpl.plot(xscatt, functs[:, i])
#     ax.legend()
#     mpl.figure()
#     mpl.scatter(range(maxnum), scores)
# =============================================================================

def multiheightdist(height, number, lowest, spacng, width):
    score = 0
    for i in range(number + 1):
        score = score - numpy.exp(-1 * (height - (lowest + i * spacng))**2 / (2 * width**2)) / (width * (2 * numpy.pi)**.5)
    return score / (number + 1)

def multiheightderi(height, number, lowest, spacng, width):
    lowder = numpy.zeros(height.shape)
    spader = numpy.zeros(height.shape)
    widder = numpy.zeros(height.shape)
    for i in range(number + 1):
        lowder = lowder + (height - (lowest + i * spacng)) * numpy.exp(-1 * (height - (lowest + i * spacng))**2 / (2 * width**2)) / (width**3 * 2**.5 * numpy.pi**.5)
        spader = spader + i * (height - (lowest + i * spacng)) * numpy.exp(-1 * (height - (lowest + i * spacng))**2 / (2 * width**2)) / (width**3 * 2**.5 * numpy.pi**.5)
        widder = widder + (-1 * (width**2 * (2 * numpy.pi)**.5)**-1 + (height - (lowest + i * spacng))**2 * (width**4 * (2 * numpy.pi)**.5)**-1) * numpy.exp(-1 * (height - (lowest + i * spacng))**2 / (2 * width**2))
    return -1 * numpy.array([numpy.sum(lowder), numpy.sum(spader), numpy.sum(widder)]) / (number + 1)

def fitgausses(imgdat, initgs, window):
    trshld = .01 #* (numpy.max(imgdat) - numpy.min(imgdat))
    # First, create initial value windows to create samples from for each data 
    # point. smlarr is a three dimensional array, the first axis representing 
    # each point, the second representing the average, low bound, and high 
    # bound for each point, and the third representing the five parameters 
    # that define a point
    # In 2.1, lowest represents a catalog of all scores lower than the current 
    # threshold, tracked by cutoff, such that a new cutoff can be calculated 
    # by finding the top xth percentile of the data
    smlarr = numpy.zeros((len(initgs), 5, 3))
    initwg = int(numpy.round((imgdat.shape[0] * imgdat.shape[1] / len(initgs))**.5 / 2))
    todrop = .75 #v2.0
    lowest = [] #v2.0
    cutoff = [] #v2.1
    tokeep = [] #v2.1
    minsmp = 25
    for i in range(len(initgs)):
        nearst = [int(numpy.round(initgs[i][0])), int(numpy.round(initgs[i][1]))]
        if nearst[0] < 0:
            nearst[0] = 0
        elif nearst[0] >= imgdat.shape[1]:
            nearst[0] = imgdat.shape[1] - 1
        if nearst[1] < 0:
            nearst[1] = 0
        elif nearst[1] >= imgdat.shape[0]:
            nearst[1] = imgdat.shape[0] - 1
        hgtest = imgdat[nearst[1]][nearst[0]]
        cnt = 1
        if nearst[0] > 0:
            hgtest = hgtest + imgdat[nearst[1]][nearst[0] - 1]
            cnt = cnt + 1
        if nearst[1] > 0:
            hgtest = hgtest + imgdat[nearst[1] - 1][nearst[0]]
            cnt = cnt + 1
        if nearst[0] < imgdat.shape[1] - 1:
            hgtest = hgtest + imgdat[nearst[1]][nearst[0] + 1]
            cnt = cnt + 1
        if nearst[1] < imgdat.shape[0] - 1:
            hgtest = hgtest + imgdat[nearst[1] + 1][nearst[0]]
            cnt = cnt + 1
        smlarr[i][0][0] = hgtest / cnt - numpy.min(imgdat)
        smlarr[i][0][1] = 0
        smlarr[i][0][2] = numpy.max(imgdat) - numpy.min(imgdat)
        smlarr[i][1][0] = nearst[0]
        smlarr[i][1][1] = nearst[0] - initwg
        smlarr[i][1][2] = nearst[0] + initwg
        smlarr[i][2][0] = nearst[1]
        smlarr[i][2][1] = nearst[1] - initwg
        smlarr[i][2][2] = nearst[1] + initwg
        smlarr[i][3][0] = -1 * initwg**2 / (2 * numpy.log(trshld))
        smlarr[i][3][1] = 1
        smlarr[i][3][2] = -1 * (numpy.min(window))**2 / (2 * numpy.log(trshld))
        smlarr[i][4][0] = numpy.min(imgdat)
        smlarr[i][4][1] = numpy.min(imgdat)
        smlarr[i][4][2] = (numpy.max(imgdat) + numpy.min(imgdat)) / 2
        lowest.append([]) #v2.0
        cutoff.append(numpy.sum(imgdat**2)) #v2.1
        tokeep.append([]) #v2.1
    # Next, generate samples and evaluate those samples. The evaluation will 
    # be done point-by-point to prevent small regions of bad fits from 
    # influencing regions of good fits
    mimdst = densidist(5, smlarr[:, :, [1, 2]]) #v2.1
    smpnum = 200
    rndsmp = numpy.zeros((smpnum, len(initgs), 5)) #v1.0-2.0
    smpwht = numpy.zeros((smpnum, len(initgs))) #v1.0-2.0
    itratn = True
    m = 0
    while itratn:
    #for m in range(50):
        #print(smlarr[:, 1, :])
        #print(smlarr[:, 2, :])
        for i in range(len(lowest)):
            lowest[i] = []
            tokeep[i] = [] #v2.1
        for i in range(smpnum):
            #rndsmp[i] = numpy.random.random((len(initgs), 5)) * (smlarr[:, :, 2] - smlarr[:, :, 1]) + smlarr[:, :, 1] v1.0-2.0
            rndsmp[i] = mimdst.drawsamp() #v2.1
            replic = interpimg(imgdat.shape, [rndsmp[i, :, 1], rndsmp[i, :, 2], rndsmp[i, :, 4]])
            for pnt in rndsmp[i]:
                nearst = [int(numpy.round(pnt[1])), int(numpy.round(pnt[2]))]
                if nearst[0] < 0:
                    nearst[0] = 0
                elif nearst[0] > imgdat.shape[1] - 1:
                    nearst[0] = imgdat.shape[1] - 1
                if nearst[1] < 0:
                    nearst[1] = 0
                elif nearst[1] > imgdat.shape[0] - 1:
                    nearst[1] = imgdat.shape[0] - 1
                replic[nearst[1]][nearst[0]] = replic[nearst[1]][nearst[0]] + pnt[0] * numpy.exp(-1 * ((nearst[0] - pnt[1])**2 + (nearst[1] - pnt[2])**2) / (2 * pnt[3]))
                nxtarr = []
                if nearst[0] > 0:
                    nxtarr.append([nearst[0] - 1, nearst[1]])
                if nearst[0] < imgdat.shape[1] - 1:
                    nxtarr.append([nearst[0] + 1, nearst[1]])
                if nearst[1] > 0:
                    nxtarr.append([nearst[0], nearst[1] - 1])
                if nearst[1] < imgdat.shape[0] - 1:
                    nxtarr.append([nearst[0], nearst[1] + 1])
                while len(nxtarr) > 0:
                    curpix = nxtarr.pop(0)
                    replic[curpix[1]][curpix[0]] = replic[curpix[1]][curpix[0]] + (pnt[0] * numpy.exp(-1 * ((curpix[0] - pnt[1])**2 + (curpix[1] - pnt[2])**2) / (2 * pnt[3])))
                    if numpy.exp(-1 * ((curpix[0] - pnt[1])**2 + (curpix[1] - pnt[2])**2) / (2 * pnt[3])) > trshld:
                        if curpix[0] > nearst[0] and curpix[1] <= nearst[1]:
                            if curpix[1] > 0:
                                nxtarr.append([curpix[0], curpix[1] - 1])
                            if curpix[1] == nearst[1] and curpix[0] < imgdat.shape[1] - 1:
                                nxtarr.append([curpix[0] + 1, curpix[1]])
                        if curpix[0] <= nearst[0] and curpix[1] < nearst[1]:
                            if curpix[0] > 0:
                                nxtarr.append([curpix[0] - 1, curpix[1]])
                            if curpix[0] == nearst[0] and curpix[1] > 0:
                                nxtarr.append([curpix[0], curpix[1] - 1])
                        if curpix[0] < nearst[0] and curpix[1] >= nearst[1]:
                            if curpix[1] < imgdat.shape[0] - 1:
                                nxtarr.append([curpix[0], curpix[1] + 1])
                            if curpix[1] == nearst[1] and curpix[0] > 0:
                                nxtarr.append([curpix[0] - 1, curpix[1]])
                        if curpix[0] >= nearst[0] and curpix[1] > nearst[1]:
                            if curpix[0] < imgdat.shape[1] - 1:
                                nxtarr.append([curpix[0] + 1, curpix[1]])
                            if curpix[0] == nearst[0] and curpix[1] < imgdat.shape[0] - 1:
                                nxtarr.append([curpix[0], curpix[1] + 1])
            errimg = (imgdat - replic)**2
            for j in range(len(rndsmp[i])):
                nearst = [int(numpy.round(rndsmp[i][j][1])), int(numpy.round(rndsmp[i][j][2]))]
                if nearst[0] < 0:
                    nearst[0] = 0
                elif nearst[0] > errimg.shape[1] - 1:
                    nearst[0] = errimg.shape[1] - 1
                if nearst[1] < 0:
                    nearst[1] = 0
                elif nearst[1] > errimg.shape[0] - 1:
                    nearst[1] = errimg.shape[0] - 1
                xrange = [numpy.max((0, int(nearst[0] - numpy.floor(window[0] / 2)))), numpy.min((errimg.shape[1] - 1, int(nearst[0] + numpy.ceil(window[0] / 2))))]
                yrange = [numpy.max((0, int(nearst[1] - numpy.floor(window[1] / 2)))), numpy.min((errimg.shape[0] - 1, int(nearst[1] + numpy.ceil(window[1] / 2))))]
                #smpwht[i][j] = 1 / numpy.sum(errimg[yrange[0]:yrange[1], xrange[0]:xrange[1]]) #v1.0-v2.0
                # Error needs to be per pixel, otherwise truncated areas will 
                # dominate
                smpwht[i][j] = numpy.sum(errimg[yrange[0]:yrange[1], xrange[0]:xrange[1]]) / ((xrange[1] - xrange[0]) * (yrange[1] - yrange[0])) #v2.1
                if smpwht[i][j] <= cutoff[j]: #v2.1
                    if len(lowest[j]) == 0: #v2.0
                        lowest[j].append(smpwht[i][j]) #v2.0
                #elif smpwht[i][j] >= lowest[j][-1]: #v2.0
                    #if len(lowest[j]) < todrop: #v2.0
                        #lowest[j].append(smpwht[i][j]) #v2.0
                    else: #v2.1
                        chcknd = len(lowest[j]) - 1 #v2.1
                        stpsze = len(lowest[j]) / 2 #v2.1
                        while stpsze > 0: #v2.1
                            if smpwht[i][j] < lowest[j][chcknd - int(stpsze)]: #v2.1
                                chcknd = chcknd - int(stpsze) #v2.1
                                if stpsze > 1: #v2.1
                                    stpsze = numpy.ceil(stpsze) #v2.1
                            stpsze = int(stpsze) / 2 #v2.1
                        if smpwht[i][j] > lowest[j][chcknd]: #v2.1
                            lowest[j].insert(chcknd + 1, smpwht[i][j]) #v2.1
                        else: #v2.1
                            lowest[j].insert(chcknd, smpwht[i][j]) #v2.1
                    tokeep[j].append(rndsmp[i][j]) #v2.1
                #else: #v2.0
                    #srchng = True #v2.0
                    #k = 0 #v2.0
                    #while k < len(lowest[j]) and srchng: #v2.0
                        #if smpwht[i][j] <= lowest[j][k]: #v2.0
                            #lowest[j].insert(k, smpwht[i][j]) #v2.0
                            #srchng = False #v2.0
                        #else: #v2.0
                            #k = k + 1 #v2.0
                    #if srchng: #v2.0
                        #lowest[j].append(smpwht[i][j]) #v2.0
                    #if len(lowest[j]) > todrop: #v2.0
                        #lowest[j].pop() #v2.0
        #print(rndsmp[:, 0, :])
        # Next, compute the new average and window for each point
        mimdst.estimate(tokeep)
        itratn = False
        print(m)
        for j in range(rndsmp.shape[1]):
            #print(len(lowest[j]))
            if len(lowest[j]) > minsmp: #v2.1
                cutnum = int(numpy.floor(todrop * len(lowest[j]))) #v2.1
                cutoff[j] = lowest[j][cutnum] #v2.1
                itratn = True
            #newavg = numpy.zeros((5))
            #newmin = numpy.copy(rndsmp[0, j, :]) #v2.0
            #newmax = numpy.copy(rndsmp[0, j, :]) #v2.0
            #totwht = 0
            #for i in range(smpnum):
                #if smpwht[i][j] >= lowest[j][-1]: #v2.0
                    #newavg = newavg + rndsmp[i, j, :] * smpwht[i][j]
                    #totwht = totwht + smpwht[i, j]
                    #for k in range(5): #v2.0
                        #if rndsmp[i][j][k] < newmin[k]: #v2.0
                            #newmin[k] = rndsmp[i][j][k] #v2.0
                        #elif rndsmp[i][j][k] > newmax[k]: #v2.0
                            #newmax[k] = rndsmp[i][j][k] #v2.0
            #smlarr[j, :, 0] = newavg / totwht
            #smlarr[j, :, 1] = newmin #v2.0
            #smlarr[j, :, 2] = newmax #v2.0
            #newstd = numpy.zeros((5)) #v1.0
            #for i in range(smpnum): #1.0
            #    newstd = newstd + (rndsmp[i, j, :] - smlarr[j][0])**2 * smpwht[i][j] #v1.0
            #newstd = (newstd / totwht)**.5 #v1.0
            #smlarr[j][1] = smlarr[j][0] - 3.232 * newstd / totwht**.5 #v1.0
            #if smlarr[j][1][0] < 0: #v1.0
            #    smlarr[j][1][0] = 0 #v1.0
            #if smlarr[j][1][3] < 1: #v1.0
            #    smlarr[j][1][3] = 0 #v1.0
            #smlarr[j][2] = smlarr[j][0] + 3.232 * newstd / totwht**.5 #v1.0
        print(cutoff)
        m = m + 1
        if m > 20:
            itratn = False
    mimdst.readprob()
    #print(lowest)
    return mimdst.getavg() #smlarr[:, 0, :]

def interpimg(imgsze, xyzpts):
    replca = numpy.zeros(imgsze)
    intrpx = []
    intrpy = []
    for j in range(imgsze[0]):
        for i in range(imgsze[1]):
            intrpx.append(i)
            intrpy.append(j)
    fltsrf = interp.griddata((xyzpts[0], xyzpts[1]), xyzpts[2], (numpy.array(intrpx), numpy.array(intrpy)), method = 'nearest')
    cubsrf = interp.griddata((xyzpts[0], xyzpts[1]), xyzpts[2], (numpy.array(intrpx), numpy.array(intrpy)), method = 'cubic')
    for k in range(len(intrpx)):
        if numpy.isnan(cubsrf[k]):
            replca[intrpy[k]][intrpx[k]] = fltsrf[k]
        else:
            replca[intrpy[k]][intrpx[k]] = cubsrf[k]
    return replca

def findarea(tstree, dtresh, stresh):
    # This will find the area over which to search for the peak center by 
    # taking the height of the half-max and finding all points that are above 
    # that height such that they make a contiguous area with the highest point
    # This will take in a sub-tree, or a node with connections, and return a 
    # radius
    highpt = tstree.centrd
    lowpnt = tstree.centrd
    arelst = []
    nxtarr = [tstree]
    while len(nxtarr) > 0:
        nxtnod = nxtarr.pop(0)
        for satlte in nxtnod.satlts:
            if satlte[1] < dtresh or satlte[0].hubscr < stresh:
                nxtarr.append(satlte[0])
                arelst.append(satlte[0].centrd)
                if satlte[0].centrd[2] <= lowpnt[2]:
                    lowpnt = satlte[0].centrd
    hafmax = (highpt[2] + lowpnt[2]) / 2
    abvavg = []
    for pnt in arelst:
        if pnt[2] >= hafmax:
            abvavg.append(pnt)
    contig = [highpt]
    nxtarr = [highpt]
    xmax = highpt[0]
    xmin = highpt[0]
    ymax = highpt[1]
    ymin = highpt[1]
    while len(nxtarr) > 0:
        curpnt = nxtarr.pop(0)
        for pnt in abvavg:
            if curpnt[0] - 1 == pnt[0] and curpnt[1] == pnt[1] and not pnt in contig:
                nxtarr.append(pnt)
                contig.append(pnt)
                if pnt[0] < xmin:
                    xmin = pnt[0]
            if curpnt[1] - 1 == pnt[1] and curpnt[0] == pnt[0] and not pnt in contig:
                nxtarr.append(pnt)
                contig.append(pnt)
                if pnt[1] < ymin:
                    ymin = pnt[1]
            if curpnt[0] + 1 == pnt[0] and curpnt[1] == pnt[1] and not pnt in contig:
                nxtarr.append(pnt)
                contig.append(pnt)
                if pnt[0] > xmax:
                    xmax = pnt[0]
            if curpnt[1] + 1 == pnt[1] and curpnt[0] == pnt[0] and not pnt in contig:
                nxtarr.append(pnt)
                contig.append(pnt)
                if pnt[1] > ymax:
                    ymax = pnt[1]
    rxmax = numpy.max([xmax - highpt[0], 1])
    rxmin = numpy.max([highpt[0] - xmin, 1])
    rymax = numpy.max([ymax - highpt[1], 1])
    rymin = numpy.max([highpt[1] - ymin, 1])
    ravg = 4 * rxmax * rxmin * rymax * rymin / ((rxmin * rymax * rymin) + (rxmax * rymax * rymin) + (rxmax * rxmin * rymin) + (rxmax * rxmin * rymax))
    return ravg

def guessssqr(imgdat, point):
    # This will provide an estimate of the squared standard deviation for a 2D 
    # Gaussian surface from the immediate area surrounding the point. This 
    # will use the numeric and value weighted averages in order to prevent 
    # values that are too large or too small
    znaght = imgdat[int(point[1])][int(point[0])]
    zone = 0
    avgcnt = 0
    if int(point[0]) > 0:
        zone = zone + imgdat[point[1]][point[0] - 1]
        avgcnt = avgcnt + 1
    if int(point[0]) < imgdat.shape[1] - 1:
        zone = zone + imgdat[point[1]][point[0] + 1]
        avgcnt = avgcnt + 1
    if int(point[1]) > 0:
        zone = zone + imgdat[point[1] - 1][point[0]]
        avgcnt = avgcnt + 1
    if int(point[1]) < imgdat.shape[0] - 1:
        zone = zone + imgdat[point[1] + 1][point[0]]
        avgcnt = avgcnt + 1
    zone = zone / avgcnt
    ztwo = 0
    avgcnt = 0
    if int(point[0]) > 1:
        ztwo = ztwo + imgdat[point[1]][point[0] - 2]
        avgcnt = avgcnt + 1
    if int(point[0]) < imgdat.shape[1] - 2:
        ztwo = ztwo + imgdat[point[1]][point[0] + 2]
        avgcnt = avgcnt + 1
    if int(point[1]) > 1:
        ztwo = ztwo + imgdat[point[1] - 2][point[0]]
        avgcnt = avgcnt + 1
    if int(point[1]) < imgdat.shape[0] - 2:
        ztwo = ztwo + imgdat[point[1] + 2][point[0]]
        avgcnt = avgcnt + 1
    ztwo = ztwo / avgcnt
    zthr = 0
    avgcnt = 0
    if int(point[0]) > 2:
        zthr = zthr + imgdat[point[1]][point[0] - 3]
        avgcnt = avgcnt + 1
    if int(point[0]) < imgdat.shape[1] - 3:
        zthr = zthr + imgdat[point[1]][point[0] + 3]
        avgcnt = avgcnt + 1
    if int(point[1]) > 2:
        zthr = zthr + imgdat[point[1] - 3][point[0]]
        avgcnt = avgcnt + 1
    if int(point[1]) < imgdat.shape[0] - 3:
        zthr = zthr + imgdat[point[1] + 3][point[0]]
        avgcnt = avgcnt + 1
    zthr = zthr / avgcnt
    zfor = 0
    avgcnt = 0
    if int(point[0]) > 3:
        zfor = zfor + imgdat[point[1]][point[0] - 4]
        avgcnt = avgcnt + 1
    if int(point[0]) < imgdat.shape[1] - 4:
        zfor = zfor + imgdat[point[1]][point[0] + 4]
        avgcnt = avgcnt + 1
    if int(point[1]) > 3:
        zfor = zfor + imgdat[point[1] - 4][point[0]]
        avgcnt = avgcnt + 1
    if int(point[1]) < imgdat.shape[0] - 4:
        zfor = zfor + imgdat[point[1] + 4][point[0]]
        avgcnt = avgcnt + 1
    zfor = zfor / avgcnt
    zfiv = 0
    avgcnt = 0
    if int(point[0]) > 4:
        zfiv = zfiv + imgdat[point[1]][point[0] - 5]
        avgcnt = avgcnt + 1
    if int(point[0]) < imgdat.shape[1] - 5:
        zfiv = zfiv + imgdat[point[1]][point[0] + 5]
        avgcnt = avgcnt + 1
    if int(point[1]) > 4:
        zfiv = zfiv + imgdat[point[1] - 5][point[0]]
        avgcnt = avgcnt + 1
    if int(point[1]) < imgdat.shape[0] - 5:
        zfiv = zfiv + imgdat[point[1] + 5][point[0]]
        avgcnt = avgcnt + 1
    zfiv = zfiv / avgcnt
    fzro = (ztwo - zone) / (zone - znaght)
    fone = (zthr - ztwo) / (ztwo - zone)
    ftwo = (zfor - zthr) / (zthr - ztwo)
    fthr = (zfiv - zfor) / (zfor - zthr)
    numsig = numpy.max([-1.2 * fzro / (fzro - 3), 1])
    numsig = numsig + numpy.max([-8 * fone / (3 * fone - 5), 1])
    numsig = numsig + numpy.max([-21.5 * ftwo / (5 * ftwo - 7), 1])
    numsig = numsig + numpy.max([-44 * fthr / (7 * fthr - 9), 1])
    sqrsig = numpy.max([(-1.2 * fzro / (fzro - 3))**2, 1])
    sqrsig = sqrsig + numpy.max([(-8 * fone / (3 * fone - 5))**2, 1])
    sqrsig = sqrsig + numpy.max([(-21.5 * ftwo / (5 * ftwo - 7))**2, 1])
    sqrsig = sqrsig + numpy.max([(-44 * fthr / (7 * fthr - 9))**2, 1])
    sqrsig = sqrsig / numsig
    numsig = numsig / 4
    sigsqr = (2 * numsig + sqrsig) / 3
    #print([sigsqr, numsig, sqrsig])
    if sigsqr < 1:
        sigsqr = 1
    return sigsqr

def ltdgauss(xyvect, xnaght, ynaght, height, sigsqr, baslin):
    # A function for calculating a radial Gaussian surface. Used primarily for 
    # scipy.optimize.curve_fit
    return height * numpy.exp(-1 * ((xyvect[0] - xnaght)**2 + (xyvect[1] - ynaght)**2) / (2 * sigsqr)) + baslin

def iitdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    return surf + baslin

def iiitdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, xiii, yiii, hiii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssi))
    return surf + baslin

def ivtdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, xiii, yiii, hiii, xiv, yiv, hiv):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssi))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssi))
    return surf + baslin

def vtdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, xiii, yiii, hiii, xiv, yiv, hiv, xv, yv, hv):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssi))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssi))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssi))
    return surf + baslin

def vitdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, xiii, yiii, hiii, xiv, yiv, hiv, xv, yv, hv, xvi, yvi, hvi):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssi))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssi))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssi))
    surf = surf + hvi * numpy.exp(-1 * ((xyvect[0] - xvi)**2 + (xyvect[1] - yvi)**2) / (2 * ssi))
    return surf + baslin

def viitdgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, xiii, yiii, hiii, xiv, yiv, hiv, xv, yv, hv, xvi, yvi, hvi, xvii, yvii, hvii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssi))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssi))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssi))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssi))
    surf = surf + hvi * numpy.exp(-1 * ((xyvect[0] - xvi)**2 + (xyvect[1] - yvi)**2) / (2 * ssi))
    surf = surf + hvii * numpy.exp(-1 * ((xyvect[0] - xvii)**2 + (xyvect[1] - yvii)**2) / (2 * ssi))
    return surf + baslin

def flatsurf(xyvect, baslin):
    return baslin

def xyhgauss(xyvect, xnaght, ynaght, height, width):
    # A function for calculating a radial Gaussian surface. Used primarily for 
    # scipy.optimize.curve_fit
    return height * numpy.exp(-1 * ((xyvect[0] - xnaght)**2 + (xyvect[1] - ynaght)**2) / (2 * width))

def iixyhgauss(xyvect, xi, yi, hi, ssi, baslin, xii, yii, hii, ssii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    return surf

def iiixyhgauss(xyvect, xi, yi, hi, ssi, xii, yii, hii, ssii, xiii, yiii, hiii, ssiii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssiii))
    return surf

def ivxyhgauss(xyvect, xi, yi, hi, ssi, xii, yii, hii, ssii, xiii, yiii, hiii, ssiii, xiv, yiv, hiv, ssiv):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssiv))
    return surf

def vxyhgauss(xyvect, xi, yi, hi, ssi, xii, yii, hii, ssii, xiii, yiii, hiii, ssiii, xiv, yiv, hiv, ssiv, xv, yv, hv, ssv):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssv))
    return surf

def vixyhgauss(xyvect, xi, yi, hi, ssi, xii, yii, hii, ssii, xiii, yiii, hiii, ssiii, xiv, yiv, hiv, ssiv, xv, yv, hv, ssv, xvi, yvi, hvi, ssvi):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xyvect[0] - xvi)**2 + (xyvect[1] - yvi)**2) / (2 * ssvi))
    return surf

def viixyhgauss(xyvect, xi, yi, hi, ssi, xii, yii, hii, ssii, xiii, yiii, hiii, ssiii, xiv, yiv, hiv, ssiv, xv, yv, hv, ssv, xvi, yvi, hvi, ssvi, xvii, yvii, hvii, ssvii):
    surf = hi * numpy.exp(-1 * ((xyvect[0] - xi)**2 + (xyvect[1] - yi)**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xyvect[0] - xii)**2 + (xyvect[1] - yii)**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xyvect[0] - xiii)**2 + (xyvect[1] - yiii)**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xyvect[0] - xiv)**2 + (xyvect[1] - yiv)**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xyvect[0] - xv)**2 + (xyvect[1] - yv)**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xyvect[0] - xvi)**2 + (xyvect[1] - yvi)**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xyvect[0] - xvii)**2 + (xyvect[1] - yvii)**2) / (2 * ssvii))
    return surf

def nsiggauss(xyvect, sigsqr):
    surf = 0
    for i in range(2, len(xyvect), 3):
        surf = surf + xyvect[i + 2] * numpy.exp(-1 * ((xyvect[0] - xyvect[i])**2 + (xyvect[1] - xyvect[i + 1])**2) / (2 * sigsqr))
    return surf

def cannedfit(imgdat, area, inpara = []):
    # Fits a Guassian-type surface to an image. This first defines a window of 
    # points to fit against based on the guess of the standard deviation, 
    # passed to the method, then uses scipy.optimize.curve_fit to do the 
    # fitting because my curve fitting algorithm sucks
    xvectr = []
    yvectr = []
    zvectr = []
    for pnt in area:
        xvectr.append(pnt[0])
        yvectr.append(pnt[1])
        zvectr.append(imgdat[pnt[1]][pnt[0]])
    lounds = []
    uounds = []
    if len(inpara) == 0:
        lounds.append(numpy.min(xvectr))
        lounds.append(numpy.min(yvectr))
        lounds.append(0)
        lounds.append(1)
        uounds.append(numpy.max(xvectr))
        uounds.append(numpy.max(yvectr))
        uounds.append(numpy.max(imgdat) - numpy.min(imgdat))
        uounds.append(numpy.min((uounds[0] - lounds[0], uounds[1] - lounds[1]))**.5)
    else:
        for i in range(numpy.min((len(inpara), 7))):
            lounds.append(inpara[i][0] - inpara[i][2]**.5)
            lounds.append(inpara[i][1] - inpara[i][2]**.5)
            lounds.append(0)
            lounds.append(1)
            uounds.append(inpara[i][0] + inpara[i][2]**.5)
            uounds.append(inpara[i][1] + inpara[i][2]**.5)
            uounds.append(numpy.max(imgdat) - numpy.min(imgdat))
            uounds.append(inpara[i][2] * 2)
    if len(inpara) <= 1:
        ptgess, fitgud = sciopt.curve_fit(xyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) == 2:
        ptgess, fitgud = sciopt.curve_fit(iixyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) == 3:
        ptgess, fitgud = sciopt.curve_fit(iiixyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) == 4:
        ptgess, fitgud = sciopt.curve_fit(ivxyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) == 5:
        ptgess, fitgud = sciopt.curve_fit(vxyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) == 6:
        ptgess, fitgud = sciopt.curve_fit(vixyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    if len(inpara) >= 7:
        ptgess, fitgud = sciopt.curve_fit(viixyhgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
    return (ptgess[2], ptgess[0], ptgess[1], ptgess[3])

def stockfit(imgdat, area, benfit = [], inpara = [], waunds = [], baonds = []):
    # Fits a Gaussian-type surface to an image. This is like canned fit, but 
    # it uses a pre-defined window obtained from a piece of software better 
    # capable of identifying appropriate windows. This will need to take in a 
    # point so that it can get the xy bounds and starting point
    xvectr = []
    yvectr = []
    zvectr = []
    for point in area:
        xvectr.append(point[0])
        yvectr.append(point[1])
        zvectr.append(imgdat[point[1]][point[0]])
        for fitnod in benfit:
            zvectr[-1] = zvectr[-1] - fitnod[0] * numpy.exp(-1 * ((point[0] - fitnod[1])**2 + (point[1] - fitnod[2])**2) / (2 * fitnod[3]))
    lounds = []
    uounds = []
    if len(inpara) == 0:
        lounds.append(numpy.min(xvectr))
        lounds.append(numpy.min(yvectr))
        lounds.append(0)
        if len(waunds) >= 2:
            lounds.append(waunds[0])
        else:
            lounds.append(1)
        if len(baonds) >= 2:
            lounds.append(baonds[0])
        else:
            lounds.append(numpy.min(imgdat))
        uounds.append(numpy.max(xvectr))
        uounds.append(numpy.max(yvectr))
        uounds.append(numpy.max(imgdat) - numpy.min(imgdat))
        if len(waunds) >= 2:
            uounds.append(waunds[1])
        else:
            uounds.append(numpy.max([numpy.max(xvectr) - numpy.min(xvectr), numpy.max(yvectr) - numpy.min(yvectr)]))
        if len(baonds) >= 2:
            uounds.append(baonds[1])
        else:
            uounds.append(numpy.max(imgdat))
    else:
        lounds.append(inpara[0][0] - inpara[0][2] / 4)
        lounds.append(inpara[0][1] - inpara[0][2] / 4)
        lounds.append(0)
        if len(waunds) >= 2:
            lounds.append(waunds[0])
        else:
            lounds.append(1)
        if len(baonds) >= 2:
            lounds.append(baonds[0])
        else:
            lounds.append(numpy.min(imgdat))
        uounds.append(inpara[0][0] + inpara[0][2] / 4)
        uounds.append(inpara[0][1] + inpara[0][2] / 4)
        uounds.append(numpy.max(imgdat) - numpy.min(imgdat))
        if len(waunds) >= 2:
            uounds.append(waunds[1])
        else:
            uounds.append((inpara[0][2]**2 + numpy.max([numpy.max(xvectr) - numpy.min(xvectr), numpy.max(yvectr) - numpy.min(yvectr)])) / 2)
        if len(baonds) >= 2:
            uounds.append(baonds[1])
        else:
            uounds.append(numpy.max(imgdat))
        for i in range(1, numpy.min((7, len(inpara)))):
            lounds.append(inpara[i][0] - inpara[0][2] / 4)
            lounds.append(inpara[i][1] - inpara[0][2] / 4)
            lounds.append(0)
            #lounds.append(1)
            uounds.append(inpara[i][0] + inpara[0][2] / 4)
            uounds.append(inpara[i][1] + inpara[0][2] / 4)
            uounds.append(numpy.max(imgdat) - numpy.min(imgdat))
            #uounds.append(numpy.max([numpy.max(xvectr) - numpy.min(xvectr), numpy.max(yvectr) - numpy.min(yvectr)]))
    try:
        if len(inpara) <= 1:
            ptgess, fitgud = sciopt.curve_fit(ltdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) == 2:
            ptgess, fitgud = sciopt.curve_fit(iitdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) == 3:
            ptgess, fitgud = sciopt.curve_fit(iiitdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) == 4:
            ptgess, fitgud = sciopt.curve_fit(ivtdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) == 5:
            ptgess, fitgud = sciopt.curve_fit(vtdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) == 6:
            ptgess, fitgud = sciopt.curve_fit(vitdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        elif len(inpara) >= 7:
            ptgess, fitgud = sciopt.curve_fit(viitdgauss, [xvectr, yvectr], zvectr, bounds = (lounds, uounds))
        return (ptgess[2], ptgess[0], ptgess[1], ptgess[3], ptgess[4])
    except:
        print(lounds)
        print(uounds)

def istdgauss(xydata, hi, ssi, bi):
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi)) + bi
    return surf

def iistdgauss(xydata, hi, ssi, bi, hii, ssii, bii):
    xinter = []
    yinter = []
    for i in range(2, 5, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    return surf + baslin

def iiistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii):
    xinter = []
    yinter = []
    for i in range(2, 7, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi)) + bi
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    return surf + baslin

def ivstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv):
    xinter = []
    yinter = []
    for i in range(2, 9, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi)) + bi
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    return surf + baslin

def vstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv):
    xinter = []
    yinter = []
    for i in range(2, 11, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    return surf + baslin

def vistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi):
    xinter = []
    yinter = []
    for i in range(2, 13, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    return surf + baslin

def viistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii):
    xinter = []
    yinter = []
    for i in range(2, 15, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    return surf + baslin

def viiistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii):
    xinter = []
    yinter = []
    for i in range(2, 17, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    return surf + baslin

def ixstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix):
    xinter = []
    yinter = []
    for i in range(2, 19, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    return surf + baslin

def xstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx):
    xinter = []
    yinter = []
    for i in range(2, 21, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    return surf + baslin

def xistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx, hxi, ssxi, bxi):
    xinter = []
    yinter = []
    for i in range(2, 23, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx, bxi]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    surf = surf + hxi * numpy.exp(-1 * ((xydata[0] - xydata[22])**2 + (xydata[1] - xydata[23])**2) / (2 * ssxi))
    return surf + baslin

def xiistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx, hxi, ssxi, bxi, hxii, ssxii, bxii):
    xinter = []
    yinter = []
    for i in range(2, 25, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx, bxi, bxii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    surf = surf + hxi * numpy.exp(-1 * ((xydata[0] - xydata[22])**2 + (xydata[1] - xydata[23])**2) / (2 * ssxi))
    surf = surf + hxii * numpy.exp(-1 * ((xydata[0] - xydata[24])**2 + (xydata[1] - xydata[25])**2) / (2 * ssxii))
    return surf + baslin

def xiiistdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx, hxi, ssxi, bxi, hxii, ssxii, bxii, hxiii, ssxiii, bxiii):
    xinter = []
    yinter = []
    for i in range(2, 27, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx, bxi, bxii, bxiii]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    surf = surf + hxi * numpy.exp(-1 * ((xydata[0] - xydata[22])**2 + (xydata[1] - xydata[23])**2) / (2 * ssxi))
    surf = surf + hxii * numpy.exp(-1 * ((xydata[0] - xydata[24])**2 + (xydata[1] - xydata[25])**2) / (2 * ssxii))
    surf = surf + hxiii * numpy.exp(-1 * ((xydata[0] - xydata[26])**2 + (xydata[1] - xydata[27])**2) / (2 * ssxiii))
    return surf + baslin

def xivstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx, hxi, ssxi, bxi, hxii, ssxii, bxii, hxiii, ssxiii, bxiii, hxiv, ssxiv, bxiv):
    xinter = []
    yinter = []
    for i in range(2, 29, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx, bxi, bxii, bxiii, bxiv]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    surf = surf + hxi * numpy.exp(-1 * ((xydata[0] - xydata[22])**2 + (xydata[1] - xydata[23])**2) / (2 * ssxi))
    surf = surf + hxii * numpy.exp(-1 * ((xydata[0] - xydata[24])**2 + (xydata[1] - xydata[25])**2) / (2 * ssxii))
    surf = surf + hxiii * numpy.exp(-1 * ((xydata[0] - xydata[26])**2 + (xydata[1] - xydata[27])**2) / (2 * ssxiii))
    surf = surf + hxiv * numpy.exp(-1 * ((xydata[0] - xydata[28])**2 + (xydata[1] - xydata[29])**2) / (2 * ssxiv))
    return surf + baslin

def xvstdgauss(xydata, hi, ssi, bi, hii, ssii, bii, hiii, ssiii, biii, hiv, ssiv, biv, hv, ssv, bv, hvi, ssvi, bvi, hvii, ssvii, bvii, hviii, ssviii, bviii, hix, ssix, bix, hx, ssx, bx, hxi, ssxi, bxi, hxii, ssxii, bxii, hxiii, ssxiii, bxiii, hxiv, ssxiv, bxiv, hxv, ssxv, bxv): #, hxvi, ssxvi, bxvi, hxvii, ssxvii, bxvii, hxviii, ssxviii, bxviii, hxix, ssxix, bxix, hxx, ssxx, bxx, hxxi, ssxxi, bxxi, hxxii, ssxxii, bxxii, hxxiii, ssxxiii, bxxiii, hxxiv, ssxxiv, bxxiv, hxxv, ssxxv, bxxv, hxxvi, ssxxvi, bxxvi, hxxvii, ssxxvii, bxxvii, hxxviii, ssxxviii, bxxviii, hxxix, ssxxix, bxxix, hxxx, ssxxx, bxxx
    xinter = []
    yinter = []
    for i in range(2, 31, 2):
        xinter.append(xydata[i])
        yinter.append(xydata[i + 1])
    baslin = interp.griddata((numpy.array(xinter), numpy.array(yinter)), numpy.array([bi, bii, biii, biv, bv, bvi, bvii, bviii, bix, bx, bxi, bxii, bxiii, bxiv, bv]), (numpy.array([xydata[0]]), numpy.array([xydata[1]])), method = 'nearest')
    surf = hi * numpy.exp(-1 * ((xydata[0] - xydata[2])**2 + (xydata[1] - xydata[3])**2) / (2 * ssi))
    surf = surf + hii * numpy.exp(-1 * ((xydata[0] - xydata[4])**2 + (xydata[1] - xydata[5])**2) / (2 * ssii))
    surf = surf + hiii * numpy.exp(-1 * ((xydata[0] - xydata[6])**2 + (xydata[1] - xydata[7])**2) / (2 * ssiii))
    surf = surf + hiv * numpy.exp(-1 * ((xydata[0] - xydata[8])**2 + (xydata[1] - xydata[9])**2) / (2 * ssiv))
    surf = surf + hv * numpy.exp(-1 * ((xydata[0] - xydata[10])**2 + (xydata[1] - xydata[11])**2) / (2 * ssv))
    surf = surf + hvi * numpy.exp(-1 * ((xydata[0] - xydata[12])**2 + (xydata[1] - xydata[13])**2) / (2 * ssvi))
    surf = surf + hvii * numpy.exp(-1 * ((xydata[0] - xydata[14])**2 + (xydata[1] - xydata[15])**2) / (2 * ssvii))
    surf = surf + hviii * numpy.exp(-1 * ((xydata[0] - xydata[16])**2 + (xydata[1] - xydata[17])**2) / (2 * ssviii))
    surf = surf + hix * numpy.exp(-1 * ((xydata[0] - xydata[18])**2 + (xydata[1] - xydata[19])**2) / (2 * ssix))
    surf = surf + hx * numpy.exp(-1 * ((xydata[0] - xydata[20])**2 + (xydata[1] - xydata[21])**2) / (2 * ssx))
    surf = surf + hxi * numpy.exp(-1 * ((xydata[0] - xydata[22])**2 + (xydata[1] - xydata[23])**2) / (2 * ssxi))
    surf = surf + hxii * numpy.exp(-1 * ((xydata[0] - xydata[24])**2 + (xydata[1] - xydata[25])**2) / (2 * ssxii))
    surf = surf + hxiii * numpy.exp(-1 * ((xydata[0] - xydata[26])**2 + (xydata[1] - xydata[27])**2) / (2 * ssxiii))
    surf = surf + hxiv * numpy.exp(-1 * ((xydata[0] - xydata[28])**2 + (xydata[1] - xydata[29])**2) / (2 * ssxiv))
    surf = surf + hxv * numpy.exp(-1 * ((xydata[0] - xydata[30])**2 + (xydata[1] - xydata[31])**2) / (2 * ssxv))
    #surf = surf + hxvi * numpy.exp(-1 * ((xydata[0] - xydata[32])**2 + (xydata[1] - xydata[33])**2) / (2 * ssxvi)) + bxvi * (totwht - ((xydata[0] - xydata[32])**2 + (xydata[1] - xydata[33])**2)**.5) / totwht
    #surf = surf + hxvii * numpy.exp(-1 * ((xydata[0] - xydata[34])**2 + (xydata[1] - xydata[35])**2) / (2 * ssxvii)) + bxvii * (totwht - ((xydata[0] - xydata[34])**2 + (xydata[1] - xydata[35])**2)**.5) / totwht
    #surf = surf + hxviii * numpy.exp(-1 * ((xydata[0] - xydata[36])**2 + (xydata[1] - xydata[37])**2) / (2 * ssxviii)) + bxviii * (totwht - ((xydata[0] - xydata[36])**2 + (xydata[1] - xydata[37])**2)**.5) / totwht
    #surf = surf + hxix * numpy.exp(-1 * ((xydata[0] - xydata[38])**2 + (xydata[1] - xydata[39])**2) / (2 * ssxix)) + bxix * (totwht - ((xydata[0] - xydata[38])**2 + (xydata[1] - xydata[39])**2)**.5) / totwht
    #surf = surf + hxx * numpy.exp(-1 * ((xydata[0] - xydata[40])**2 + (xydata[1] - xydata[41])**2) / (2 * ssxx)) + bxx * (totwht - ((xydata[0] - xydata[40])**2 + (xydata[1] - xydata[41])**2)**.5) / totwht
    #surf = surf + hxxi * numpy.exp(-1 * ((xydata[0] - xydata[42])**2 + (xydata[1] - xydata[43])**2) / (2 * ssxxi)) + bxxi * (totwht - ((xydata[0] - xydata[42])**2 + (xydata[1] - xydata[43])**2)**.5) / totwht
    #surf = surf + hxxii * numpy.exp(-1 * ((xydata[0] - xydata[44])**2 + (xydata[1] - xydata[45])**2) / (2 * ssxxii)) + bxxii * (totwht - ((xydata[0] - xydata[44])**2 + (xydata[1] - xydata[45])**2)**.5) / totwht
    #surf = surf + hxxiii * numpy.exp(-1 * ((xydata[0] - xydata[46])**2 + (xydata[1] - xydata[47])**2) / (2 * ssxxiii)) + bxxiii * (totwht - ((xydata[0] - xydata[46])**2 + (xydata[1] - xydata[47])**2)**.5) / totwht
    #surf = surf + hxxiv * numpy.exp(-1 * ((xydata[0] - xydata[48])**2 + (xydata[1] - xydata[49])**2) / (2 * ssxxiv)) + bxxiv * (totwht - ((xydata[0] - xydata[48])**2 + (xydata[1] - xydata[49])**2)**.5) / totwht
    #surf = surf + hxxv * numpy.exp(-1 * ((xydata[0] - xydata[50])**2 + (xydata[1] - xydata[51])**2) / (2 * ssxxv)) + bxxv * (totwht - ((xydata[0] - xydata[50])**2 + (xydata[1] - xydata[51])**2)**.5) / totwht
    #surf = surf + hxxvi * numpy.exp(-1 * ((xydata[0] - xydata[52])**2 + (xydata[1] - xydata[53])**2) / (2 * ssxxvi)) + bxxvi * (totwht - ((xydata[0] - xydata[52])**2 + (xydata[1] - xydata[53])**2)**.5) / totwht
    #surf = surf + hxxvii * numpy.exp(-1 * ((xydata[0] - xydata[54])**2 + (xydata[1] - xydata[55])**2) / (2 * ssxxvii)) + bxxvii * (totwht - ((xydata[0] - xydata[54])**2 + (xydata[1] - xydata[55])**2)**.5) / totwht
    #surf = surf + hxxviii * numpy.exp(-1 * ((xydata[0] - xydata[56])**2 + (xydata[1] - xydata[57])**2) / (2 * ssxxviii)) + bxxviii * (totwht - ((xydata[0] - xydata[56])**2 + (xydata[1] - xydata[57])**2)**.5) / totwht
    #surf = surf + hxxix * numpy.exp(-1 * ((xydata[0] - xydata[58])**2 + (xydata[1] - xydata[59])**2) / (2 * ssxxix)) + bxxix * (totwht - ((xydata[0] - xydata[58])**2 + (xydata[1] - xydata[59])**2)**.5) / totwht
    #surf = surf + hxxx * numpy.exp(-1 * ((xydata[0] - xydata[60])**2 + (xydata[1] - xydata[61])**2) / (2 * ssxxx)) + bxxx * (totwht - ((xydata[0] - xydata[60])**2 + (xydata[1] - xydata[61])**2)**.5) / totwht
    return surf + baslin

def stdfit(imgdat, area, inbund, pntinf):
    xvect = []
    yvect = []
    zvect = []
    for pnt in area:
        xvect.append(pnt[0])
        yvect.append(pnt[1])
        zvect.append(imgdat[pnt[1]][pnt[0]])
    lounds = [inbund[0][0][0], inbund[0][0][3], inbund[0][0][4]]
    uounds = [inbund[0][1][0], inbund[0][1][3], inbund[0][1][4]]
    for i in range(1, numpy.min((len(inbund), 15))):
        lounds.append(inbund[i][0][0])
        lounds.append(inbund[i][0][3])
        uounds.append(inbund[i][1][0])
        uounds.append(inbund[i][1][3])
    indata = [xvect, yvect]
    for i in range(numpy.min((len(pntinf), 15))):
        indata.append([])
        indata.append([])
    for i in range(len(xvect)):
        for j in range(2, len(indata), 2):
            indata[j].append(pntinf[int((j - 2) / 2)][0])
            indata[j + 1].append(pntinf[int((j - 2) / 2)][1])
    if len(pntinf) == 1:
        ptgess, fitgud = sciopt.curve_fit(istdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 2:
        ptgess, fitgud = sciopt.curve_fit(iistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 3:
        ptgess, fitgud = sciopt.curve_fit(iiistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 4:
        ptgess, fitgud = sciopt.curve_fit(ivstdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 5:
        ptgess, fitgud = sciopt.curve_fit(vstdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 6:
        ptgess, fitgud = sciopt.curve_fit(vistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 7:
        ptgess, fitgud = sciopt.curve_fit(viistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 8:
        ptgess, fitgud = sciopt.curve_fit(viiistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 9:
        ptgess, fitgud = sciopt.curve_fit(ixstdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 10:
        ptgess, fitgud = sciopt.curve_fit(xstdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 11:
        ptgess, fitgud = sciopt.curve_fit(xistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 12:
        ptgess, fitgud = sciopt.curve_fit(xiistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 13:
        ptgess, fitgud = sciopt.curve_fit(xiiistdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 14:
        ptgess, fitgud = sciopt.curve_fit(xivstdgauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) >= 15:
        ptgess, fitgud = sciopt.curve_fit(xvstdgauss, indata, zvect, bounds = (lounds, uounds))
    return ptgess

def ixygauss(xydata, xi, yi):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    return surf

def iixygauss(xydata, xi, yi, xii, yii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    return surf

def iiixygauss(xydata, xi, yi, xii, yii, xiii, yiii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    return surf

def ivxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    return surf

def vxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    return surf

def vixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    return surf

def viixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    return surf

def viiixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    return surf

def ixxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    return surf

def xxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    return surf

def xixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx, xxi, yxi):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    surf = surf + xydata[23] * numpy.exp(-1 * ((xydata[0] - xxi)**2 + (xydata[1] - yxi)**2) / (2 * xydata[24]))
    return surf

def xiixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx, xxi, yxi, xxii, yxii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    surf = surf + xydata[23] * numpy.exp(-1 * ((xydata[0] - xxi)**2 + (xydata[1] - yxi)**2) / (2 * xydata[24]))
    surf = surf + xydata[25] * numpy.exp(-1 * ((xydata[0] - xxii)**2 + (xydata[1] - yxii)**2) / (2 * xydata[26]))
    return surf

def xiiixygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx, xxi, yxi, xxii, yxii, xxiii, yxiii):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    surf = surf + xydata[23] * numpy.exp(-1 * ((xydata[0] - xxi)**2 + (xydata[1] - yxi)**2) / (2 * xydata[24]))
    surf = surf + xydata[25] * numpy.exp(-1 * ((xydata[0] - xxii)**2 + (xydata[1] - yxii)**2) / (2 * xydata[26]))
    surf = surf + xydata[27] * numpy.exp(-1 * ((xydata[0] - xxiii)**2 + (xydata[1] - yxiii)**2) / (2 * xydata[28]))
    return surf

def xivxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx, xxi, yxi, xxii, yxii, xxiii, yxiii, xxiv, yxiv):
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    surf = surf + xydata[23] * numpy.exp(-1 * ((xydata[0] - xxi)**2 + (xydata[1] - yxi)**2) / (2 * xydata[24]))
    surf = surf + xydata[25] * numpy.exp(-1 * ((xydata[0] - xxii)**2 + (xydata[1] - yxii)**2) / (2 * xydata[26]))
    surf = surf + xydata[27] * numpy.exp(-1 * ((xydata[0] - xxiii)**2 + (xydata[1] - yxiii)**2) / (2 * xydata[28]))
    surf = surf + xydata[29] * numpy.exp(-1 * ((xydata[0] - xxiv)**2 + (xydata[1] - yxiv)**2) / (2 * xydata[30]))
    return surf

def xvxygauss(xydata, xi, yi, xii, yii, xiii, yiii, xiv, yiv, xv, yv, xvi, yvi, xvii, yvii, xviii, yviii, xix, yix, xx, yx, xxi, yxi, xxii, yxii, xxiii, yxiii, xxiv, yxiv, xxv, yxv): #, xxvi, yxvi, xxvii, yxvii, xxviii, yxviii, xxix, yxix, xxx, yxx, xxxi, yxxi, xxxii, yxxii, xxxiii, yxxiii, xxxiv, yxxiv, xxxv, yxxv, xxxvi, yxxvi, xxxvii, yxxvii, xxxviii, yxxviii, xxxix, yxxix, xxxx, yxxx
    #totwht = totwht + ((xydata[0] - xxvi)**2 + (xydata[1] - yxvi)**2)**.5
    #totwht = totwht + ((xydata[0] - xxvii)**2 + (xydata[1] - yxvii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxviii)**2 + (xydata[1] - yxviii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxix)**2 + (xydata[1] - yxix)**2)**.5
    #totwht = totwht + ((xydata[0] - xxx)**2 + (xydata[1] - yxx)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxi)**2 + (xydata[1] - yxxi)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxii)**2 + (xydata[1] - yxxii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxiii)**2 + (xydata[1] - yxxiii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxiv)**2 + (xydata[1] - yxxiv)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxv)**2 + (xydata[1] - yxxv)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxvi)**2 + (xydata[1] - yxxvi)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxvii)**2 + (xydata[1] - yxxvii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxviii)**2 + (xydata[1] - yxxviii)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxix)**2 + (xydata[1] - yxxix)**2)**.5
    #totwht = totwht + ((xydata[0] - xxxx)**2 + (xydata[1] - yxxx)**2)**.5
    surf = xydata[2] * numpy.exp(-1 * ((xydata[0] - xi)**2 + (xydata[1] - yi)**2) / (2 * xydata[3])) + xydata[4]
    surf = surf + xydata[5] * numpy.exp(-1 * ((xydata[0] - xii)**2 + (xydata[1] - yii)**2) / (2 * xydata[6]))
    surf = surf + xydata[7] * numpy.exp(-1 * ((xydata[0] - xiii)**2 + (xydata[1] - yiii)**2) / (2 * xydata[8]))
    surf = surf + xydata[9] * numpy.exp(-1 * ((xydata[0] - xiv)**2 + (xydata[1] - yiv)**2) / (2 * xydata[10]))
    surf = surf + xydata[11] * numpy.exp(-1 * ((xydata[0] - xv)**2 + (xydata[1] - yv)**2) / (2 * xydata[12]))
    surf = surf + xydata[13] * numpy.exp(-1 * ((xydata[0] - xvi)**2 + (xydata[1] - yvi)**2) / (2 * xydata[14]))
    surf = surf + xydata[15] * numpy.exp(-1 * ((xydata[0] - xvii)**2 + (xydata[1] - yvii)**2) / (2 * xydata[16]))
    surf = surf + xydata[17] * numpy.exp(-1 * ((xydata[0] - xviii)**2 + (xydata[1] - yviii)**2) / (2 * xydata[18]))
    surf = surf + xydata[19] * numpy.exp(-1 * ((xydata[0] - xix)**2 + (xydata[1] - yix)**2) / (2 * xydata[20]))
    surf = surf + xydata[21] * numpy.exp(-1 * ((xydata[0] - xx)**2 + (xydata[1] - yx)**2) / (2 * xydata[22]))
    surf = surf + xydata[23] * numpy.exp(-1 * ((xydata[0] - xxi)**2 + (xydata[1] - yxi)**2) / (2 * xydata[24]))
    surf = surf + xydata[25] * numpy.exp(-1 * ((xydata[0] - xxii)**2 + (xydata[1] - yxii)**2) / (2 * xydata[26]))
    surf = surf + xydata[27] * numpy.exp(-1 * ((xydata[0] - xxiii)**2 + (xydata[1] - yxiii)**2) / (2 * xydata[28]))
    surf = surf + xydata[29] * numpy.exp(-1 * ((xydata[0] - xxiv)**2 + (xydata[1] - yxiv)**2) / (2 * xydata[30]))
    surf = surf + xydata[31] * numpy.exp(-1 * ((xydata[0] - xxv)**2 + (xydata[1] - yxv)**2) / (2 * xydata[32]))
    #surf = surf + xydata[47] * numpy.exp(-1 * ((xydata[0] - xxvi)**2 + (xydata[1] - yxvi)**2)**.5 / (2 * xydata[48])) + xydata[49] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[50] * numpy.exp(-1 * ((xydata[0] - xxvii)**2 + (xydata[1] - yxvii)**2)**.5 / (2 * xydata[51])) + xydata[52] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[53] * numpy.exp(-1 * ((xydata[0] - xxviii)**2 + (xydata[1] - yxviii)**2)**.5 / (2 * xydata[54])) + xydata[55] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[56] * numpy.exp(-1 * ((xydata[0] - xxix)**2 + (xydata[1] - yxix)**2)**.5 / (2 * xydata[57])) + xydata[58] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[59] * numpy.exp(-1 * ((xydata[0] - xxx)**2 + (xydata[1] - yxx)**2)**.5 / (2 * xydata[60])) + xydata[61] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[62] * numpy.exp(-1 * ((xydata[0] - xxxi)**2 + (xydata[1] - yxxi)**2)**.5 / (2 * xydata[63])) + xydata[64] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[65] * numpy.exp(-1 * ((xydata[0] - xxxii)**2 + (xydata[1] - yxxii)**2)**.5 / (2 * xydata[66])) + xydata[67] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[68] * numpy.exp(-1 * ((xydata[0] - xxxiii)**2 + (xydata[1] - yxxiii)**2)**.5 / (2 * xydata[69])) + xydata[70] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[71] * numpy.exp(-1 * ((xydata[0] - xxxiv)**2 + (xydata[1] - yxxiv)**2)**.5 / (2 * xydata[72])) + xydata[73] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[74] * numpy.exp(-1 * ((xydata[0] - xxxv)**2 + (xydata[1] - yxxv)**2)**.5 / (2 * xydata[75])) + xydata[76] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[77] * numpy.exp(-1 * ((xydata[0] - xxxvi)**2 + (xydata[1] - yxxvi)**2)**.5 / (2 * xydata[78])) + xydata[79] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[80] * numpy.exp(-1 * ((xydata[0] - xxxvii)**2 + (xydata[1] - yxxvii)**2)**.5 / (2 * xydata[81])) + xydata[82] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[83] * numpy.exp(-1 * ((xydata[0] - xxxviii)**2 + (xydata[1] - yxxviii)**2)**.5 / (2 * xydata[84])) + xydata[85] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[86] * numpy.exp(-1 * ((xydata[0] - xxxix)**2 + (xydata[1] - yxxix)**2)**.5 / (2 * xydata[87])) + xydata[88] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    #surf = surf + xydata[89] * numpy.exp(-1 * ((xydata[0] - xxxx)**2 + (xydata[1] - yxxx)**2)**.5 / (2 * xydata[90])) + xydata[91] * (totwht - ((xydata[0] - xi)**2 + (xydata[1] - yi)**2)**.5) / totwht
    return surf

def posfit(imgdat, area, inbund, pntinf):
    xvect = []
    yvect = []
    zvect = []
    for pnt in area:
        xvect.append(pnt[0])
        yvect.append(pnt[1])
        zvect.append(imgdat[pnt[1]][pnt[0]])
    lounds = []
    uounds = []
    for i in range(numpy.min((len(inbund), 15))):
        lounds.append(inbund[i][0][1])
        lounds.append(inbund[i][0][2])
        uounds.append(inbund[i][1][1])
        uounds.append(inbund[i][1][2])
    indata = [xvect, yvect, [], [], []]
    for i in range(1, numpy.min((len(pntinf), 15))):
        indata.append([])
        indata.append([])
    for i in range(len(xvect)):
        indata[2].append(pntinf[0][0])
        indata[3].append(pntinf[0][1])
        indata[4].append(pntinf[0][2])
        for j in range(5, len(indata), 2):
            indata[j].append(pntinf[int((j - 2) / 3)][0])
            indata[j + 1].append(pntinf[int((j - 2) / 3)][1])
    if len(pntinf) == 1:
        ptgess, fitgud = sciopt.curve_fit(ixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 2:
        ptgess, fitgud = sciopt.curve_fit(iixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 3:
        ptgess, fitgud = sciopt.curve_fit(iiixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 4:
        ptgess, fitgud = sciopt.curve_fit(ivxygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 5:
        ptgess, fitgud = sciopt.curve_fit(vxygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 6:
        ptgess, fitgud = sciopt.curve_fit(vixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 7:
        ptgess, fitgud = sciopt.curve_fit(viixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 8:
        ptgess, fitgud = sciopt.curve_fit(viiixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 9:
        ptgess, fitgud = sciopt.curve_fit(ixxygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 10:
        ptgess, fitgud = sciopt.curve_fit(xxygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 11:
        ptgess, fitgud = sciopt.curve_fit(xixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 12:
        ptgess, fitgud = sciopt.curve_fit(xiixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 13:
        ptgess, fitgud = sciopt.curve_fit(xiiixygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) == 14:
        ptgess, fitgud = sciopt.curve_fit(xivxygauss, indata, zvect, bounds = (lounds, uounds))
    elif len(pntinf) >= 15:
        ptgess, fitgud = sciopt.curve_fit(xvxygauss, indata, zvect, bounds = (lounds, uounds))
    return ptgess

if __name__ == '__main__':
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    if ftype == 'csv':
        imgdat = numpy.genfromtxt(flname, delimiter = ',')
    else:
        imgdat = imageio.imread(flname)
    print(emforsize(imgdat, tstnum = 20))
# =============================================================================
#     smthed = sandwhichsmooth.interpmerge(imgdat)
#     hostre = levelgraph.intensgraph()
#     hostre.bfsnncnst(smthed)
#     hostre.hubevl()
#     hostre.hubcmp(5.5, 50)
#     guesss = hostre.getcoords()
#     errlst = fitgausses(imgdat, guesss, [15, 15])
#     replic = interpimg(imgdat.shape, [errlst[:, 1], errlst[:, 2], errlst[:, 4]])
#     for pnt in errlst:
#         nearst = [int(numpy.round(pnt[1])), int(numpy.round(pnt[2]))]
#         if nearst[0] < 0:
#             nearst[0] = 0
#         elif nearst[0] > imgdat.shape[1] - 1:
#             nearst[0] = imgdat.shape[1] - 1
#         if nearst[1] < 0:
#             nearst[1] = 0
#         elif nearst[1] > imgdat.shape[0] - 1:
#             nearst[1] = imgdat.shape[0] - 1
#         replic[nearst[1]][nearst[0]] = replic[nearst[1]][nearst[0]] + pnt[0] * numpy.exp(-1 * ((nearst[0] - pnt[1])**2 + (nearst[1] - pnt[2])**2) / (2 * pnt[3]))
#         nxtarr = []
#         if nearst[0] > 0:
#             nxtarr.append([nearst[0] - 1, nearst[1]])
#         if nearst[0] < imgdat.shape[1] - 1:
#             nxtarr.append([nearst[0] + 1, nearst[1]])
#         if nearst[1] > 0:
#             nxtarr.append([nearst[0], nearst[1] - 1])
#         if nearst[1] < imgdat.shape[0] - 1:
#             nxtarr.append([nearst[0], nearst[1] + 1])
#         while len(nxtarr) > 0:
#             curpix = nxtarr.pop(0)
#             replic[curpix[1]][curpix[0]] = replic[curpix[1]][curpix[0]] + (pnt[0] * numpy.exp(-1 * ((curpix[0] - pnt[1])**2 + (curpix[1] - pnt[2])**2) / (2 * pnt[3])))
#             if numpy.exp(-1 * ((curpix[0] - pnt[1])**2 + (curpix[1] - pnt[2])**2) / (2 * pnt[3])) > .01:
#                 if curpix[0] > nearst[0] and curpix[1] <= nearst[1]:
#                     if curpix[1] > 0:
#                         nxtarr.append([curpix[0], curpix[1] - 1])
#                     if curpix[1] == nearst[1] and curpix[0] < imgdat.shape[1] - 1:
#                         nxtarr.append([curpix[0] + 1, curpix[1]])
#                 if curpix[0] <= nearst[0] and curpix[1] < nearst[1]:
#                     if curpix[0] > 0:
#                         nxtarr.append([curpix[0] - 1, curpix[1]])
#                     if curpix[0] == nearst[0] and curpix[1] > 0:
#                         nxtarr.append([curpix[0], curpix[1] - 1])
#                 if curpix[0] < nearst[0] and curpix[1] >= nearst[1]:
#                     if curpix[1] < imgdat.shape[0] - 1:
#                         nxtarr.append([curpix[0], curpix[1] + 1])
#                     if curpix[1] == nearst[1] and curpix[0] > 0:
#                         nxtarr.append([curpix[0] - 1, curpix[1]])
#                 if curpix[0] >= nearst[0] and curpix[1] > nearst[1]:
#                     if curpix[0] < imgdat.shape[1] - 1:
#                         nxtarr.append([curpix[0] + 1, curpix[1]])
#                     if curpix[0] == nearst[0] and curpix[1] < imgdat.shape[0] - 1:
#                         nxtarr.append([curpix[0], curpix[1] + 1])
#     print(errlst)
#     #numpy.savetxt(f'{fname}_c_original.csv', guesss, delimiter = ',')
#     #numpy.savetxt(f'{fname}_c_fit.csv', errlst, delimiter = ',')
#     mpl.figure()
#     mpable = mpl.imshow(imgdat)
#     mpl.colorbar(mpable)
#     mpl.figure()
#     mpable = mpl.imshow(replic)
#     mpl.colorbar(mpable)
#     mpl.figure()
#     mpable = mpl.imshow(imgdat - replic)
#     mpl.colorbar(mpable)
# =============================================================================
