# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:02:36 2019

@author: shini
"""

import numpy
import imageio
import levelgraph
import pointfitting
import h5py
import sandwhichsmooth
import matplotlib.pyplot as mpl
import scipy.interpolate as interp
import scipy.optimize as sciopt
from time import time

class atomnode:
    
    def __init__(self, sizpos):
        if len(sizpos) == 2:
            self.xypost = [sizpos[0], sizpos[1]]
            self.height = 0
            self.width = 1
            self.base = 0
        elif len(sizpos) == 3:
            self.height = sizpos[0]
            self.xypost = [sizpos[1], sizpos[2]]
            self.width = 1
            self.base = 0
        elif len(sizpos) == 4:
            self.height = sizpos[0]
            self.xypost = [sizpos[1], sizpos[2]]
            self.width = sizpos[3]
            self.base = 0
        else:
            self.height = sizpos[0]
            self.xypost = [sizpos[1], sizpos[2]]
            self.width = sizpos[3]
            self.base = sizpos[4]
        self.sizcat = 0
        self.neibrs = []
    
    def rootnehbrs(self):
        # Returns a list of all the root's neighbors and its neighbors' 
        # neighbors and it neighbors' neighbors' neighbors and so on
        neblst = []
        nxtlst = []
        for neibor in self.neibrs:
            nxtlst.append(neibor)
        while len(nxtlst) > 0:
            curneb = nxtlst.pop(0)
            neblst.append(curneb)
            for nebneb in curneb.neibrs:
                if not nebneb in neblst:
                    nxtlst.append(nebneb)
        return neblst
    
    def inarea(self, xycord):
        # Returns whether a point is within the domain of the node
        inside = True
        pntvec = [xycord[0] - self.xypost[0], xycord[1] - self.xypost[1]]
        for convec in self.neibrs:
            comvec = [convec[0].xypost[0] - self.xypost[0], convec[0].xypost[1] - self.xypost[1]]
            if (pntvec[0] * comvec[0] + pntvec[1] * comvec[1]) > (comvec[0]**2 + comvec[1]**2) / 2 and inside:
                inside = False
        return inside

class nodecubby:
    
    def __init__(self, points, ptspbn):
        self.ptbins = []
        self.relpts = []
        self.stpszs = []
        self.minims = []
        minbns = numpy.ceil(len(points) / ptspbn)
        d = 1
        dims = 2 #points[0].curpos.shape[0]
        while d**dims < minbns:
            d = d + 1
        maxims = []
        for i in range(dims):
            self.minims.append(points[0].xypost[i])
            maxims.append(points[0].xypost[i])
        for pnt in points:
            for i in range(dims):
                if pnt.xypost[i] < self.minims[i]:
                    self.minims[i] = pnt.xypost[i]
                elif pnt.xypost[i] > maxims[i]:
                    maxims[i] = pnt.xypost[i]
        self.minims = numpy.array(self.minims)
        maxims = numpy.array(maxims)
        self.stpszs = (maxims - self.minims) / d
        self.ptbins = [] #self.createndarray(d, dims)
        self.relpts = [] #self.createndarray(d, dims)\
        for i in range(d):
            ptrow = []
            rlrow = []
            for j in range(d):
                ptrow.append([])
                rlrow.append([])
            self.ptbins.append(ptrow)
            self.relpts.append(rlrow)
        for pnt in points:
            indarr = numpy.floor([(pnt.xypost[0] - self.minims[0]) / self.stpszs[0], (pnt.xypost[1] - self.minims[1]) / self.stpszs[1]])
            #print(indarr)
            if int(indarr[0]) == d:
                #curbuk = self.ptbins[int(indarr[0]) - 1]
                indarr[0] = indarr[0] - 1
            #else:
            #    curbuk = self.ptbins[int(indarr[0])]
            #for dm in range(1, dims):
            if indarr[1] == d:
            #        curbuk = curbuk[int(indarr[dm]) - 1]
                indarr[1] = indarr[1] - 1
            #    else:
            #        curbuk = curbuk[int(indarr[dm])]
            #curbuk.append(pnt)
            self.ptbins[int(indarr[0])][int(indarr[1])].append(pnt)
    
    def createndarray(self, length, depth):
        # This creates a hypersquare array of n-dimensions through recursion 
        # Python is kind of dumb and will assume you want the same base array 
        # l^n times without it. That's not fair, you will technically tell it 
        # to fill an l^n array with a list at the same address in each one, 
        # but that defeats the point of the n-dimensional matrix
        newarr = []
        for l in range(length):
            if depth == 1:
                newarr.append([])
            else:
                newarr.append(self.createndarray(length, depth - 1))
        return newarr
    
    def gatherchoices(self, centrn, numcom):
        # Rather than take in a single point and find the nodes to compare to 
        # for that node, instead preprocess the entire n-dimensional bucket 
        # space because each bucket will always draw from the same set of 
        # buckets. Perhaps instead just remember which buckets have been 
        # investigated before and only look for nodes if that bucket has not 
        # previously been visited. Will require a second set of node buckets 
        # that contain nodes from all relevant buckets. This should be nodes 
        # because a bucket of indices would not be time efficient even if it 
        # would be space efficient
        mainbk = numpy.array([int((centrn.xypost[0] - self.minims[0]) / self.stpszs[0]), int((centrn.xypost[1] - self.minims[1]) / self.stpszs[1])])
        #for i in range(mainbk.shape[0]):
        #    mainbk[i] = int(mainbk[i])
        if mainbk[0] == len(self.ptbins):
            mainbk[0] = mainbk[0] - 1
        if mainbk[1] == len(self.ptbins):
            mainbk[1] = mainbk[1] - 1
        # Next look through the bucket for all relevent comparison nodes, and 
        # if the bucket is empty, go looking for nodes
        relbuk = self.relpts[int(mainbk[0])][int(mainbk[1])]
        #for ind in mainbk:
        #    relbuk = relbuk[int(ind)]
        if len(relbuk) == 0:
            nxtbks = []
            for i in range(mainbk.shape[0]):
                for j in range(i, mainbk.shape[0]):
                    avec = numpy.zeros(len(self.stpszs))
                    avec[i] = 1
                    avec[j] = 1
                    nxtbks.append([numcom, avec])
                    avec = numpy.zeros(len(self.stpszs))
                    avec[i] = 1
                    avec[j] = -1
                    nxtbks.append([numcom, avec])
                    if i != j:
                        avec = numpy.zeros(len(self.stpszs))
                        avec[i] = -1
                        avec[j] = 1
                        nxtbks.append([numcom, avec])
                        avec = numpy.zeros(len(self.stpszs))
                        avec[i] = -1
                        avec[j] = -1
                        nxtbks.append([numcom, avec])
            ndstpl = [mainbk]
            while len(nxtbks) > 0:
                curoff = nxtbks.pop(0)
                curcor = curoff[1] + mainbk
                #curbuk = self.ptbins
                #inrang = True
                #i = 0
                # Retrieve the bucket in question if it is in the array
                #while i < curcor.shape[0] and inrang:
                #    if curcor[i] >= 0 and curcor[i] < len(curbuk):
                #        curbuk = curbuk[int(curcor[i])]
                #        i = i + 1
                #    else:
                #        inrang = False
                # If it is in the array, add it to the list of buckets to draw nodes 
                # from, then find which buckets to draw nodes from next
                if curcor[0] >= 0 and curcor[1] >= 0 and curcor[0] < len(self.ptbins) and curcor[1] < len(self.ptbins): #inrang
                    curbuk = self.ptbins[int(curcor[0])][int(curcor[1])]
                    ndstpl.append(curcor)
                    if len(curbuk) < curoff[0]:
                        u = -1
                        v = -1
                        # Identify the non-zero elements of the offset vectors (there 
                        # should only be two)
                        for ind in range(len(curoff[1])):
                            if curoff[1][ind] != 0:
                                if u == -1:
                                    u = ind
                                else:
                                    v = ind
                        # If there is only one non-zero element, increment the index 
                        # by one and add all other dimensional vector unit 
                        # combinations +/- 1
                        if v == -1:
                            if curoff[1][u] > 0:
                                curoff[1][u] = curoff[1][u] + 1
                            else:
                                curoff[1][u] = curoff[1][u] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1]])
                            for v in range(curoff[1].shape[0]):
                                if v != u:
                                    exoffs = numpy.zeros(curoff[1].shape)
                                    exoffs[v] = exoffs[v] + 1
                                    nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1] + exoffs])
                                    nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1] - exoffs])
                        # If there are two and their magnitudes are not equal, then 
                        # increment the larger one and add the incremented and non-
                        # incremented minor offsets
                        elif abs(curoff[1][u]) > abs(curoff[1][v]):
                            if curoff[1][u] > 0:
                                curoff[1][u] = curoff[1][u] + 1
                            else:
                                curoff[1][u] = curoff[1][u] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1]])
                            if curoff[1][v] > 0:
                                curoff[1][v] = curoff[1][v] + 1
                            else:
                                curoff[1][v] = curoff[1][v] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1]])
                        elif abs(curoff[1][u]) < abs(curoff[1][v]):
                            if curoff[1][v] > 0:
                                curoff[1][v] = curoff[1][v] + 1
                            else:
                                curoff[1][v] = curoff[1][v] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1]])
                            if curoff[1][u] > 0:
                                curoff[1][u] = curoff[1][u] + 1
                            else:
                                curoff[1][u] = curoff[1][u] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1]])
                        # If there are two and their magnitudes are equal, then add 
                        # one incremented but not the other, vice versa, and both 
                        # incremented
                        else:
                            uoffst = numpy.zeros(curoff[1].shape)
                            voffst = numpy.zeros(curoff[1].shape)
                            if curoff[1][u] > 0:
                                uoffst[u] = uoffst[u] + 1
                            else:
                                uoffst[u] = uoffst[u] - 1
                            if curoff[1][v] > 0:
                                voffst[v] = voffst[v] + 1
                            else:
                                voffst[v] = voffst[v] - 1
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1] + uoffst])
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1] + voffst])
                            nxtbks.insert(0, [curoff[0] - len(curbuk), curoff[1] + uoffst + voffst])
            for bukset in ndstpl:
                tolbuk = self.ptbins
                for ind in bukset:
                    tolbuk = tolbuk[int(ind)]
                for node in tolbuk:
                    if not node in relbuk:
                        relbuk.append(node)
        # Once the relevant node bucket is filled, return all relevant nodes
        return relbuk

class lonelattice:
    
    def __init__(self):
        self.nodlst = []
    
    def connect(self, nodea, nodeb):
        #edglen = ((nodea.xypost[0] - nodeb.xypost[0])**2 + (nodea.xypost[1] - nodeb.xypost[1])**2)**.5
        cosang = (nodeb.xypost[0] - nodea.xypost[0]) / ((nodeb.xypost[0] - nodea.xypost[0])**2 + (nodeb.xypost[1] - nodea.xypost[1])**2)**.5
        if nodeb.xypost[1] - nodea.xypost[1] >= 0:
            cosang = numpy.arccos(cosang)
        else:
            cosang = 2 * numpy.pi - numpy.arccos(cosang)
        i = 0
        srchng = True
        while i < len(nodea.neibrs) and srchng:
            if cosang > nodea.neibrs[i][1]:
                i = i + 1
            else:
                srchng = False
        nodea.neibrs.insert(i, [nodeb, cosang])
        if cosang < numpy.pi:
            angcos = numpy.pi + cosang
        else:
            angcos = cosang - numpy.pi
        i = 0
        srchng = True
        while i < len(nodeb.neibrs) and srchng:
            if angcos > nodeb.neibrs[i][1]:
                i = i + 1
            else:
                srchng = False
        nodeb.neibrs.insert(i, [nodea, angcos])
    
    def disconnect(self, nodea, nodeb):
        i = 0
        while i < len(nodeb.neibrs):
            if nodeb.neibrs[i][0] == nodea:
                nodeb.neibrs.pop(i)
            else:
                i = i + 1
        i = 0
        while i < len(nodea.neibrs):
            if nodea.neibrs[i][0] == nodeb:
                nodea.neibrs.pop(i)
            else:
                i = i + 1
    
# =============================================================================
#     def minspanbuild(self, pntlst):
#         nodarr = []
#         for pnt in pntlst:
#             nodarr.append(atomnode(pnt))
#         edglst = []
#         for k in range(len(pntlst) - 1):
#             print(k)
#             for l in range(k + 1, len(pntlst)):
#                 curedg = [nodarr[k], nodarr[l], ((nodarr[k].xypost[0] - nodarr[l].xypost[0])**2 + (nodarr[k].xypost[1] - nodarr[l].xypost[1])**2)**.5]
#                 m = 0
#                 srchng = True
#                 while srchng:
#                     if m >= len(edglst):
#                         srchng = False
#                     elif curedg[2] <= edglst[m][2]:
#                         srchng = False
#                     else:
#                         m = m + 1
#                 edglst.insert(m, curedg) #flag for implementation
#         while len(edglst) > 0:
#             curedg = edglst.pop(0)
#             if not curedg[0] in curedg[1].rootnehbrs():
#                 self.connect(curedg[0], curedg[1])
#             if not curedg[0] in self.nodlst:
#                 self.nodlst.append(curedg[0])
#             if not curedg[1] in self.nodlst:
#                 self.nodlst.append(curedg[1])
#     
#     def densespanbuild(self, pntlst, trshld = 1.35):
#         # This will not create a minimum spanning tree, but will instead 
#         # create a densely connected graph that is guaranteed to have at least 
#         # one route from one node to any other on the graph.
#         self.nodlst = []
#         for pnt in pntlst:
#             self.nodlst.append(atomnode(pnt))
#         potneb = []
#         # First, for each node, find the nearest node and all other nodes 
#         # which are within a certain distance of that shortest connection, 
#         # then connect them
#         print('Connecting nodes')
#         for k in range(len(self.nodlst) - 1):
#             if k % 1000 == 0:
#                 print(f'The graph is {100 * k / len(self.nodlst)}% connected.')
#             potneb = [[self.nodlst[k + 1], ((self.nodlst[k].xypost[0] - self.nodlst[k + 1].xypost[0])**2 + (self.nodlst[k].xypost[1] - self.nodlst[k + 1].xypost[1])**2)**.5]]
#             shtdst = potneb[0][1]
#             for l in range(k + 2, len(self.nodlst)):
#                 nebdst = ((self.nodlst[k].xypost[0] - self.nodlst[l].xypost[0])**2 + (self.nodlst[k].xypost[1] - self.nodlst[l].xypost[1])**2)**.5
#                 if nebdst <= shtdst * trshld:
#                     if nebdst < shtdst:
#                         shtdst = nebdst
#                         nebitr = 0
#                         while nebitr < len(potneb):
#                             if potneb[nebitr][1] > shtdst * trshld:
#                                 potneb.pop(nebitr)
#                             else:
#                                 nebitr = nebitr + 1
#                     potneb.append([self.nodlst[l], nebdst])
#             for neb in potneb:
#                 self.connect(self.nodlst[k], neb[0])
#         edglst = []
#         vstlst = []
#         # Next, look over the edges, and disconnect any that are crossing 
#         # other edges
#         print('Compiling edges.')
#         for node in self.nodlst:
#             vstlst.append(node)
#             for neb in node.neibrs:
#                 if not neb[0] in vstlst:
#                     edglst.append([node, neb[0]])
#         print('Pruning nodes.')
#         retcon = []
#         for k in range(len(edglst) - 1):
#             # This will be removing edges, but I don't want to change the size 
#             # of the list while iterating over it, so I need to check that 
#             # the connection still exists when I get to it.
#             if k % 1000 == 0:
#                 print(f'{100 * k / len(edglst)}% of edges have been checked.')
#             exists = False
#             for neb in edglst[k][0].neibrs:
#                 if neb[0] == edglst[k][1]:
#                     exists = True
#             if exists:
#                 for l in range(k + 1, len(edglst)):
#                     exists = False
#                     for neb in edglst[l][0].neibrs:
#                         if neb[0] == edglst[l][1]:
#                             exists = True
#                     if exists:
#                         # The way this works is that I take the vector defined 
#                         # by one edge and transform the two points defining 
#                         # the other edge into a new coordinate system where 
#                         # where the first edge is the x-axis. If the two 
#                         # points lie on opposite sides of the x-axis then it  
#                         # is possible they are overlapping. One must first 
#                         # check by performing the same transformation with the 
#                         # edges swapped, otherwise it may be that one vector 
#                         # is pointing to the midpoint of the other.
#                         basvec = [edglst[k][1].xypost[0] - edglst[k][0].xypost[0], edglst[k][1].xypost[1] - edglst[k][0].xypost[1]]
#                         vrtone = [edglst[l][0].xypost[0] - edglst[k][0].xypost[0], edglst[l][0].xypost[1] - edglst[k][0].xypost[1]]
#                         vrttwo = [edglst[l][1].xypost[0] - edglst[k][0].xypost[0], edglst[l][1].xypost[1] - edglst[k][0].xypost[1]]
#                         magone = basvec[0]**2 + basvec[1]**2
#                         # The matrix transformation is
#                         # [bx  by][v1x  v2x] = [p1x  p2x]
#                         # [by -bx][v1y  v2y]   [p1y  p2y]
#                         # but since this only cares about the y component, it 
#                         # will not calculate the x component to save time
#                         pntone = basvec[1] * vrtone[0] - basvec[0] * vrtone[1]
#                         pnttwo = basvec[1] * vrttwo[0] - basvec[0] * vrttwo[1]
#                         if (pntone < 0 and pnttwo > 0) or (pntone > 0 and pnttwo < 0):
#                             basvec = [edglst[l][1].xypost[0] - edglst[l][0].xypost[0], edglst[l][1].xypost[1] - edglst[l][0].xypost[1]]
#                             vrtone = [edglst[k][0].xypost[0] - edglst[l][0].xypost[0], edglst[k][0].xypost[1] - edglst[l][0].xypost[1]]
#                             vrttwo = [edglst[k][1].xypost[0] - edglst[l][0].xypost[0], edglst[k][1].xypost[1] - edglst[l][0].xypost[1]]
#                             magtwo = basvec[0]**2 + basvec[1]**2
#                             pntone = basvec[1] * vrtone[0] - basvec[0] * vrtone[1]
#                             pnttwo = basvec[1] * vrttwo[0] - basvec[0] * vrttwo[1]
#                             if (pntone < 0 and pnttwo > 0) or (pntone > 0 and pnttwo < 0):
#                                 if magone <= magtwo:
#                                     self.disconnect(edglst[l][0], edglst[l][1])
#                                     if len(edglst[l][0].neibrs) == 0:
#                                         retcon.append(edglst[k][0])
#                                     if len(edglst[l][1].neibrs) == 0:
#                                         retcon.append(edglst[k][0])
#                                 else:
#                                     self.disconnect(edglst[k][0], edglst[k][1])
#                                     if len(edglst[k][0].neibrs) == 0:
#                                         retcon.append(edglst[k][0])
#                                     if len(edglst[k][1].neibrs) == 0:
#                                         retcon.append(edglst[k][0])
#         # Finally, if you isolated any nodes from the graph, repeat the 
#         # initial step but allow the algorithm to search across the entire 
#         # set of nodes.
#         for k in range(len(retcon)):
#             print(f'{k / len(retcon)}% of the disconnected nodes have been reconnected.')
#             if self.nodlst[0] == retcon[k]:
#                 potneb = [[self.nodlst[1], ((self.nodlst[1].xypost[0] - retcon[k].xypost[0])**2 + (self.nodlst[1].xypost[1] - retcon[k].xypost[1])**2)**.5]]
#                 shtdst = potneb[0][1]
#                 for l in range(2, len(self.nodlst)):
#                     nebdst = ((retcon[k].xypost[0] - self.nodlst[l].xypost[0])**2 + (retcon[k].xypost[1] - self.nodlst[l].xypost[1])**2)**.5
#                     if nebdst <= shtdst * trshld:
#                         if nebdst < shtdst:
#                             shtdst = nebdst
#                             nebitr = 0
#                             while nebitr < len(potneb):
#                                 if potneb[nebitr][1] > shtdst * trshld:
#                                     potneb.pop(nebitr)
#                                 else:
#                                     nebitr = nebitr + 1
#                         potneb.append([self.nodlst[l], nebdst])
#                 for neb in potneb:
#                     self.connect(retcon, neb[0])
#             else:
#                 potneb = [[self.nodlst[0], ((self.nodlst[0].xypost[0] - retcon[k].xypost[0])**2 + (self.nodlst[0].xypost[1] - retcon[k].xypost[1])**2)**.5]]
#                 shtdst = potneb[0][1]
#                 for l in range(1, len(self.nodlst)):
#                     if not retcon[k] == self.nodlst[l]:
#                         nebdst = ((retcon[k].xypost[0] - self.nodlst[l].xypost[0])**2 + (retcon[k].xypost[1] - self.nodlst[l].xypost[1])**2)**.5
#                         if nebdst <= shtdst * trshld:
#                             if nebdst < shtdst:
#                                 shtdst = nebdst
#                                 nebitr = 0
#                                 while nebitr < len(potneb):
#                                     if potneb[nebitr][1] > shtdst * trshld:
#                                         potneb.pop(nebitr)
#                                     else:
#                                         nebitr = nebitr + 1
#                             potneb.append([self.nodlst[l], nebdst])
#                 for neb in potneb:
#                     self.connect(retcon[k], neb[0])
#     
#     def boxspanbuild(self, pntlst, trshld = 75):
#         # This will attempt to create a fully connected graph by looking at 
#         # the domain of each point and make connections which have a 
#         # significant impact on the domain and create a domain such that no 
#         # other point is within that domain
#         for pnt in pntlst:
#             self.nodlst.append(atomnode(pnt))
#         devval = numpy.sin(75 * numpy.pi / 180) / numpy.sin(52.5 * numpy.pi / 180)
#         for k in range(len(self.nodlst) - 1):
#             if k % 100 == 0:
#                 print(f'The graph is {100 * k / len(self.nodlst)}% connected.')
#             curnod = self.nodlst[k]
#             for l in range(len(self.nodlst)):
#                 if not k == l:
#                     potnod = self.nodlst[l]
#                     edgvec = [potnod.xypost[0] - curnod.xypost[0], potnod.xypost[1] - curnod.xypost[1]]
#                     edgmag = (edgvec[0]**2 + edgvec[1]**2)**.5
#                     isnehb = True
#                     elimar = []
#                     arelim = []
#                     for brdpnt in curnod.neibrs:
#                         brdvec = [brdpnt[0].xypost[0] - curnod.xypost[0], brdpnt[0].xypost[1] - curnod.xypost[1]]
#                         dotpro = edgvec[0] * brdvec[0] + edgvec[1] * brdvec[1]
#                         if dotpro > 0:
#                             brdmag = (brdvec[0]**2 + brdvec[1]**2)**.5
#                             cosang = dotpro / (edgmag * brdmag)
#                             if edgmag * cosang > brdmag * (cosang + devval) / (1 + devval):
#                                 isnehb = False
#                             if brdmag * cosang > edgmag * (cosang + devval) / (1 + devval):
#                                 elimar.append(brdpnt)
#                     for fncpnt in potnod.neibrs:
#                         fncvec = [fncpnt[0].xypost[0] - potnod.xypost[0], fncpnt[0].xypost[1] - potnod.xypost[1]]
#                         prodot = -1 * edgvec[0] * fncvec[0] - edgvec[1] * fncvec[1]
#                         if prodot > 0:
#                             fncmag = (fncvec[0]**2 + fncvec[1]**2)**.5
#                             angcos = prodot / (edgmag * fncmag)
#                             if edgmag * angcos > fncmag * (angcos + devval) / (1 + devval):
#                                 isnehb = False
#                             if fncmag * angcos > edgmag * (angcos + devval) / (1 + devval):
#                                 arelim.append(fncpnt)
#                     if isnehb:
#                         for oldnod in elimar:
#                             self.disconnect(curnod, oldnod)
#                         for nodold in arelim:
#                             self.disconnect(curnod, nodold)
#                         self.connect(curnod, potnod)
#     
#     def gndupbuild(self, pntlst, trshld = 1):
#         for pnt in pntlst:
#             self.nodlst.append(atomnode(pnt))
#         for k in range(len(self.nodlst)):
#             pointk = self.nodlst[k]
#             for l in range(len(self.nodlst)):
#                 pointl = self.nodlst[l]
#                 newedg = [pointl.xypost[0] - pointk.xypost[0], pointl.xypost[1] - pointk.xypost[1]]
#                 tooadd = True
#                 elimnk = []
#                 elimnl = []
#                 for kfence in pointk.neibrs:
#                     if trshld * (newedg[0] * (kfence[0].xypost[0] - pointk.xypost[0]) + newedg[1] * (kfence[0].xypost[1] - pointk.xypost[1])) > ((kfence[0].xypost[0] - pointk.xypost[0])**2 + (kfence[0].xypost[1] - pointk.xypost[1])**2) / 2:
#                         tooadd = False
#                     if trshld * (newedg[0] * (kfence[0].xypost[0] - pointk.xypost[0]) + newedg[1] * (kfence[0].xypost[1] - pointk.xypost[1])) > (newedg[0]**2 + newedg[1]**2) / 2:
#                         elimnk.append(kfence)
#                 for lfence in pointl.neibrs:
#                     if trshld * (-1 * newedg[0] * (lfence[0].xypost[0] - pointl.xypost[0]) - newedg[1] * (lfence[0].xypost[1] - pointl.xypost[1])) > ((lfence[0].xypost[0] - pointl.xypost[0])**2 + (lfence[0].xypost[1] - pointl.xypost[1])**2) / 2:
#                         tooadd = False
#                     if trshld * (-1 * newedg[0] * (lfence[0].xypost[0] - pointl.xypost[0]) - newedg[1] * (lfence[0].xypost[1] - pointl.xypost[1])) > (newedg[0]**2 + newedg[1]**2) / 2:
#                         elimnl.append(lfence)
#                 if tooadd:
#                     for toremv in elimnk:
#                         self.disconnect(toremv, pointk)
#                     for toremv in elimnl:
#                         self.disconnect(toremv, pointl)
#                     self.connect(pointk, pointl)
#     
#     def qntbuild(self, pntlst, trshld = 1.3):
#         for pnt in pntlst:
#             self.nodlst.append(atomnode(pnt))
#         for k in range(len(self.nodlst)):
#             pointk = self.nodlst[k]
#             if k % 100 == 0:
#                 print(f'The graph is {100 * k / len(self.nodlst)}% connected')
#             for l in range(len(self.nodlst)):
#                 pointl = self.nodlst[l]
#                 notneb = True
#                 if k == l:
#                     notneb = False
#                 elif pointk.xypost[0] == pointl.xypost[0] and pointk.xypost[1] == pointl.xypost[1]:
#                     notneb = False
#                 else:
#                     for neb in pointk.neibrs:
#                         if pointl == neb[0]:
#                             notneb = False
#                 if notneb:
#                     self.connect(pointk, pointl)
#                     if len(pointk.neibrs) > 2:
#                         toremv = []
#                         for i in range(len(pointk.neibrs)):
#                             pointa = pointk.neibrs[i][0]
#                             if i == len(pointk.neibrs) - 1:
#                                 pointb = pointk.neibrs[0][0]
#                                 pointc = pointk.neibrs[1][0]
#                             elif i == len(pointk.neibrs) - 2:
#                                 pointb = pointk.neibrs[len(pointk.neibrs) - 1][0]
#                                 pointc = pointk.neibrs[0][0]
#                             else:
#                                 pointb = pointk.neibrs[i + 1][0]
#                                 pointc = pointk.neibrs[i + 2][0]
#                             bsvecc = [pointc.xypost[0] - pointa.xypost[0], pointc.xypost[1] - pointa.xypost[1]]
#                             rlveck = [pointk.xypost[0] - pointa.xypost[0], pointk.xypost[1] - pointa.xypost[1]]
#                             rlvecb = [pointb.xypost[0] - pointa.xypost[0], pointb.xypost[1] - pointa.xypost[1]]
#                             bsvecb = [pointb.xypost[0] - pointk.xypost[0], pointb.xypost[1] - pointk.xypost[1]]
#                             rlveca = [pointa.xypost[0] - pointk.xypost[0], pointa.xypost[1] - pointk.xypost[1]]
#                             rlvecc = [pointc.xypost[0] - pointk.xypost[0], pointc.xypost[1] - pointk.xypost[1]]
#                             trnskb = [bsvecc[1] * rlveck[0] - bsvecc[0] * rlveck[1], bsvecc[1] * rlvecb[0] - bsvecc[0] * rlvecb[1]]
#                             trnsca = [bsvecb[1] * rlvecc[0] - bsvecb[0] * rlvecc[1], bsvecb[1] * rlveca[0] - bsvecb[0] * rlveca[1]]
#                             if (trnskb[0] > 0 and trnskb[1] < 0) or (trnskb[0] < 0 and trnskb[1] > 0) or trnsca[0] == 0 or trnsca[1] == 0:
#                                 if (bsvecb[0]**2 + bsvecb[1]**2) * trshld > (bsvecc[0]**2 + bsvecc[1]**2):
#                                     toremv.append(pointb)
#                         while len(toremv) > 0:
#                             while len(toremv) > 0:
#                                 badneb = toremv.pop(0)
#                                 self.disconnect(pointk, badneb)
#                             if len(pointk.neibrs) > 2:
#                                 for i in range(len(pointk.neibrs)):
#                                     pointa = pointk.neibrs[i][0]
#                                     if i == len(pointk.neibrs) - 1:
#                                         pointb = pointk.neibrs[0][0]
#                                         pointc = pointk.neibrs[1][0]
#                                     elif i == len(pointk.neibrs) - 2:
#                                         pointb = pointk.neibrs[len(pointk.neibrs) - 1][0]
#                                         pointc = pointk.neibrs[0][0]
#                                     else:
#                                         pointb = pointk.neibrs[i + 1][0]
#                                         pointc = pointk.neibrs[i + 2][0]
#                                     bsvecc = [pointc.xypost[0] - pointa.xypost[0], pointc.xypost[1] - pointa.xypost[1]]
#                                     rlveck = [pointk.xypost[0] - pointa.xypost[0], pointk.xypost[1] - pointa.xypost[1]]
#                                     rlvecb = [pointb.xypost[0] - pointa.xypost[0], pointb.xypost[1] - pointa.xypost[1]]
#                                     bsvecb = [pointb.xypost[0] - pointk.xypost[0], pointb.xypost[1] - pointk.xypost[1]]
#                                     rlveca = [pointa.xypost[0] - pointk.xypost[0], pointa.xypost[1] - pointk.xypost[1]]
#                                     rlvecc = [pointc.xypost[0] - pointk.xypost[0], pointc.xypost[1] - pointk.xypost[1]]
#                                     trnskb = [bsvecc[1] * rlveck[0] - bsvecc[0] * rlveck[1], bsvecc[1] * rlvecb[0] - bsvecc[0] * rlvecb[1]]
#                                     trnsca = [bsvecb[1] * rlvecc[0] - bsvecb[0] * rlvecc[1], bsvecb[1] * rlveca[0] - bsvecb[0] * rlveca[1]]
#                                     if (trnskb[0] > 0 and trnskb[1] < 0) or (trnskb[0] < 0 and trnskb[1] > 0) or trnsca[0] == 0 or trnsca[1] == 0:
#                                         # A note in case I get weird 
#                                         # connections/disconnections later: I 
#                                         # might want to make separate cases 
#                                         # for parallel points 
#                                         # (trnsca[0/1] == 0) because currently 
#                                         # comparing the distance to the point 
#                                         # to the distance between neighbors 
#                                         # when I should be comparing the 
#                                         # parallel points. This also applies 
#                                         # to the above example, so be sure to 
#                                         # change that as well if this changes
#                                         if (bsvecb[0]**2 + bsvecb[1]**2) * trshld > (bsvecc[0]**2 + bsvecc[1]**2):
#                                             toremv.append(pointb)
# =============================================================================
    
    def djkbuild(self, pntlst, thresh = 0.0525588331227637):
        # First, generate the nodes
        for pnt in pntlst:
            self.nodlst.append(atomnode(pnt))
        nodfil = nodecubby(self.nodlst, 20) #flag
        for k in range(len(self.nodlst)):
            pointk = self.nodlst[k]
            if k % 100 == 0:
                print(f'The graph is {100 * k / len(self.nodlst):.2f}% connected')
            potnbs = []
            sublst = nodfil.gatherchoices(pointk, 10)
            for l in range(len(sublst)): #len(self.nodlst)
                pointl = sublst[l]
                notneb = True
                # Because this cycles over all nodes, you must avoid comparing 
                # nodes with themselves or their neighbors
                if pointk.xypost[0] == pointl.xypost[0] and pointk.xypost[1] == pointl.xypost[1]:
                    notneb = False
                else:
                    for neb in pointk.neibrs:
                        if pointl == neb[0]:
                            notneb = False
                i = 0
                # Compare with all current neighbors. These should be fixed, 
                # since all preexisting neighbors should be to nodes whose 
                # optimal connections have been found, so there is no need to 
                # check for conflicts in the other direction
                #distlk = (pointl.xypost[0] - pointk.xypost[0])**2 + (pointl.xypost[1] - pointk.xypost[1])**2
                #vectlk = numpy.array([pointl.xypost[0] - pointk.xypost[0], pointl.xypost[1] - pointk.xypost[1]])
                while i < len(pointk.neibrs) and notneb:
                    vectki = numpy.array([pointk.xypost[0] - pointk.neibrs[i][0].xypost[0], pointk.xypost[1] - pointk.neibrs[i][0].xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - pointk.neibrs[i][0].xypost[0], pointl.xypost[1] - pointk.neibrs[i][0].xypost[1]])
                    #if distlk >= (pointk.neibrs[i][0].xypost[0] - pointk.xypost[0])**2 + (pointk.neibrs[i][0].xypost[1] - pointk.xypost[1])**2 + (pointl.xypost[0] - pointk.neibrs[i][0].xypost[0])**2 + (pointl.xypost[1] - pointk.neibrs[i][0].xypost[1])**2:
                    if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                        notneb = False
                    else:
                        i = i + 1
                # Compare with all current potential neighbors. If there is a 
                # conflict in the other direction (node l would exclude a 
                # potential neighbor) - which should not happed because I 
                # should compare against all other nodes to become a proper 
                # potential neighbor, but whatever - then remove the offending 
                # neighbor candidate
                i = 0
                while i < len(potnbs) and notneb:
                    distik = (potnbs[i].xypost[0] - pointk.xypost[0])**2 + (potnbs[i].xypost[1] - pointk.xypost[1])**2
                    distli = (pointl.xypost[0] - potnbs[i].xypost[0])**2 + (pointl.xypost[1] - potnbs[i].xypost[1])**2
                    vectki = numpy.array([pointk.xypost[0] - potnbs[i].xypost[0], pointk.xypost[1] - potnbs[i].xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - potnbs[i].xypost[0], pointl.xypost[1] - potnbs[i].xypost[1]])
                    vectkl = numpy.array([pointk.xypost[0] - pointl.xypost[0], pointk.xypost[1] - pointl.xypost[1]])
                    #if distlk >= distik + distli:
                    if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                        notneb = False
                    #elif distik >= distlk + distli:
                    elif thresh >= (vectkl[0] * -1 * vectli[0] + vectkl[1] * -1 * vectli[1]) / (numpy.sum(vectkl**2)**.5 * numpy.sum(vectli**2)**.5):
                        potnbs.remove(potnbs[i])
                    else:
                        i = i + 1
                # Now compare against all other nodes. Make sure to skip over 
                # the central node and the compared node, and allow for 
                # duplicates
                i = 0
                while i < len(sublst) and notneb:
                    pointi = sublst[i]
                    distik = (pointi.xypost[0] - pointk.xypost[0])**2 + (pointi.xypost[1] - pointk.xypost[1])**2
                    distli = (pointl.xypost[0] - pointi.xypost[0])**2 + (pointl.xypost[1] - pointi.xypost[1])**2
                    vectki = numpy.array([pointk.xypost[0] - pointi.xypost[0], pointk.xypost[1] - pointi.xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - pointi.xypost[0], pointl.xypost[1] - pointi.xypost[1]])
                    if (distik != 0) and (distli != 0):
                        if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                            notneb = False
                        else:
                            i = i + 1
                    else:
                        i = i + 1
                if notneb:
                    potnbs.append(pointl)
                    # Connect the two points
# =============================================================================
#                     self.connect(pointk, pointl)
#                     toremv = []
#                     # Compare the current neighbors with one another. Remove 
#                     # any conflicts
#                     for m in range(len(pointk.neibrs) - 1):
#                         for n in range(m + 1, len(pointk.neibrs)):
#                             distmk = (pointk.neibrs[m][0].xypost[0] - pointk.xypost[0])**2 + (pointk.neibrs[m][0].xypost[1] - pointk.xypost[1])**2
#                             distnk = (pointk.neibrs[n][0].xypost[0] - pointk.xypost[0])**2 + (pointk.neibrs[n][0].xypost[1] - pointk.xypost[1])**2
#                             distmn = (pointk.neibrs[n][0].xypost[0] - pointk.neibrs[m][0].xypost[0])**2 + (pointk.neibrs[n][0].xypost[1] - pointk.neibrs[m][0].xypost[1])**2
#                             if distnk * slack**2 > distmk + distmn:
#                                 toremv.append(pointk.neibrs[n][0])
#                             if distmk * slack**2 > distnk + distmn:
#                                 toremv.append(pointk.neibrs[m][0])
#                     while len(toremv) > 0:
#                         for badneb in toremv:
#                             self.disconnect(pointk, badneb)
#                         toremv = []
#                         for m in range(len(pointk.neibrs) - 1):
#                             for n in range(m + 1, len(pointk.neibrs)):
#                                 distmk = (pointk.neibrs[m][0].xypost[0] - pointk.xypost[0])**2 + (pointk.neibrs[m][0].xypost[1] - pointk.xypost[1])**2
#                                 distnk = (pointk.neibrs[n][0].xypost[0] - pointk.xypost[0])**2 + (pointk.neibrs[n][0].xypost[1] - pointk.xypost[1])**2
#                                 distmn = (pointk.neibrs[n][0].xypost[0] - pointk.neibrs[m][0].xypost[0])**2 + (pointk.neibrs[n][0].xypost[1] - pointk.neibrs[m][0].xypost[1])**2
#                                 if distnk * slack**2 > distmk + distmn:
#                                     toremv.append(pointk.neibrs[n][0])
#                                 if distmk * slack**2 > distnk + distmn:
#                                     toremv.append(pointk.neibrs[m][0])
# =============================================================================
            for truneb in potnbs:
                self.connect(pointk, truneb)
    
    def redjkbuild(self, newpts = [], thresh = 0.0525588331227637):
        # This is for reconnecting the graph after it has been refined or had
        # nodes removed. Will also be able to add new nodes to the graph
        # First, reset connections between and generate new nodes from the 
        # additional points
        for node in self.nodlst:
            node.neibrs = []
        for pnt in newpts:
            self.nodlst.append(atomnode(pnt))
        nodfil = nodecubby(self.nodlst, 20) #flag
        for k in range(len(self.nodlst)):
            pointk = self.nodlst[k]
            if k % 100 == 0:
                print(f'The graph is {100 * k / len(self.nodlst):.2f}% connected')
            potnbs = []
            sublst = nodfil.gatherchoices(pointk, 10)
            for l in range(len(sublst)): #len(self.nodlst)
                pointl = sublst[l]
                notneb = True
                # Because this cycles over all nodes, you must avoid comparing 
                # nodes with themselves or their neighbors
                if pointk.xypost[0] == pointl.xypost[0] and pointk.xypost[1] == pointl.xypost[1]:
                    notneb = False
                else:
                    for neb in pointk.neibrs:
                        if pointl == neb[0]:
                            notneb = False
                i = 0
                # Compare with all current neighbors. These should be fixed, 
                # since all preexisting neighbors should be to nodes whose 
                # optimal connections have been found, so there is no need to 
                # check for conflicts in the other direction
                while i < len(pointk.neibrs) and notneb:
                    vectki = numpy.array([pointk.xypost[0] - pointk.neibrs[i][0].xypost[0], pointk.xypost[1] - pointk.neibrs[i][0].xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - pointk.neibrs[i][0].xypost[0], pointl.xypost[1] - pointk.neibrs[i][0].xypost[1]])
                    if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                        notneb = False
                    else:
                        i = i + 1
                # Compare with all current potential neighbors. If there is a 
                # conflict in the other direction (node l would exclude a 
                # potential neighbor) - which should not happed because I 
                # should compare against all other nodes to become a proper 
                # potential neighbor, but whatever - then remove the offending 
                # neighbor candidate
                i = 0
                while i < len(potnbs) and notneb:
                    distik = (potnbs[i].xypost[0] - pointk.xypost[0])**2 + (potnbs[i].xypost[1] - pointk.xypost[1])**2
                    distli = (pointl.xypost[0] - potnbs[i].xypost[0])**2 + (pointl.xypost[1] - potnbs[i].xypost[1])**2
                    vectki = numpy.array([pointk.xypost[0] - potnbs[i].xypost[0], pointk.xypost[1] - potnbs[i].xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - potnbs[i].xypost[0], pointl.xypost[1] - potnbs[i].xypost[1]])
                    vectkl = numpy.array([pointk.xypost[0] - pointl.xypost[0], pointk.xypost[1] - pointl.xypost[1]])
                    if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                        notneb = False
                    elif thresh >= (vectkl[0] * -1 * vectli[0] + vectkl[1] * -1 * vectli[1]) / (numpy.sum(vectkl**2)**.5 * numpy.sum(vectli**2)**.5):
                        potnbs.remove(potnbs[i])
                    else:
                        i = i + 1
                # Now compare against all other nodes. Make sure to skip over 
                # the central node and the compared node, and allow for 
                # duplicates
                i = 0
                while i < len(sublst) and notneb:
                    pointi = sublst[i]
                    distik = (pointi.xypost[0] - pointk.xypost[0])**2 + (pointi.xypost[1] - pointk.xypost[1])**2
                    distli = (pointl.xypost[0] - pointi.xypost[0])**2 + (pointl.xypost[1] - pointi.xypost[1])**2
                    vectki = numpy.array([pointk.xypost[0] - pointi.xypost[0], pointk.xypost[1] - pointi.xypost[1]])
                    vectli = numpy.array([pointl.xypost[0] - pointi.xypost[0], pointl.xypost[1] - pointi.xypost[1]])
                    if (distik != 0) and (distli != 0):
                        if thresh >= (vectki[0] * vectli[0] + vectki[1] * vectli[1]) / (numpy.sum(vectki**2)**.5 * numpy.sum(vectli**2)**.5):
                            notneb = False
                        else:
                            i = i + 1
                    else:
                        i = i + 1
                if notneb:
                    potnbs.append(pointl)
            for truneb in potnbs:
                self.connect(pointk, truneb)
    
    def store(self, filename):
        hdfile = h5py.File(filename, 'w')
        try:
            nodarr = []
            nodrow = []
            edglst = []
            vstlst = []
            for node in self.nodlst:
                vstlst.append(node)
                nodrow = []
                nodrow.append(node.height)
                nodrow.append(node.xypost[0])
                nodrow.append(node.xypost[1])
                nodrow.append(node.width)
                nodrow.append(node.base)
                for neb in node.neibrs:
                    if not neb[0] in vstlst:
                        edglst.append([node.xypost[0], node.xypost[1], neb[0].xypost[0], neb[0].xypost[1]])
                nodarr.append(nodrow)
            hdfile.create_dataset('node_list', data = nodarr)
            hdfile.create_dataset('edge_list', data = edglst)
        finally:
            hdfile.close()
    
    def restore(self, filename):
        hdfile = h5py.File(filename, 'r')
        try:
            noddat = hdfile['node_list']
            edglst = hdfile['edge_list']
            self.nodlst = []
            for data in noddat:
                self.nodlst.append(atomnode([data[0], data[1], data[2], data[3], data[4]]))
            for edge in edglst:
                nodea = None
                nodeb = None
                k = 0
                while k < len(self.nodlst) and (nodea == None or nodeb == None):
                    if self.nodlst[k].xypost[0] == edge[0] and self.nodlst[k].xypost[1] == edge[1]:
                        nodea = self.nodlst[k]
                    elif self.nodlst[k].xypost[0] == edge[2] and self.nodlst[k].xypost[1] == edge[3]:
                        nodeb = self.nodlst[k]
                    k = k + 1
                if not nodea == None and not nodeb == None:
                    self.connect(nodea, nodeb)
        finally:
            hdfile.close()
    
    def disply(self, imgsze):
        retimg = numpy.zeros(imgsze)
        visitd = []
        for node in self.nodlst:
            visitd.append(node)
            for neb in node.neibrs:
                if not neb[0] in visitd:
                    direct = numpy.array([neb[0].xypost[0] - node.xypost[0], neb[0].xypost[1] - node.xypost[1], neb[0].sizcat - node.sizcat])
                    pathln = (direct[0]**2 + direct[1]**2)**.5
                    direct = direct / pathln
                    pathgn = 1
                    while pathgn < pathln:
                        if numpy.round(node.xypost[1] + pathgn * direct[1]) >= 0 and numpy.round(node.xypost[1] + pathgn * direct[1]) < retimg.shape[0] and numpy.round(node.xypost[0] + pathgn * direct[0]) >= 0 and numpy.round(node.xypost[0] + pathgn * direct[0]) < retimg.shape[1]:
                            retimg[int(numpy.round(node.xypost[1] + pathgn * direct[1]))][int(numpy.round(node.xypost[0] + pathgn * direct[0]))] = direct[2] * pathgn / pathln + node.sizcat + .5
                        pathgn = pathgn + 1
            for j in range(int(numpy.round(node.xypost[1])) - 1, int(numpy.round(node.xypost[1])) + 2):
                for i in range(int(numpy.round(node.xypost[0])) - 1, int(numpy.round(node.xypost[0])) + 2):
                    if i >= 0 and i < imgsze[1] and j >= 0 and j < imgsze[0]:
                        retimg[j][i] = node.sizcat + 1
        return retimg
    
    def graaph(self, image = None):
        mpl.figure()
        if not image is None:
            mpl.imshow(image)
        visted = []
        sctlst = [[], []]
        for node in self.nodlst:
            visted.append(node)
            sctlst[0].append(node.xypost[0])
            sctlst[1].append(node.xypost[1])
            for neb in node.neibrs:
                if not (neb[0] in visted):
                    mpl.plot([node.xypost[0], neb[0].xypost[0]], [node.xypost[1], neb[0].xypost[1]])
        mpl.scatter(sctlst[0], sctlst[1])
        mpl.show()
    
    def edgrep(self):
        curnod = self.nodlst[0]
        vstlst = [self.nodlst[0]]
        nxtlst = []
        edglst = []
        for node in curnod.neibrs:
            edglst.append([curnod.xypost[0], curnod.xypost[1], curnod.height, curnod.sizcat, node[0].xypost[0], node[0].xypost[1], node[0].height, node[0].sizcat, ((node[0].xypost[0] - curnod.xypost[0])**2 + (node[0].xypost[1] - curnod.xypost[1])**2)**.5])
            nxtlst.append(node[0])
        while len(nxtlst) > 0:
            curnod = nxtlst.pop(0)
            if not curnod in vstlst:
                vstlst.append(curnod)
                for neibor in curnod.neibrs:
                    if not neibor[0] in vstlst:
                        nxtlst.append(neibor[0])
                        edglst.append([curnod.xypost[0], curnod.xypost[1], curnod.height, curnod.sizcat, neibor[0].xypost[0], neibor[0].xypost[1], neibor[0].height, neibor[0].sizcat, ((neibor[0].xypost[0] - curnod.xypost[0])**2 + (neibor[0].xypost[1] - curnod.xypost[1])**2)**.5])
        return edglst
    
    def pntrep(self):
        pntlst = []
        for node in self.nodlst:
            pntlst.append([node.height, node.xypost[0], node.xypost[1], node.width, node.base])
        return pntlst
    
# =============================================================================
#     def densify(self, ubound, lbound = 0):
#         for k in range(len(self.nodlst)):
#             for l in range(k + 1, len(self.nodlst)):
#                 hypdst = ((self.nodlst[k].xypost[0] - self.nodlst[l].xypost[0])**2 + (self.nodlst[k].xypost[1] - self.nodlst[l].xypost[1])**2)**.5
#                 if hypdst >= lbound and hypdst <= ubound:
#                     srchng = True
#                     for neb in self.nodlst[l].neibrs:
#                         if self.nodlst[k] == neb[0]:
#                             srchng = False
#                     if srchng:
#                         self.connect(self.nodlst[k], self.nodlst[l])
# =============================================================================
    
    def subdiv(self, ulbound, brbound):
        sublst = []
        for node in self.nodlst:
            if node.xypost[0] >= ulbound[0] and node.xypost[0] <= brbound[0] and node.xypost[1] <= ulbound[1] and node.xypost[1] >= brbound[1]:
                sublst.append(node)
        return sublst
    
    def findarea(self, node, bounds):
        trnsvecs = []
        for neb in node.neibrs:
            trnsvecs.append([neb[0].xypost[0] - node.xypost[0], neb[0].xypost[1] - node.xypost[1]])
        tovsit = [[int(numpy.round(node.xypost[0])), int(numpy.round(node.xypost[1]))]]
        tovsit.append([tovsit[0][0] + 1, tovsit[0][1]])
        tovsit.append([tovsit[0][0], tovsit[0][1] + 1])
        tovsit.append([tovsit[0][0] - 1, tovsit[0][1]])
        tovsit.append([tovsit[0][0], tovsit[0][1] - 1])
        domain = []
        while len(tovsit) > 0:
            curpos = tovsit.pop(0)
            if curpos[0] >= 0 and curpos[0] < bounds[1] and curpos[1] >= 0 and curpos[1] < bounds[0]:
                posvec = [curpos[0] - node.xypost[0], curpos[1] - node.xypost[1]]
                inside = True
                for vec in trnsvecs:
                    if (posvec[0] * vec[0] + posvec[1] * vec[1]) > (vec[0]**2 + vec[1]**2) / 2 and inside:
                        inside = False
                if inside:
                    domain.append(curpos)
                    # If I am to the right and down from the center, always 
                    # add down, and add right if I am on the axis
                    if posvec[0] > 0 and posvec[1] <= 0:
                        tovsit.append([curpos[0], curpos[1] - 1])
                        if posvec[1] == 0:
                            tovsit.append([curpos[0] + 1, curpos[1]])
                    # If I am down and to the left of the center, always add 
                    # left, and add down if I am on the axis
                    elif posvec[1] < 0 and posvec[0] <= 0:
                        tovsit.append([curpos[0] - 1, curpos[1]])
                        if posvec[0] == 0:
                            tovsit.append([curpos[0], curpos[1] - 1])
                    # If I am to the left and up from the center, always add 
                    # up, and add left if I am on the axis
                    elif posvec[0] < 0 and posvec[1] >= 0:
                        tovsit.append([curpos[0], curpos[1] + 1])
                        if posvec[1] == 0:
                            tovsit.append([curpos[0] - 1, curpos[1]])
                    # If I am up and to the right from the center, always add 
                    # right, and add up if I am on the axis
                    elif posvec[1] > 0 and posvec[0] >= 0:
                        tovsit.append([curpos[0] + 1, curpos[1]])
                        if posvec[0] == 0:
                            tovsit.append([curpos[0], curpos[1] + 1])
        return domain
    
    def fitarea(self, node, bounds):
        # Similar to findarea, but for getting the area over which to fit a 
        # point, currently in cyclic refine
        trnsvecs = []
        maxdst = ((node.neibrs[0][0].xypost[0] - node.xypost[0])**2 + (node.neibrs[0][0].xypost[1] - node.xypost[1])**2)**.5
        for neb in node.neibrs:
            trnsvecs.append([neb[0].xypost[0] - node.xypost[0], neb[0].xypost[1] - node.xypost[1]])
            if ((neb[0].xypost[0] - node.xypost[0])**2 + (neb[0].xypost[1] - node.xypost[1])**2)**.5 > maxdst:
                maxdst = ((neb[0].xypost[0] - node.xypost[0])**2 + (neb[0].xypost[1] - node.xypost[1])**2)**.5
        tovsit = [[int(numpy.round(node.xypost[0])), int(numpy.round(node.xypost[1]))]]
        tovsit.append([tovsit[0][0] + 1, tovsit[0][1]])
        tovsit.append([tovsit[0][0], tovsit[0][1] + 1])
        tovsit.append([tovsit[0][0] - 1, tovsit[0][1]])
        tovsit.append([tovsit[0][0], tovsit[0][1] - 1])
        domain = []
        while len(tovsit) > 0:
            curpos = tovsit.pop(0)
            if curpos[0] >= 0 and curpos[0] < bounds[1] and curpos[1] >= 0 and curpos[1] < bounds[0]:
                posvec = [curpos[0] - node.xypost[0], curpos[1] - node.xypost[1]]
                inside = True
                for vec in trnsvecs:
                    if inside and (posvec[0] * vec[0] + posvec[1] * vec[1]) > (vec[0]**2 + vec[1]**2):
                        inside = False
                if inside and (posvec[0]**2 + posvec[1]**2)**.5 > maxdst:
                    inside = False
                if inside:
                    domain.append(curpos)
                    # If I am to the right and down from the center, always 
                    # add down, and add right if I am on the axis
                    if posvec[0] > 0 and posvec[1] <= 0:
                        tovsit.append([curpos[0], curpos[1] - 1])
                        if posvec[1] == 0:
                            tovsit.append([curpos[0] + 1, curpos[1]])
                    # If I am down and to the left of the center, always add 
                    # left, and add down if I am on the axis
                    elif posvec[1] < 0 and posvec[0] <= 0:
                        tovsit.append([curpos[0] - 1, curpos[1]])
                        if posvec[0] == 0:
                            tovsit.append([curpos[0], curpos[1] - 1])
                    # If I am to the left and up from the center, always add 
                    # up, and add left if I am on the axis
                    elif posvec[0] < 0 and posvec[1] >= 0:
                        tovsit.append([curpos[0], curpos[1] + 1])
                        if posvec[1] == 0:
                            tovsit.append([curpos[0] - 1, curpos[1]])
                    # If I am up and to the right from the center, always add 
                    # right, and add up if I am on the axis
                    elif posvec[1] > 0 and posvec[0] >= 0:
                        tovsit.append([curpos[0] + 1, curpos[1]])
                        if posvec[0] == 0:
                            tovsit.append([curpos[0], curpos[1] + 1])
        return [maxdst, domain]
    
    def refinepoints(self, image, wounds = [], baunds = []):
        # This takes in an image and uses it to refine the positions of the 
        # points in the lattice
        # First, estimate the height of each node to figure out which node 
        # should be investigated first
        hgtest = 0
        tovist = []
        for node in self.nodlst:
            hgtest = 0
            domain = self.findarea(node, image.shape)
            #print(node.xypost)
            for pnt in domain:
                hgtest = hgtest + image[pnt[1]][pnt[0]]
            if len(domain) > 0:
                hgtest = hgtest / len(domain)
            if len(tovist) == 0:
                tovist.append([hgtest, node])
            else:
                chcknd = len(tovist) - 1
                stpsze = len(tovist) / 2
                #print(f'Sort value: {edg[0]}')
                while stpsze > 0:
                    #print(f'{chcknd}: {edghep[chcknd][0]}')
                    if hgtest < tovist[chcknd - int(stpsze)][0]:
                        chcknd = chcknd - int(stpsze)
                        if stpsze > 1:
                            stpsze = numpy.ceil(stpsze)
                    stpsze = int(stpsze) / 2
                #print(f'Final check {chcknd}: {edghep[chcknd][0]}')
                if hgtest > tovist[chcknd][0]:
                    tovist.insert(chcknd + 1, [hgtest, node])
                else:
                    tovist.insert(chcknd, [hgtest, node])
        k = 1
        #timex = []
        #timea = []
        #timer = []
        for curnod in tovist:
            if k % 100 == 0:
                print(f'The points are {100 * k / len(self.nodlst):.2f}% refined.')
            k = k + 1
            #sloop = time()
            # domain is the area over which to fit
            domain = self.findarea(curnod[1], image.shape)
            #pdget = time()
            # mindst is the minimum 
            mindst = ((curnod[1].neibrs[0][0].xypost[0] - curnod[1].xypost[0])**2 + (curnod[1].neibrs[0][0].xypost[1] - curnod[1].xypost[1])**2)**.5
            # I don't think I can do n-point fitting. I can do multi-point 
            # fitting but I think I have to hard code the number, which isn't 
            # preferred since I don't know how many neighbors I might need to 
            # make cases for. What I can do is feed my 1-point curve a 
            # modified set of z-values based on neighboring points that have 
            # already been fit. I suppose hard-coding n-point surface 
            # equations is the next step
            extpts = []
            extref = []
            for neb in curnod[1].neibrs:
                if ((neb[0].xypost[0] - curnod[1].xypost[0])**2 + (neb[0].xypost[1] - curnod[1].xypost[1])**2)**.5 < mindst:
                    mindst = ((neb[0].xypost[0] - curnod[1].xypost[0])**2 + (neb[0].xypost[1] - curnod[1].xypost[1])**2)**.5
                # If the point has been fit already, use it as additional 
                # fitting information
                extare = self.findarea(neb[0], image.shape)
                for pnt in extare:
                    domain.append(pnt)
                if neb[0].height != 0 and neb[0].width != 1 and neb[0].base != 0:
                    extpts.append([neb[0].height, neb[0].xypost[0], neb[0].xypost[1], neb[0].width, neb[0].base])
                else:
                    extref.append([neb[0].xypost[0], neb[0].xypost[1]])
            extref.insert(0, [curnod[1].xypost[0], curnod[1].xypost[1], mindst])
            #pmind = time()
            refnod = pointfitting.stockfit(image, domain, benfit = extpts, inpara = extref, waunds = wounds, baonds = baunds)
            #prefp = time()
            #timex.append(k)
            #timea.append(pdget - sloop)
            #timer.append(prefp - pmind)
            if not (refnod is None):
                curnod[1].height = refnod[0]
                curnod[1].xypost[0] = refnod[1]
                curnod[1].xypost[1] = refnod[2]
                curnod[1].width = refnod[3]
                curnod[1].base = refnod[4]
        #mpl.figure
        #mpl.plot(timex, timea)
        #mpl.plot(timex, timer)
    
    def cyclicrefine(self, image):
        # First, sort the nodes based on their estimated height, and get their 
        # bounds (x, y, h, radius). Fitting them 
        # in order of height might help with the fitting of more intense nodes
        # depending on how you fit things
        globas = numpy.min(image)
        tovist = []
        bounda = numpy.zeros((len(self.nodlst), 2, 5))
        for i in range(len(self.nodlst)):
            nearst = numpy.round(self.nodlst[i].xypost)
            if nearst[0] < 0:
                nearst[0] = 0
            elif nearst[0] >= image.shape[1]:
                nearst[0] = image.shape[1] - 1
            if nearst[1] < 0:
                nearst[1] = 0
            elif nearst[1] >= image.shape[0]:
                nearst[1] = image.shape[0] - 1
            hgtest = image[int(nearst[1])][int(nearst[0])] - globas
            maxhgt = image[int(nearst[1])][int(nearst[0])] - globas
            cnt = 1
            if nearst[0] > 0:
                hgtest = hgtest + image[int(nearst[1])][int(nearst[0]) - 1] - globas
                cnt = cnt + 1
                if image[int(nearst[1])][int(nearst[0]) - 1] - globas > maxhgt:
                    maxhgt = image[int(nearst[1])][int(nearst[0]) - 1] - globas
            if nearst[0] < image.shape[1] - 1:
                hgtest = hgtest + image[int(nearst[1])][int(nearst[0]) + 1] - globas
                cnt = cnt + 1
                if image[int(nearst[1])][int(nearst[0]) + 1] - globas > maxhgt:
                    maxhgt = image[int(nearst[1])][int(nearst[0]) + 1] - globas
            if nearst[1] > 0:
                hgtest = hgtest + image[int(nearst[1]) - 1][int(nearst[0])] - globas
                cnt = cnt + 1
                if image[int(nearst[1]) - 1][int(nearst[0])] - globas > maxhgt:
                    maxhgt = image[int(nearst[1]) - 1][int(nearst[0])] - globas
            if nearst[1] < image.shape[0] - 1:
                hgtest = hgtest + image[int(nearst[1]) + 1][int(nearst[0])] - globas
                cnt = cnt + 1
                if image[int(nearst[1]) + 1][int(nearst[0])] - globas > maxhgt:
                    maxhgt = image[int(nearst[1]) + 1][int(nearst[0])] - globas
            hgtest = hgtest / cnt
            mindst = ((self.nodlst[i].neibrs[0][0].xypost[0] - self.nodlst[i].xypost[0])**2 + (self.nodlst[i].neibrs[0][0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5
            maxdst = ((self.nodlst[i].neibrs[0][0].xypost[0] - self.nodlst[i].xypost[0])**2 + (self.nodlst[i].neibrs[0][0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5
            for neb in self.nodlst[i].neibrs:
                if ((neb[0].xypost[0] - self.nodlst[i].xypost[0])**2 + (neb[0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5 < mindst:
                    mindst = ((neb[0].xypost[0] - self.nodlst[i].xypost[0])**2 + (neb[0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5
                elif ((neb[0].xypost[0] - self.nodlst[i].xypost[0])**2 + (neb[0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5 > maxdst:
                    maxdst = ((neb[0].xypost[0] - self.nodlst[i].xypost[0])**2 + (neb[0].xypost[1] - self.nodlst[i].xypost[1])**2)**.5
            bounda[i][0][0] = 0
            bounda[i][0][1] = nearst[0] - mindst / 2
            bounda[i][0][2] = nearst[1] - mindst / 2
            bounda[i][0][3] = 1
            bounda[i][0][4] = globas
            bounda[i][1][0] = maxhgt
            bounda[i][1][1] = nearst[0] + mindst / 2
            bounda[i][1][2] = nearst[1] + mindst / 2
            bounda[i][1][3] = maxdst**2 / 4
            bounda[i][1][4] = hgtest
            if len(tovist) == 0:
                tovist.append([hgtest, self.nodlst[i], i])
            else:
                chcknd = len(tovist) - 1
                stpsze = len(tovist) / 2
                while stpsze > 0:
                    if hgtest < tovist[chcknd - int(stpsze)][0]:
                        chcknd = chcknd - int(stpsze)
                        if stpsze > 1:
                            stpsze = numpy.ceil(stpsze)
                    stpsze = int(stpsze) / 2
                if hgtest > tovist[chcknd][0]:
                    tovist.insert(chcknd + 1, [hgtest, self.nodlst[i], i])
                else:
                    tovist.insert(chcknd, [hgtest, self.nodlst[i], i])
        repeat = True
        r = 0
        while repeat:
            # Second, fit the height, width, and baseline for each point. Fit 
            # these independently of x and y
            k = 0
            fitdom = []
            for i in range(len(self.nodlst)):
                fitdom.append([[], [], [], [], []])
            msdlst = []
            for entery in tovist:
                if k % 100 == 0:
                    print(f'The dimensions are {100 * k / len(self.nodlst):.2f}% refined.')
                k = k + 1
                curnod = entery[1]
                # I will want to fit a number of neighboring nodes simultaneously 
                # in order to account for the potential effects of bleed over. 
                # Typically I used nearest neighbors, but I might want to use more 
                # depending on the size of the area I want to fit to. The area I 
                # select should be the area over which I expect a point to have 
                # some amount of influence, where z > .01 * f(0, 0) or something 
                # similar. However, since I don't have a good estimate of the 
                # influence area, this might bring in too many points to fit, 
                # bringing in too much uncertainty that I missed a point with 
                # significant influence over the area and requiring too many 
                # function definitions, so perhaps it would be better to go with 
                # something similar to before where the area is bounded by the 
                # neighbors with a maximum distance in case there are no neighbors 
                # on one side. I should also choose points which I expect to have 
                # some influence over the area I wish to fit as well, using the 
                # same criteria. I ought to keep track of these points as well, in 
                # case I want to reference them again (a likely scenario). 
                # Selecting the area is both crucial and tricky because it should 
                # be based on my expected width, but I don't know what that should 
                # choose. My width should be at maximum the  smallest width 
                # necessary to create a drop in intensity sufficient to  make each 
                # point indistinguishable from its nearest neighbor as separate 
                # points, considering their height difference. This requires an 
                # assumption about the width of the neighboring point as well, so 
                # I suppose we will have to assume similar or equal widths. The 
                # vanishing point for this width is what I should use to obtain my 
                # fitting area
                effrad, fitarr = self.fitarea(curnod, image.shape)
                parlst = [bounda[entery[2]]]
                xylist = [[curnod.xypost[0], curnod.xypost[1]]]
                tosrch = []
                allchk = [curnod]
                indlst = [entery[2]]
                for neb in curnod.neibrs:
                    tosrch.append(neb[0])
                    parlst.append(bounda[self.nodlst.index(neb[0])])
                    xylist.append([neb[0].xypost[0], neb[0].xypost[1]])
                    allchk.append(neb[0])
                    indlst.append(self.nodlst.index(neb[0]))
                while len(tosrch) > 0:
                    presnt = tosrch.pop(0)
                    for neb in presnt.neibrs:
                        if not neb[0] in allchk:
                            chkdst = ((neb[0].xypost[0] - curnod.xypost[0])**2 + (neb[0].xypost[1] - curnod.xypost[1])**2)**.5
                            allchk.append(neb[0])
                            if chkdst < 3.035 * effrad / 2: #3.035 * bounda[self.nodlst.index(neb[0])][1][3]**.5
                                #tosrch.append(neb[0])
                                parlst.append(bounda[self.nodlst.index(neb[0])])
                                xylist.append([neb[0].xypost[0], neb[0].xypost[1]])
                                indlst.append(self.nodlst.index(neb[0]))
                #print(len(xylist))
                try:
                    zwbfit = pointfitting.stdfit(image, fitarr, parlst, xylist)
                    curnod.height = zwbfit[0]
                    curnod.width = zwbfit[1]
                    curnod.base = zwbfit[2]
                    fitdom[indlst[0]][0].append(zwbfit[0])
                    fitdom[indlst[0]][3].append(zwbfit[1])
                    fitdom[indlst[0]][4].append(zwbfit[2])
                    for i in range(1, numpy.min((len(indlst), 15))):
                        fitdom[indlst[i]][0].append(zwbfit[2 * i + 1])
                        fitdom[indlst[i]][3].append(zwbfit[2 * i + 2])
                        fitdom[indlst[i]][4].append(zwbfit[2])
                except:
                    print([curnod.xypost, self.nodlst[entery[2]].xypost])
                    msdlst.append(entery[2])
            for slip in msdlst:
                self.nodlst[slip].height = numpy.average(fitdom[slip][0])
                self.nodlst[slip].width = numpy.average(fitdom[slip][3])
                self.nodlst[slip].base = numpy.average(fitdom[slip][4])
            # Third, estimate the position each point. This will be independent of 
            # height, width, and base
            k = 1
            msdlst = []
            for entery in tovist:
                if k % 100 == 0:
                    print(f'The positions are {100 * k / len(self.nodlst):.2f}% refined.')
                k = k + 1
                curnod = entery[1]
                # domain is the area over which to fit
                effrad, fitarr = self.fitarea(curnod, image.shape)
                parlst = [bounda[entery[2]]]
                hwblst = [[curnod.height, curnod.width, curnod.base]]
                tosrch = []
                allchk = [curnod]
                indlst = [entery[2]]
                for neb in curnod.neibrs:
                    tosrch.append(neb[0])
                    parlst.append(bounda[self.nodlst.index(neb[0])])
                    hwblst.append([neb[0].height, neb[0].width, neb[0].base])
                    allchk.append(neb[0])
                    indlst.append(self.nodlst.index(neb[0]))
                while len(tosrch) > 0:
                    presnt = tosrch.pop(0)
                    for neb in presnt.neibrs:
                        if not neb[0] in allchk:
                            chkdst = ((neb[0].xypost[0] - curnod.xypost[0])**2 + (neb[0].xypost[1] - curnod.xypost[1])**2)**.5
                            allchk.append(neb[0])
                            if chkdst < 3.035 * effrad / 2:
                                tosrch.append(neb[0])
                                parlst.append(bounda[self.nodlst.index(neb[0])])
                                hwblst.append([neb[0].height, neb[0].width, neb[0].base])
                                indlst.append(self.nodlst.index(neb[0]))
                try:
                    xyfit = pointfitting.posfit(image, fitarr, parlst, hwblst)
                    curnod.xypost[0] = xyfit[0]
                    curnod.xypost[1] = xyfit[1]
                    for i in range(numpy.min((len(indlst), 15))):
                        fitdom[indlst[i]][1].append(xyfit[2 * i])
                        fitdom[indlst[i]][2].append(xyfit[2 * i + 1])
                except:
                    print([curnod.xypost, self.nodlst[entery[2]].xypost])
                    msdlst.append(entery[2])
            for slip in msdlst:
                self.nodlst[slip].xypost[0] = numpy.average(fitdom[slip][1])
                self.nodlst[slip].xypost[1] = numpy.average(fitdom[slip][2])
            r = r + 1
            if r > 2:
                repeat = False
            if not repeat:
                # Next, evaluate new bounds and decide if you want to repeat
                for i in range(len(fitdom)):
                    print(bounda[i])
                    print([[numpy.min(fitdom[i][0]), numpy.min(fitdom[i][1]), numpy.min(fitdom[i][2]), numpy.min(fitdom[i][3]), numpy.min(fitdom[i][4])], [numpy.max(fitdom[i][0]), numpy.max(fitdom[i][1]), numpy.max(fitdom[i][2]), numpy.max(fitdom[i][3]), numpy.max(fitdom[i][4])]])
                    print([numpy.average(fitdom[i][0]), numpy.average(fitdom[i][1]), numpy.average(fitdom[i][2]), numpy.average(fitdom[i][3]), numpy.average(fitdom[i][4])])
    
    def heightest(self, image):
        # This will estimate the heights of each node using the supplied image 
        # by simple pixel average near the central point
        minhgt = numpy.min(image)
        print(minhgt)
        for node in self.nodlst:
            nearst = numpy.round(node.xypost)
            if nearst[0] < 0:
                nearst[0] = 0
            elif nearst[0] >= image.shape[1]:
                nearst[0] = image.shape[1] - 1
            if nearst[1] < 0:
                nearst[1] = 0
            elif nearst[1] >= image.shape[0]:
                nearst[1] = image.shape[0] - 1
            if nearst[0] == 0:
                if nearst[1] == 0:
                    avgara = image[int(nearst[1]):int(nearst[1] + 2), int(nearst[0]):int(nearst[0] + 2)]
                elif nearst[1] == image.shape[0] - 1:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 1), int(nearst[0]):int(nearst[0] + 2)]
                else:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 2), int(nearst[0]):int(nearst[0] + 2)]
            elif nearst[0] == image.shape[1] - 1:
                if nearst[1] == 0:
                    avgara = image[int(nearst[1]):int(nearst[1] + 2), int(nearst[0] - 1):int(nearst[0] + 1)]
                elif nearst[1] == image.shape[0] - 1:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 1), int(nearst[0] - 1):int(nearst[0] + 1)]
                else:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 2), int(nearst[0] - 1):int(nearst[0] + 1)]
            else:
                if nearst[1] == 0:
                    avgara = image[int(nearst[1]):int(nearst[1] + 2), int(nearst[0] - 1):int(nearst[0] + 2)]
                elif nearst[1] == image.shape[0] - 1:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 1), int(nearst[0] - 1):int(nearst[0] + 2)]
                else:
                    avgara = image[int(nearst[1] - 1):int(nearst[1] + 2), int(nearst[0] - 1):int(nearst[0] + 2)]
            node.height = numpy.sum(avgara) / (avgara.shape[0] * avgara.shape[1]) - minhgt
    
    def catheight(self, hgtlst):
        # This will sort the nodes of the graph into categories based on their 
        # height
        for node in self.nodlst:
            nrstht = numpy.abs(node.height - hgtlst[0])
            bstcat = 0
            for i in range(1, len(hgtlst)):
                if nrstht > numpy.abs(node.height - hgtlst[i]):
                    nrstht = numpy.abs(node.height - hgtlst[i])
                    bstcat = i
            node.sizcat = bstcat
    
    def brushfire(self, hightt = -1, widtht = -1):
        # This will look through the identified, refined points and remove 
        # ones deemed insignificant or blatantly wrong. I don't know if I want 
        # to do the analysis of what the thresholds should be in this method 
        # or elsewhere, so I won't include that for now
        brnlst = []
        widact = True
        if widtht < 0:
            widact = False
        for node in self.nodlst:
            if node.height < hightt:
                while len(node.neibrs) > 0:
                    self.disconnect(node, node.neibrs[0][0])
                brnlst.append(node)
            elif node.width > widtht and widact:
                while len(node.neibrs) > 0:
                    self.disconnect(node, node.neibrs[0][0])
                brnlst.append(node)
        for node in brnlst:
            self.nodlst.remove(node)
    
# =============================================================================
#     def erroranalysis(self, image):
#         # This will attempt to determine which points provide the most error 
#         # in replicating the image
#         pntimg = self.pntreplct(image.shape)
#         bckimg = self.bckgndreplct(image.shape)
#         errimg = (image - pntimg - bckimg)**2
#         errlst = []
#         for node in self.nodlst:
#             noderr = 0
#             nearst = numpy.round(node.xypost)
#             if nearst[0] < 0:
#                 nearst[0] = 0
#             elif nearst[0] >= image.shape[1]:
#                 nearst[0] = image.shape[1] - 1
#             if nearst[1] < 0:
#                 nearst[1] = 0
#             elif nearst[1] >= image.shape[0]:
#                 nearst[1] = image.shape[0] - 1
#             bckper = numpy.abs(bckimg[int(nearst[1])][int(nearst[0])]) / (pntimg[int(nearst[1])][int(nearst[0])] + numpy.abs(bckimg[int(nearst[1])][int(nearst[0])]))
#             noderr = noderr + errimg[int(nearst[1])][int(nearst[0])] * (1 - bckper) * node.height * numpy.exp(-1 * ((nearst[0] - node.xypost[0])**2 + (nearst[1] - node.xypost[1])**2) / (2 * node.width)) / pntimg[int(nearst[1])][int(nearst[0])]
#             nxtlst = []
#             nxtlst.append(nearst + numpy.array([1, 0]))
#             nxtlst.append(nearst + numpy.array([0, 1]))
#             nxtlst.append(nearst + numpy.array([-1, 0]))
#             nxtlst.append(nearst + numpy.array([0, -1]))
#             while len(nxtlst) > 0:
#                 curcor = nxtlst.pop(0)
#                 if curcor[0] >= 0 and curcor[0] < image.shape[1] and curcor[1] >= 0 and curcor[1] < image.shape[0]:
#                     bump = node.height * numpy.exp(-1 * ((curcor[0] - node.xypost[0])**2 + (curcor[1] - node.xypost[1])**2) / (2 * node.width))
#                     bckper = numpy.abs(bckimg[int(curcor[1])][int(curcor[0])]) / (pntimg[int(curcor[1])][int(curcor[0])] + numpy.abs(bckimg[int(curcor[1])][int(curcor[0])]))
#                     noderr = noderr + errimg[int(curcor[1])][int(curcor[0])] * (1 - bckper) * bump / pntimg[int(curcor[1])][int(curcor[0])]
#                     if bump / node.height >= .001:
#                         if curcor[0] > nearst[0] and curcor[1] <= nearst[1]:
#                             nxtlst.append(curcor + numpy.array([0, -1]))
#                             if curcor[1] == nearst[1]:
#                                 nxtlst.append(curcor + numpy.array([1, 0]))
#                         elif curcor[1] < nearst[1] and curcor[0] <= nearst[0]:
#                             nxtlst.append(curcor + numpy.array([-1, 0]))
#                             if curcor[0] == nearst[0]:
#                                 nxtlst.append(curcor + numpy.array([0, -1]))
#                         elif curcor[0] < nearst[0] and curcor[1] >= nearst[1]:
#                             nxtlst.append(curcor + numpy.array([0, 1]))
#                             if curcor[1] == nearst[1]:
#                                 nxtlst.append(curcor + numpy.array([-1, 0]))
#                         elif curcor[1] > nearst[1] and curcor[0] >= nearst[0]:
#                             nxtlst.append(curcor + numpy.array([1, 0]))
#                             if curcor[0] == nearst[0]:
#                                 nxtlst.append(curcor + numpy.array([0, 1]))
#             errlst.append(noderr)
#         return errlst
#     
#     def split(self, image, node):
#         # Gather the appropriate area over which to search
#         domain = self.findarea(node, image.shape)
#         for neb in node.neibrs:
#             araara = self.findarea(neb[0], image.shape)
#             for pnt in araara:
#                 domain.append(pnt)
#         # The image fed to the method should be a modified image with the 
#         # background and other nodes subtracted out, so it should serve as a 
#         # measure of error. Adding the curve of the node back in should 
#         # reproduce the original image within the local area minus the 
#         # contribution of the surrounding features. This can then be used to 
#         # re-fit two imaginary curves in the same region of the first
#         orgerr = 0
#         xvect = []
#         yvect = []
#         zvect = []
#         for pnt in domain:
#             xvect.append(pnt[0])
#             yvect.append(pnt[1])
#             zvect.append(image[pnt[1]][pnt[0]] + node.height * numpy.exp(-1 * ((pnt[0] - node.xypost[0])**2 + (pnt[1] - node.xypost[1])**2) / (2 * node.width)))
#             orgerr = orgerr + image[pnt[1]][pnt[0]]**2
#         # Fit two surfaces with independent positions, heights, and widths, 
#         # but a common baseline
#         lounds = [node.xypost[0] - 2 * node.width**.5]
#         lounds.append(node.xypost[1] - 2 * node.width**.5)
#         lounds.append(0)
#         lounds.append(1)
#         lounds.append(numpy.min(image))
#         lounds.append(node.xypost[0] - 2 * node.width**.5)
#         lounds.append(node.xypost[1] - 2 * node.width**.5)
#         lounds.append(0)
#         lounds.append(1)
#         uounds = [node.xypost[0] + 2 * node.width**.5]
#         uounds.append(node.xypost[1] + 2 * node.width**.5)
#         uounds.append(node.height * 2)
#         uounds.append(node.width * 2)
#         uounds.append(numpy.max(image))
#         uounds.append(node.xypost[0] + 2 * node.width**.5)
#         uounds.append(node.xypost[1] + 2 * node.width**.5)
#         uounds.append(node.height * 2)
#         uounds.append(node.width * 2)
#         try:
#             dblnod, fitgud = sciopt.curve_fit(pointfitting.iixyhgauss, [xvect, yvect], zvect, bounds = (lounds, uounds))
#             print(dblnod)
#             newerr = 0
#             for i in range(len(zvect)):
#                 newerr = newerr + (zvect[i] - dblnod[2] * numpy.exp(-1 * ((xvect[i] - dblnod[0])**2 + (yvect[i] - dblnod[1])**2) / (2 * dblnod[3])) - dblnod[7] * numpy.exp(-1 * ((xvect[i] - dblnod[5])**2 + (yvect[i] - dblnod[6])**2) / (2 * dblnod[8])) - dblnod[4])**2
#             print([orgerr, newerr])
#             if newerr < orgerr:
#                 if ((dblnod[0] - dblnod[5])**2 + (dblnod[1] - dblnod[6])**2)**.5 > .25 and dblnod[2] > .1 * node.height and dblnod[7] > .1 * node.height:
#                     node.height = dblnod[2]
#                     node.xypost[0] = dblnod[0]
#                     node.xypost[1] = dblnod[1]
#                     node.width = dblnod[3]
#                     node.base = dblnod[4]
#                     newnod = atomnode([dblnod[7], dblnod[5], dblnod[6], dblnod[8], dblnod[4]])
#                     self.nodlst.append(newnod)
#                     self.connect(newnod, node)
#         except:
#             print('Could not find an optimal split')
# =============================================================================
    
    def pntreplct(self, imgsze):
        # Returns an replica of the lattice using the refined points as 
        # parameters for a Gaussian surface
        replca = numpy.zeros(imgsze)
        #intpx = []
        #intpy = []
        #intpz = []
        for node in self.nodlst:
            nearst = numpy.round(node.xypost)
        #    intpx.append(node.xypost[0])
        #    intpy.append(node.xypost[1])
        #    intpz.append(node.base)
            if nearst[0] < 0:
                nearst[0] = 0
            elif nearst[0] >= imgsze[1]:
                nearst[0] = imgsze[1] - 1
            if nearst[1] < 0:
                nearst[1] = 0
            elif nearst[1] >= imgsze[0]:
                nearst[1] = imgsze[0] - 1
            replca[int(nearst[1])][int(nearst[0])] = replca[int(nearst[1])][int(nearst[0])] + node.height * numpy.exp(-1 * ((nearst[0] - node.xypost[0])**2 + (nearst[1] - node.xypost[1])**2) / (2 * node.width))
            nxtlst = []
            nxtlst.append(nearst + numpy.array([1, 0]))
            nxtlst.append(nearst + numpy.array([0, 1]))
            nxtlst.append(nearst + numpy.array([-1, 0]))
            nxtlst.append(nearst + numpy.array([0, -1]))
            while len(nxtlst) > 0:
                curcor = nxtlst.pop(0)
                if curcor[0] >= 0 and curcor[0] < imgsze[1] and curcor[1] >= 0 and curcor[1] < imgsze[0]:
                    bump = node.height * numpy.exp(-1 * ((curcor[0] - node.xypost[0])**2 + (curcor[1] - node.xypost[1])**2) / (2 * node.width))
                    replca[int(curcor[1])][int(curcor[0])] = replca[int(curcor[1])][int(curcor[0])] + bump
                    if bump / node.height >= .001:
                        if curcor[0] > nearst[0] and curcor[1] <= nearst[1]:
                            nxtlst.append(curcor + numpy.array([0, -1]))
                            if curcor[1] == nearst[1]:
                                nxtlst.append(curcor + numpy.array([1, 0]))
                        elif curcor[1] < nearst[1] and curcor[0] <= nearst[0]:
                            nxtlst.append(curcor + numpy.array([-1, 0]))
                            if curcor[0] == nearst[0]:
                                nxtlst.append(curcor + numpy.array([0, -1]))
                        elif curcor[0] < nearst[0] and curcor[1] >= nearst[1]:
                            nxtlst.append(curcor + numpy.array([0, 1]))
                            if curcor[1] == nearst[1]:
                                nxtlst.append(curcor + numpy.array([-1, 0]))
                        elif curcor[1] > nearst[1] and curcor[0] >= nearst[0]:
                            nxtlst.append(curcor + numpy.array([1, 0]))
                            if curcor[0] == nearst[0]:
                                nxtlst.append(curcor + numpy.array([0, 1]))
        return replca
    
    def edgestat(self, binning = 'none', minpts = 2):
        # Returns a sorted list of all the edge lengths in the graph
        vsited = []
        edglst = []
        for node in self.nodlst:
            vsited.append(node)
            for neibr in node.neibrs:
                if not neibr[0] in vsited:
                    edglst.append(((neibr[0].xypost[0] - node.xypost[0])**2 + (neibr[0].xypost[1] - node.xypost[1])**2)**.5)
        srtarr = numpy.sort(numpy.array(edglst))
        if binning == 'dynamic':
            widnss = numpy.max(edglst) - numpy.min(edglst)
            digits = numpy.floor(numpy.log10(widnss))
            widnss = 10**digits * numpy.ceil(widnss / 10**digits)
            binsze = widnss * minpts / srtarr.shape
            strtvl = srtarr[0]
            binnum = 1
            edgbns = [[strtvl + binsze, 0]]
            edgnum = 0
            while edgnum < len(srtarr):
                if srtarr[edgnum] < strtvl + binnum * binsze:
                    edgnum = edgnum + 1
                    edgbns[binnum - 1][1] = edgbns[binnum - 1][1] + 1
                else:
                    binnum = binnum + 1
                    edgbns.append([strtvl + binnum * binsze, 0])
        elif binning == 'individual':
            edgbns = [[srtarr[0], 0]]
            binnum = 0
            for val in srtarr:
                if val > edgbns[binnum][0]:
                    binnum = binnum + 1
                    edgbns.append([val, 1])
                else:
                    edgbns[binnum][1] = edgbns[binnum][1] + 1
        else:
            edgbns = srtarr
        return edgbns
    
    def heightstat(self, binning = 'none', minpts = 2):
        hgtlst = []
        for node in self.nodlst:
            hgtlst.append(node.height)
        srtarr = numpy.sort(hgtlst)
        if binning == 'dynamic':
            widnss = numpy.max(hgtlst)
            digits = numpy.floor(numpy.log10(widnss))
            widnss = 10**digits * numpy.ceil(widnss / 10**digits)
            binsze = widnss * minpts / srtarr.shape
            binnum = 1
            hgtbns = [[binsze, 0]]
            hgtnum = 0
            while hgtnum < len(srtarr):
                if srtarr[hgtnum] < binnum * binsze:
                    hgtnum = hgtnum + 1
                    hgtbns[binnum - 1][1] = hgtbns[binnum - 1][1] + 1
                else:
                    binnum = binnum + 1
                    hgtbns.append([binnum * binsze, 0])
        elif binning == 'individual':
            hgtbns = [[srtarr[0], 0]]
            binnum = 0
            for val in srtarr:
                if val > hgtbns[binnum][0]:
                    binnum = binnum + 1
                    hgtbns.append([val, 1])
                else:
                    hgtbns[binnum][1] = hgtbns[binnum][1] + 1
        else:
            hgtbns = srtarr
        return hgtbns
    
    def widthstat(self, binning = 'none', minpts = 2):
        wdtlst = []
        for node in self.nodlst:
            wdtlst.append(node.width)
        srtarr = numpy.sort(wdtlst)
        if binning == 'dynamic':
            widnss = numpy.max(wdtlst)
            digits = numpy.floor(numpy.log10(widnss))
            widnss = 10**digits * numpy.ceil(widnss / 10**digits)
            binsze = widnss * minpts / srtarr.shape
            binnum = 1
            wdtbns = [[binsze, 0]]
            wdtnum = 0
            while wdtnum < len(srtarr):
                if srtarr[wdtnum] < binnum * binsze:
                    wdtnum = wdtnum + 1
                    wdtbns[binnum - 1][1] = wdtbns[binnum - 1][1] + 1
                else:
                    binnum = binnum + 1
                    wdtbns.append([binnum * binsze, 0])
        elif binning == 'individual':
            wdtbns = [[srtarr[0], 0]]
            binnum = 0
            for val in srtarr:
                if val > wdtbns[binnum][0]:
                    binnum = binnum + 1
                    wdtbns.append([val, 1])
                else:
                    wdtbns[binnum][1] = wdtbns[binnum][1] + 1
        else:
            wdtbns = srtarr
        return wdtbns
    
    def getcoords(self):
        xylist = []
        for node in self.nodlst:
            xylist.append([node.xypost[0], node.xypost[1]])
        return xylist
    
    def bckgndreplct(self, imgsze):
        replca = numpy.zeros(imgsze)
        interx = []
        intery = []
        interz = []
        for pnt in self.nodlst:
            interx.append(pnt.xypost[0])
            intery.append(pnt.xypost[1])
            interz.append(pnt.base)
        intrpx = []
        intrpy = []
        for j in range(imgsze[0]):
            for i in range(imgsze[1]):
                intrpx.append(i)
                intrpy.append(j)
        fitsrf = interp.griddata((numpy.array(interx), numpy.array(intery)), numpy.array(interz), (numpy.array(intrpx), numpy.array(intrpy)), method = 'nearest')
        for k in range(len(intrpx)):
            if not numpy.isnan(fitsrf[k]):
                if replca[intrpy[k]][intrpx[k]] == 0:
                    replca[intrpy[k]][intrpx[k]] = fitsrf[k]
                else:
                    replca[intrpy[k]][intrpy[k]] = (replca[intrpy[k]][intrpx[k]] + fitsrf[k])
        return replca
    
# =============================================================================
#     def trngleinterp(self, xycord, pointa, pointb, pointc):
#         vecam = ((xycord[0] - pointa[0])**2 + (xycord[1] - pointa[1])**2)**.5
#         vecbm = ((xycord[0] - pointb[0])**2 + (xycord[1] - pointb[1])**2)**.5
#         veccm = ((xycord[0] - pointc[0])**2 + (xycord[1] - pointc[1])**2)**.5
#         sidam = ((pointb[0] - pointa[0])**2 + (pointb[1] - pointa[1])**2)**.5
#         sidbm = ((pointc[0] - pointb[0])**2 + (pointc[1] - pointb[1])**2)**.5
#         sidcm = ((pointa[0] - pointc[0])**2 + (pointa[1] - pointc[1])**2)**.5
#         sempab = (vecam + vecbm + sidam) / 2
#         sempbc = (vecbm + veccm + sidbm) / 2
#         sempca = (veccm + vecam + sidcm) / 2
#         triabm = (sempab * (sempab - vecam) * (sempab - vecbm) * (sempab - sidam))**.5
#         tribcm = (sempbc * (sempbc - vecbm) * (sempbc - veccm) * (sempbc - sidbm))**.5
#         tricam = (sempca * (sempca - veccm) * (sempca - vecam) * (sempca - sidcm))**.5
#         fraca = tribcm / (triabm + tribcm + tricam)
#         fracb = tricam / (triabm + tribcm + tricam)
#         fracc = triabm / (triabm + tribcm + tricam)
#         return pointa[2] * fraca + pointb[2] * fracb + pointc[2] * fracc
# =============================================================================
    
    def getcircle(start, radius, bounds):
        points = [start]
        nxtarr = []
        if start[0] + 1 < bounds[0][1]:
            nxtarr.append([1, 0])
        if start[1] + 1 < bounds[1][1]:
            nxtarr.append([0, 1])
        if start[0] - 1 >= bounds[0][0]:
            nxtarr.append([-1, 0])
        if start[1] - 1 >= bounds[1][0]:
            nxtarr.append([0, -1])
        while len(nxtarr) > 0:
            cursft = nxtarr.pop(0)
            if (cursft[0]**2 + cursft[1]**2)**.5 <= rad:
                if start[0] + cursft[0] < bounds[0][1] and start[0] + cursft[0] >= bounds[0][0] and start[1] + cursft[1] < bounds[1][1] and start[1] + cursft[1] >= bounds[1][0]:
                    points.append([start[0] + cursft[0], start[1] + cursft[1]])
                    if start[0] > 0 and start[1] >= 0:
                        nxtarr.append([cursft[0], cursft[1] + 1])
                        if start[1] == 0:
                            nxtarr.append([cursft[0] + 1, cursft[1]])
                    if start[1] < 0 and start[0] >= 0:
                        nxtarr.append([cursft[0] + 1, cursft[1]])
                        if start[0] == 0:
                            nxtarr.append([cursft[0], cursft[1] - 1])
                    if start[0] < 0 and start[1] <= 0:
                        nxtarr.append([cursft[0], cursft[1] - 1])
                        if start[1] == 0:
                            nxtarr.append([cursft[0] - 1, cursft[1]])
                    if start[1] > 0 and start[0] <= 0:
                        nxtarr.append([cursft[0] - 1, cursft[1]])
                        if start[1] == 0:
                            nxtarr.append([cursft[0], cursft[1] + 1])
        return points
    
    def fixedpointeval(self, height, width, image):
        nrmimg = image - numpy.min(image)
        repimg = numpy.zeros(image.shape)
        trpxyz = [[], [], []]
        stalst = []
        effrad = (width * 9.21)**.5
        for node in self.nodlst:
            truval = []
            simval = []
            nearst = numpy.round(node.xypost)
            basweb = self.getcircle(nearst, effrad, [[0, image.shape[1]], [0, image.shape[0]]])
            for pnt in basweb:
                truval.append(nrmimg[int(pnt[1])][int(pnt[0])])
                simval.append(height * numpy.exp(-1 * ((pnt[0] - node.xypost[0])**2 + (pnt[1] - node.xypost[1])**2) / (2 * width)))
            truval = numpy.array(truval)
            simval = numpy.array(simval)
            premin = numpy.min(truval)
            preerr = numpy.sum((truval - premin)**2)
            h = 1
            brutef = True
            while brutef:
                curmin = numpy.min(truval - h * simval)
                curerr = numpy.sum((truval - h * simval - curmin)**2)
                if curerr > preerr:
                    brutef = False
                    stalst.append([h - 1, premin, preerr])
                else:
                    h = h + 1
            for i in range(len(basweb)):
                repimg[basweb[i][1]][basweb[i][0]] = repimg[basweb[i][1]][basweb[i][0]] + simval[i] * (h - 1)
            trpxyz[0].append(node.xypost[0])
            trpxyz[1].append(node.xypost[1])
            trpxyz[2].append(premin)
        imgcrd = [[], []]
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                imgcrd[0].append(i)
                imgcrd[1].append(j)
        bckgnd = interp.griddata((numpy.array(trpxyz[0]), numpy.array(trpxyz[1])), numpy.array(trpxyz[2]))
        return stalst

if __name__ == '__main__':
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    if ftype == 'hdf5':
        lonlat = lonelattice()
        lonlat.restore(flname)
        imgdat = numpy.genfromtxt(f'{fname}.csv', delimiter = ',')
    else:
        if ftype == 'csv':
            imgdat = numpy.genfromtxt(flname, delimiter = ',')
        else:
            imgdat = imageio.imread(flname)
        smthed = sandwhichsmooth.cycleinterp(imgdat, 4)
        scrtre = levelgraph.intensgraph()
        scrtre.altmetbfsnncnst(smthed)
        scrtre.hubevl()
        scrtre.hubcmp(2.5, 14)
        pntlst = scrtre.getcoords()
        lonlat = lonelattice()
        lonlat.djkbuild(pntlst)
        #lonlat.refinepoints(imgdat)
        #lonlat.redjkbuild()
        #lonlat.store(f'{fname}.hdf5')
        lonlat.heightest(smthed)
        hgtfit = pointfitting.emforsize(lonlat.heightstat(), 30)
        lonlat.catheight(hgtfit)
    #numpy.savetxt(f'{fname} points.csv', lonlat.pntrep(), delimiter = ',')
    #fitlst = lonlat.fixedpointeval(.003, 10, imgdat)
    #replca = lonlat.pntreplct(imgdat.shape) + lonlat.bckgndreplct(imgdat.shape)
    mpl.figure()
    bsma = mpl.imshow(smthed)
    mpl.colorbar(bsma)
    mpl.figure()
    bsmb = mpl.imshow(lonlat.disply(imgdat.shape))
    mpl.colorbar(bsmb)
# =============================================================================
#     mpl.figure()
#     bsmc = mpl.imshow(replca)
#     mpl.colorbar(bsmc)
#     mpl.figure()
#     bsmd = mpl.imshow(imgdat - replca)
# =============================================================================
