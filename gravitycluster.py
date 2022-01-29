# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:03:50 2019

@author: shini
"""

import numpy
import imageio
import heapq
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mpl

class ndmovenode:
    
    def __init__(self, orgpos, curpos = None, clasif = -1):
        # orgpos is the position of the data point in the original sample 
        # space so that the data structure can map between original and 
        # transformed data space as necessary.
        self.orgpos = numpy.array(orgpos)
        if curpos is None:
            self.curpos = numpy.array(orgpos)
        else:
            self.curpos = numpy.array(curpos)
        self.neibrs = []
        self.clasif = clasif
        self.mass = 1
        self.respon = [0]
        self.availa = [0]
        self.inflnc = []

class nodebucket:
    
    def __init__(self, itnods, badge):
        self.nodlst = []
        self.badge = badge
        for node in itnods:
            self.collct(node)
        self.edgavg = 0
    
    def collct(self, node):
        self.nodlst.append(node)
        node.clasif = self.badge
    
    def calavgedg(self):
        edgsum = 0
        edgcnt = 0
        vstlst = []
        for node in self.nodlst:
            for neb in node.neibrs:
                if neb.clasif == self.badge:
                    if not neb in vstlst:
                        edgsum = edgsum + numpy.sum((neb.curpos - node.curpos)**2)**.5
                        edgcnt = edgcnt + 1
            vstlst.append(node)
        self.edgavg = edgsum / edgcnt
        return edgsum / edgcnt
    
    def calstdedg(self):
        edgsum = 0
        edgcnt = 0
        vstlst = []
        for node in self.nodlst:
            for neb in node.neibrs:
                if neb.clasif == self.badge:
                    if not neb in vstlst:
                        edgsum = edgsum + (numpy.sum((neb.curpos - node.curpos)**2)**.5 - self.edgavg)**2
                        edgcnt = edgcnt + 1
            vstlst.append(node)
        if edgcnt == 1:
            edgcnt = edgcnt + 1
        return (edgsum / (edgcnt - 1))**.5
    
    def findborder(self):
        brdlst = []
        for node in self.nodlst:
            isbrdr = False
            for neb in node.neibrs:
                if neb.clasif != self.badge:
                    isbrdr = True
            if isbrdr:
                brdlst.append(node)
        return brdlst

class nodecubby:
    
    def __init__(self, points, ptspbn):
        self.ptbins = []
        self.relpts = []
        self.stpszs = []
        self.minims = []
        minbns = numpy.ceil(len(points) / ptspbn)
        d = 1
        dims = points[0].curpos.shape[0]
        while d**dims < minbns:
            d = d + 1
        maxims = []
        for i in range(dims):
            self.minims.append(points[0].curpos[i])
            maxims.append(points[0].curpos[i])
        for pnt in points:
            for i in range(dims):
                if pnt.curpos[i] < self.minims[i]:
                    self.minims[i] = pnt.curpos[i]
                elif pnt.curpos[i] > maxims[i]:
                    maxims[i] = pnt.curpos[i]
        self.minims = numpy.array(self.minims)
        maxims = numpy.array(maxims)
        self.stpszs = (maxims - self.minims) / d
        # Old code to remind me of my hubris. To think this was possible 
        # without recursion
        #dumrow = []
        #dumarr = []
        #emprow = []
        #emparr = []
        #for dm in range(dims):
        #    dumrow = []
        #    emprow = []
        #    for i in range(d):
        #        dumrow.append(dumarr)
        #        emprow.append(emparr)
        #    dumarr = dumrow
        #    emparr = emprow
        self.ptbins = self.createndarray(d, dims)
        self.relpts = self.createndarray(d, dims)
        for pnt in points:
            indarr = numpy.floor((pnt.curpos - self.minims) / self.stpszs)
            #print(indarr)
            if int(indarr[0]) == d:
                curbuk = self.ptbins[int(indarr[0]) - 1]
            else:
                curbuk = self.ptbins[int(indarr[0])]
            for dm in range(1, dims):
                if indarr[dm] == d:
                    curbuk = curbuk[int(indarr[dm]) - 1]
                else:
                    curbuk = curbuk[int(indarr[dm])]
            curbuk.append(pnt)
    
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
        mainbk = (centrn.curpos - self.minims) / self.stpszs
        for i in range(mainbk.shape[0]):
            mainbk[i] = int(mainbk[i])
            if mainbk[i] == len(self.ptbins):
                mainbk[i] = mainbk[i] - 1
        # Next look through the bucket for all relevent comparison nodes, and 
        # if the bucket is empty, go looking for nodes
        relbuk = self.relpts
        for ind in mainbk:
            relbuk = relbuk[int(ind)]
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
                curbuk = self.ptbins
                inrang = True
                i = 0
                # Retrieve the bucket in question if it is in the array
                while i < curcor.shape[0] and inrang:
                    if curcor[i] >= 0 and curcor[i] < len(curbuk):
                        curbuk = curbuk[int(curcor[i])]
                        i = i + 1
                    else:
                        inrang = False
                # If it is in the array, add it to the list of buckets to draw nodes 
                # from, then find which buckets to draw nodes from next
                if inrang:
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

class pulltugger:
    
    def __init__(self):
        self.nodlst = []
        self.clstst = []
    
    def connect(self, nodea, nodeb):
        #edglen = ((nodea.xypost[0] - nodeb.xypost[0])**2 + (nodea.xypost[1] - nodeb.xypost[1])**2)**.5
        nodea.neibrs.append(nodeb)
        nodeb.neibrs.append(nodea)
        nodea.respon.append(0)
        nodeb.respon.append(0)
        nodea.availa.append(0)
        nodeb.availa.append(0)
        nodea.inflnc.append(1)
        nodeb.inflnc.append(1)
    
    def disconnect(self, nodea, nodeb):
        i = 0
        while i < len(nodeb.neibrs):
            if nodeb.neibrs[i] == nodea:
                nodeb.neibrs.pop(i)
            else:
                i = i + 1
        i = 0
        while i < len(nodea.neibrs):
            if nodea.neibrs[i] == nodeb:
                nodea.neibrs.pop(i)
            else:
                i = i + 1
    
    def simmintrebfsbuild(self, pntlst, slack = 1, cubsze = 20, scalng = False):
        # Code to pre-sort the input data before turning them into nodes in 
        # case it's relevant. It doesn't seem to help much.
        #pnthep = [pntlst[0]]
        #for i in range(1, pntlst.shape[0]):
        #    chcknd = len(pnthep) - 1
        #    stpsze = len(pnthep) / 2
        #    #print(f'Sort value: {edg[0]}')
        #    while stpsze > 0:
        #        #print(f'{chcknd}: {edghep[chcknd][0]}')
        #        if pntlst[i][0] < pnthep[chcknd - int(stpsze)][0]:
        #            chcknd = chcknd - int(stpsze)
        #            if stpsze > 1:
        #                stpsze = numpy.ceil(stpsze)
        #        stpsze = int(stpsze) / 2
        #    #print(f'Final check {chcknd}: {edghep[chcknd][0]}')
        #    if pntlst[i][0] > pnthep[chcknd][0]:
        #        pnthep.insert(chcknd + 1, pntlst[i])
        #    else:
        #        pnthep.insert(chcknd, pntlst[i])
        #srtlst = numpy.array(pnthep)
        #print('Array sorted')
        if scalng:
            scaler = StandardScaler()
            scaler.fit(pntlst)
            scldpt = scaler.transform(pntlst)
            tovist = []
            for k in range(len(pntlst)):
                self.nodlst.append(ndmovenode(pntlst[k], curpos = scldpt[k]))
                tovist.append(self.nodlst[-1])
        else:
            tovist = []
            for pnt in pntlst:
                self.nodlst.append(ndmovenode(pnt))
                tovist.append(self.nodlst[-1])
        # Now the nodes need to be organized such that they can be easily 
        # accessed as chunks to reduce the number of comparisons to be made
        nodorg = nodecubby(self.nodlst, cubsze)
        print('Node grid organized.')
        vstlst = []
        nxtlst = []
        curnod = tovist.pop(0)
        #print(f'Initial node: {curnod.curpos}')
        # One issue with the usability of this algorithm is that it scales 
        # horribly with n. I haven't worked out the O(n) yet, but based on 
        # test cases, it's like O(n!) or something - way more than O(n^3). 
        # Since this only cares about the nearest nodes, why not artificially 
        # reduce our data set to a fixed amount that is quick to manage? If we 
        # bin our points based on proximity, we should be able to make sure 
        # that we only ever compare with a subset of the data that should be 
        # sufficient to establish nearest neighbors. The method to retrieve 
        # comparison nodes would need to account for the fact that the density 
        # is not even in all directions, so it may need to draw from more bins 
        # in one direction than others
        potnbs = []
        avgerm = 0
        tocomp = nodorg.gatherchoices(curnod, cubsze / 2)
        avgerm = avgerm + len(tocomp)
        for k in range(len(tocomp)): # formerly tovist
            isneb = True
            toremv = []
            # The distance from the central node to the node being tested
            distk = numpy.sum((tocomp[k].curpos - curnod.curpos)**2) # formerly tovist
            # If the node being tested is the central node, then it is not a 
            # neighbor
            if tocomp[k] == curnod:
                isneb = False
            # If the node being tested is a duplicate of the central node 
            # (a.curpos == c.curpos), then it is a neighbor no matter what
            elif distk == 0:
                potnbs.append(tocomp[k])
            else:
                i = 0
                while i < len(potnbs) and isneb:
                    # The distance from the central node to the neighbor being 
                    # checked against
                    distn = numpy.sum((potnbs[i].curpos - curnod.curpos)**2)
                    # The distance from the neighbor being checked against to the 
                    # node being tested
                    distl = numpy.sum((tocomp[k].curpos - potnbs[i].curpos)**2) # formerly tovist
                    # As long as the comparison node is not a duplicate of 
                    # either then central or tested node
                    if not (distn == 0 or distl == 0):
                        # If the path length from the central node to the 
                        # tested node is greater than the sum of the path from 
                        # the central node to the checking neighbor and the 
                        # path from the checking neighbor to the tested node, 
                        # then it is not a neighbor
                        if distk * slack >= distn + distl:
                            isneb = False
                        # If the path length from the central node to the 
                        # checking neighbor is greater than the sum of the 
                        # path from the central node to the tested node and 
                        # the path from the the tested node to the checking 
                        # neighbor, then the checking neighbor should be 
                        # removed when then tested node is added to the list 
                        # of neighbors
                        if distn * slack >= distk + distl:
                            toremv.append(potnbs[i])
                    i = i + 1
                # I still need to compare against all possible nodes, even 
                # though no other nodes have been compared yet 
                i = 0
                while i < len(tocomp) and isneb:
                    # As long as the node being checked against is not either 
                    # of the other two nodes
                    if not (i == k or curnod == tocomp[i]):
                        distn = numpy.sum((tocomp[i].curpos - curnod.curpos)**2)
                        distl = numpy.sum((tocomp[k].curpos - tocomp[i].curpos)**2)
                        # As long as the node being checked against is not 
                        # a duplicate of either node
                        if not (distn == 0 or distl == 0):
                            if distk * slack >= distn + distl:
                                isneb = False
                    i = i + 1
                if isneb:
                    # Remove potential neighbors that should be removed
                    for badneb in toremv:
                        potnbs.remove(badneb)
                    potnbs.append(tocomp[k]) # formerly tovist
        # After the optimal set of neighbors are found, connect them and queue 
        # the neighbors up as the next nodes to connect
        #print(f'Identified neighbors:')
        for actneb in potnbs:
            #print(f'  {actneb.curpos}')
            self.connect(actneb, curnod)
            nxtlst.append(actneb)
        #print(curnod.neibrs)
        # Now, there should no longer be a need to question whether the node 
        # that was just connected needs to be reassessed as a neighbor for the 
        # neighbors it was just connected to or be assessed as a neighbor for 
        # any other nodes, so mark it as visited.
        vstlst.append(curnod)
        # Now repeat for all other nodes
        while len(tovist) > 0:  # formerly nxtlst, but I think I meant tovist because it needs to be done for all nodes, even if they aren't all connected, although they should be
        #for n in range(60):
            #print(len(nxtlst))
            if len(vstlst) % 100 == 0:
                print(f'The graph is {100 * len(vstlst) / len(self.nodlst):.2f}% connected.')
            if len(nxtlst) > 0:
                curnod = nxtlst.pop(0)
            else:
                curnod = tovist[0]
            # Because I will likely encounter the same node multiple times 
            # as a neighbor before handling it itself, I will need to make 
            # sure that the node I take from the next queue has not been 
            # visited.
            if not curnod in vstlst:
                #print(f'Current node: {curnod.curpos}')
                # Remove the current node from the list of nodes to be visited 
                # because it is currently being visited
                tovist.remove(curnod)
                # Reset the list of potential neighbors
                potnbs = []
                # Find the nearest nodes in all directions to compare with
                tocomp = nodorg.gatherchoices(curnod, cubsze / 2)
                avgerm = avgerm + len(tocomp)
                for k in range(len(tocomp)): # formerly tovist
                    isneb = True
                    toremv = []
                    distk = numpy.sum((tocomp[k].curpos - curnod.curpos)**2) # formerly tovist
                    # I need to make sure that the node is not attempting to 
                    # compare with itself, otherwise it will connect with 
                    # itself and exclude all others
                    if tocomp[k] == curnod:
                        isneb = False
                        #print(f'This is the node: {curnod.curpos}')
                    # However, if the nodes are duplicates of one another, 
                    # always connect them
                    elif distk == 0:
                        potnbs.append(tocomp[k])
                    else:
                        # Because I know that these nodes will have neighbors, 
                        # I need to make sure that the the potential neighbor 
                        # agrees with them. These are fixed, so they do not 
                        # need to be tested whether there is a conflict in the 
                        # other direction. However it is possible that they 
                        # will be the same node or a duplicate because I am 
                        # not only pulling from nodes that have not been 
                        # processed yet
                        for fxdneb in curnod.neibrs:
                            if not (tocomp[k] == fxdneb):
                                distn = numpy.sum((fxdneb.curpos - curnod.curpos)**2)
                                distl = numpy.sum((tocomp[k].curpos - fxdneb.curpos)**2) # formerly tovist
                                if not (distn == 0 or distl == 0):
                                    if distk * slack >= distn + distl:
                                        isneb = False
                                        #print(f'Conflicts with a current neighbor: {fxdneb.curpos}')
                            else:
                                isneb = False
                        # Now to check whether this is consistent with other 
                        # potential neighbors. These need to be reviewed for 
                        # conflicts and reject older ones that are less suited
                        i = 0
                        while i < len(potnbs) and isneb:
                            if not (tocomp[k] == potnbs[i]):
                                distn = numpy.sum((potnbs[i].curpos - curnod.curpos)**2)
                                distl = numpy.sum((tocomp[k].curpos - potnbs[i].curpos)**2) # formerly tovist
                                if not (distn == 0 or distl == 0):
                                    if distk * slack >= distn + distl:
                                        isneb = False
                                        #print(f'Conflicts with potential neighbors: {potnbs[i].curpos}')
                                    if distn * slack >= distk + distl:
                                        toremv.append(potnbs[i])
                                        #print(f'Potential neighbor conflicts with node: {potnbs[i].curpos}')
                            i = i + 1
                        # There is another issue in that it is possible for this 
                        # criteria to be asymmetric. If only considering already 
                        # connected neighbors and potential neighbors, it is 
                        # possible that one node would find another unacceptable, 
                        # and all of it's neighbors find it unacceptable, but for 
                        # the other node, based on its current and current 
                        # potential neighbors, to find that original node 
                        # acceptable. If that second node does not compare other 
                        # candidates against already visited nodes, or other nodes 
                        # already considered and rejected, it will believe that a 
                        # node behind the first node is one of its nearest 
                        # neighbors
                        # If I am searching through all of the surrounding nodes 
                        # as a final check, then maybe comparing against all 
                        # visited nodes is unnecessary and take too much time
# =============================================================================
#                         i = len(vstlst) - 1
#                         while i > -1 and isneb:
#                             if not (tocomp[k] == vstlst[i]):
#                                 distn = numpy.sum((vstlst[i].curpos - curnod.curpos)**2)
#                                 distl = numpy.sum((tocomp[k].curpos - vstlst[i].curpos)**2) # formerly tovist
#                                 if not (distn == 0 or distl == 0):
#                                     if distk * slack >= distn + distl:
#                                         isneb = False
#                                         #print(f'Conflicts with a node that has been visited: {vstlst[i].curpos}')
#                             i = i - 1
# =============================================================================
                        i = 0
                        while i < len(tocomp) and isneb: # formerly tovist
                            if not (i == k or tocomp[i] == curnod):
                                distn = numpy.sum((tocomp[i].curpos - curnod.curpos)**2) # formerly tovist
                                distl = numpy.sum((tocomp[k].curpos - tocomp[i].curpos)**2) # formerly tovist
                                if not (distn == 0 or distl == 0):
                                    if distk * slack >= distn + distl:
                                        isneb = False
                                        #print(f'Conflicts with an unvisited node: {tocomp[i].curpos}')
                            i = i + 1
                        if isneb:
                            for badneb in toremv:
                                potnbs.remove(badneb)
                            potnbs.append(tocomp[k]) # formerly tovist
                # Connect the nodes and add them to the queue
                #print(f'Identified neighbors:')
                for newneb in potnbs:
                    #print(f'  {newneb.curpos}')
                    self.connect(curnod, newneb)
                    if not (newneb in vstlst or newneb in nxtlst):
                        nxtlst.append(newneb)
                # Mark the current node as visited
                #print(curnod.neibrs)
                vstlst.append(curnod)
        print(avgerm / len(self.nodlst))
    
    def edgebuild(self, pntlst, scalng = True):
        # This builds a graph by looking for the k nearest (non-duplicate) 
        # nodes of each other point and making them neighbors
        if scalng:
            scaler = StandardScaler()
            scaler.fit(pntlst)
            scldpt = scaler.transform(pntlst)
            #tovist = []
            for k in range(len(pntlst)):
                self.nodlst.append(ndmovenode(pntlst[k], curpos = scldpt[k]))
                #tovist.append(self.nodlst[-1])
        else:
            #tovist = []
            for pnt in pntlst:
                self.nodlst.append(ndmovenode(pnt))
                #tovist.append(self.nodlst[-1])
        edgehp = []
        for i in range(len(self.nodlst) - 1):
            print(i)
            for j in range(i, len(self.nodlst)):
                heapq.heappush(edgehp, [numpy.sum((self.nodlst[i].curpos - self.nodlst[j].curpos)**2), i, j])
        while len(edgehp) > 0:
            if len(edgehp) % 100 == 0:
                print(f'There are {len(edgehp)} left to test')
            curedg = heapq.heappop(edgehp)
            if curedg[0] == 0:
                self.connect(self.nodlst[curedg[1]], self.nodlst[curedg[2]])
            else:
                nodone = self.nodlst[curedg[1]]
                nodtwo = self.nodlst[curedg[2]]
                concct = True
                for neb in nodone.neibrs:
                    if curedg[0] > numpy.sum((neb.curpos - nodone.curpos)**2) + numpy.sum((neb.curpos - nodtwo.curpos)**2):
                        concct = False
                for neb in nodtwo.neibrs:
                    if curedg[0] > numpy.sum((neb.curpos - nodone.curpos)**2) + numpy.sum((neb.curpos - nodtwo.curpos)**2):
                        concct = False
                if concct:
                    self.connect(self.nodlst[curedg[1]], self.nodlst[curedg[2]])
# =============================================================================
#         for i in range(len(self.nodlst)):
#             if i % 100 == 0:
#                 print(f'The graph is {100 * i / len(self.nodlst):.2f}% complete')
#             nerhep = []
#             duplst = []
#             for j in range(len(self.nodlst)):
#                 if j != i:
#                     nebdst = numpy.sum((self.nodlst[j].curpos - self.nodlst[i].curpos)**2)**.5
#                     if nebdst == 0 and not (self.nodlst[j] in duplst):
#                         duplst.append(self.nodlst[j])
#                     else:
#                         heapq.heappush(nerhep, [nebdst, self.nodlst[j]])
#             for dup in duplst:
#                 if not dup in self.nodlst[i].neibrs:
#                     self.connect(self.nodlst[i], dup)
#             for k in range(minneb):
#                 nerneb = heapq.heappop(nerhep)
#                 if not nerneb in self.nodlst[i].neibrs:
#                     self.connect(self.nodlst[i], nerneb[1])
# =============================================================================
    
    def standardscale(self):
        orgdat = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
        for i in range(len(self.nodlst)):
            orgdat[i] = self.nodlst[i].orgpos
        thsscl = StandardScaler()
        stddat = thsscl.fit_transform(orgdat)
        for i in range(len(self.nodlst)):
            self.nodlst[i].curpos = numpy.array(stddat[i])
    
    def pull(self, equips, tmestp, iterat):
        mincon = numpy.sum((self.nodlst[0].neibrs[0].curpos - self.nodlst[0].curpos)**2)**.5
        for node in self.nodlst:
            for neb in node.neibrs:
                if numpy.sum((neb.curpos - node.curpos)**2)**.5 < mincon:
                    mincon = numpy.sum((neb.curpos - node.curpos)**2)**.5
        mincon = mincon * equips
        for k in range(iterat):
            delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
            for i in range(len(self.nodlst)):
                force = numpy.zeros(self.nodlst[i].orgpos.shape)
                #repuls = False
                weight = 0
                for neb in self.nodlst[i].neibrs:
                    fdir = neb.curpos - self.nodlst[i].curpos
                    frad = numpy.sum(fdir**2)**.5
                    #if frad - mincon < 0:
                    #    repuls = True
                    fmag = ((frad / mincon)**-2 - (frad / mincon)**-3)
                    force = force + fmag * fdir * (1 - mincon / frad)
                    weight = weight + numpy.abs(fmag)
                #if repuls:
                #    if numpy.sum(force**2)**.5 * tmestp > mincon:
                #        force = force * mincon / (tmestp * numpy.sum(force**2)**.5)
                force = force / weight
                delpos[i] = delpos[i] + force * tmestp
            for i in range(len(self.nodlst)):
                self.nodlst[i].curpos = self.nodlst[i].curpos + delpos[i]
    
    def tug(self, equips, tmestp, iterat, decayr):
        mincon = numpy.sum((self.nodlst[0].neibrs[0].curpos - self.nodlst[0].curpos)**2)**.5
        for node in self.nodlst:
            for neb in node.neibrs:
                if numpy.sum((neb.curpos - node.curpos)**2)**.5 < mincon:
                    mincon = numpy.sum((neb.curpos - node.curpos)**2)**.5
        mincon = mincon * equips
        for k in range(iterat):
            delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
            self.masspass(decayr)
            for i in range(len(self.nodlst)):
                force = numpy.zeros(self.nodlst[i].orgpos.shape)
                #repuls = False
                weight = 0
                for neb in self.nodlst[i].neibrs:
                    #if neb.mass > self.nodlst[i].mass:
                    fdir = neb.curpos - self.nodlst[i].curpos
                    frad = numpy.sum(fdir**2)**.5
                        #if frad - mincon < 0:
                        #    repuls = True
                    fmag = neb.mass * ((frad / mincon)**-2 - (frad / mincon)**-3)
                    force = force + fmag * fdir * (1 - mincon / frad)
                    weight = weight + numpy.abs(fmag)
                #if repuls:
                #    if numpy.sum(force**2)**.5 * tmestp > mincon:
                #        force = force * mincon / (tmestp * numpy.sum(force**2)**.5)
                #if weight > 0:
                force = force / weight
                delpos[i] = delpos[i] + force * tmestp
            for i in range(len(self.nodlst)):
                self.nodlst[i].curpos = self.nodlst[i].curpos + delpos[i]
    
    def settle(self, equips, tmestp, tottim, numneb = 1):
        mincon = numpy.sum((self.nodlst[0].neibrs[0].curpos - self.nodlst[0].curpos)**2)**.5
        for node in self.nodlst:
            for neb in node.neibrs:
                if numpy.sum((neb.curpos - node.curpos)**2)**.5 < mincon:
                    mincon = numpy.sum((neb.curpos - node.curpos)**2)**.5
        mincon = mincon * equips
        print(mincon)
        delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
        i = 0
        while i < tottim:
            self.masssett(mincon, .5)
            delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
            for j in range(len(self.nodlst)):
                #fmag = 0
                fdir = numpy.zeros((self.nodlst[j].orgpos.shape[0]))
                minr = -1
                nebarr = []
                whtarr = []
                nxtarr = []
                nxtgen = []
                for k in range(len(self.nodlst[j].neibrs)):
                    nebarr.append(self.nodlst[j].neibrs[k])
                    whtarr.append(self.nodlst[j].inflnc[k])
                    nxtarr.append(self.nodlst[j].neibrs[k])
                n = 1
                while n < numneb:
                    while len(nxtarr) > 0:
                        curnod = nxtarr.pop(0)
                        for k in range(len(curnod.neibrs)):
                            if not (curnod.neibrs[k] in nebarr or curnod.neibrs[k] == self.nodlst[j]):
                                cosvec = numpy.dot((curnod.neibrs[k].curpos - self.nodlst[j].curpos), (curnod.neibrs[k].curpos - curnod.curpos)) / (numpy.sum((curnod.neibrs[k].curpos - self.nodlst[j].curpos)**2)**.5 * numpy.sum((curnod.neibrs[k].curpos - curnod.curpos)**2)**.5)
                                nxtgen.append([curnod.neibrs[k], curnod.inflnc[k] + curnod.inflnc[k] * cosvec - cosvec])
                    while len(nxtgen) > 0:
                        curpar = nxtgen.pop(0)
                        if curpar[0] in nebarr:
                            whtarr[nebarr.index(curpar[0])] = whtarr[nebarr.index(curpar[0])] + curpar[1]
                        else:
                            nebarr.append(curpar[0])
                            whtarr.append(curpar[1])
                    n = n + 1
                for k in range(len(nebarr)):
                    dvec = nebarr[k].curpos - self.nodlst[j].curpos
                    rmag = numpy.sum(dvec**2)**.5
                    mmag = whtarr[k]
                    #k = 0
                    #while k < len(neb.neibrs):
                    #    if neb.neibrs[k] == self.nodlst[j]:
                    #        mmag = neb.inflnc[k]
                    #        k = len(neb.neibrs)
                    #    else:
                    #        k = k + 1
                    #fmag = fmag + (mincon / rmag)**2 * (1 - (mincon / rmag))
                    fdir = fdir + dvec * (mincon / rmag**3) * (2 * mincon / rmag - 1) * mmag
                    if minr > rmag or minr == -1:
                        minr = rmag
                #delfor = fmag * fdir
                nrmmag = numpy.sum((fdir)**2)**.5
                delpos[j] = delpos[j] + minr * fdir / (2 * nrmmag)
            for j in range(len(self.nodlst)):
                self.nodlst[j].curpos = self.nodlst[j].curpos - tmestp * delpos[j]
            i = i + 1
    
    def pacify(self, equips, iterat, bias = 1):
        mincon = numpy.sum((self.nodlst[0].neibrs[0].curpos - self.nodlst[0].curpos)**2)**.5
        for node in self.nodlst:
            for neb in node.neibrs:
                if numpy.sum((neb.curpos - node.curpos)**2)**.5 < mincon:
                    mincon = numpy.sum((neb.curpos - node.curpos)**2)**.5
        mincon = mincon * equips
        delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
        i = 0
        while i < iterat:
            delpos = numpy.zeros((len(self.nodlst), self.nodlst[0].orgpos.shape[0]))
            #self.masssett(mincon, .5)
            self.densty(mincon)
            for j in range(len(self.nodlst)):
                #potscr = 0
                #newpos = self.nodlst[j].curpos
                posarr = numpy.zeros((len(self.nodlst[j].neibrs), self.nodlst[0].orgpos.shape[0]))
                scrarr = numpy.zeros((len(self.nodlst[j].neibrs)))
                # For each neighbor, find the equilibrium position the node 
                # would find for that neighbor and calculate the potential 
                # score at that position. Move halfway to the equilibrium 
                # position for whichever neighbor gives the best score
                for l in range(len(self.nodlst[j].neibrs)):
                    curdst = numpy.sum((self.nodlst[j].neibrs[l].curpos - self.nodlst[j].curpos)**2)**.5
                    potpos = self.nodlst[j].curpos + (curdst - mincon) * (self.nodlst[j].neibrs[l].curpos - self.nodlst[j].curpos) / curdst
                    posscr = 0
                    for k in range(len(self.nodlst[j].neibrs)):
                        posscr = posscr + self.nodlst[j].mass * mincon**2 / numpy.sum((self.nodlst[j].neibrs[k].curpos - potpos)**2)
                    #if posscr > potscr:
                    #    potscr = posscr
                    #    newpos = (self.nodlst[j].curpos + potpos) / 2
                    posarr[l] = potpos
                    scrarr[l] = posscr**bias
                newpos = numpy.zeros(posarr.shape[1])
                for k in range(len(posarr)):
                    newpos = newpos + posarr[k] * scrarr[k]
                newpos = newpos / numpy.sum(scrarr)
                delpos[j] = newpos
            for j in range(len(self.nodlst)):
                self.nodlst[j].curpos = delpos[j]
            i = i + 1
    
    def masspass(self, decayr, iterat = 1):
        totmas = 0
        maslst = numpy.zeros((len(self.nodlst)))
        j = 0
        while j < iterat:
            for i in range(len(self.nodlst)):
                maslst[i] = self.nodlst[i].mass
                for neb in self.nodlst[i].neibrs:
                    maslst[i] = maslst[i] + neb.mass * numpy.exp(-1 * numpy.sum((neb.curpos - self.nodlst[i].curpos)**2)**.5 / (2 * decayr**2))
                totmas = totmas + maslst[i]
            maslst = maslst / totmas
            for i in range(len(self.nodlst)):
                self.nodlst[i].mass = maslst[i]
            j = j + 1
    
    def masscont(self, equips):
        #curmas = 0
        delmas = numpy.zeros((len(self.nodlst)))
        for i in range(len(self.nodlst)):
            summas = 1
            for neb in self.nodlst[i].neibrs:
                mainnr = numpy.sum((self.nodlst[i].curpos - neb.curpos)**2)**.5
                otherr = 0
                for nneb in neb.neibrs:
                    otherr = otherr + numpy.sum((neb.curpos - nneb.curpos)**2)**.5
                summas = summas + (equips / mainnr) * (neb.mass - self.nodlst[i].mass * mainnr / (1 + otherr))
            delmas[i] = summas
        for i in range(len(self.nodlst)):
            self.nodlst[i].mass = delmas[i]
        #    curmas = curmas + delmas[i]
        #prvmas = curmas * 2.1
        #while (numpy.abs(prvmas - curmas) / prvmas) > .00001:
        #    prvmas = curmas
        #    curmas = 0
        #    delmas = numpy.zeros((len(self.nodlst)))
        #    for i in range(len(self.nodlst)):
        #        summas = 1
        #        for neb in self.nodlst[i].neibrs:
        #            mainnr = numpy.sum((self.nodlst[i].curpos - neb.curpos)**2)**.5
        #            otherr = 0
        #            for nneb in neb.neibrs:
        #                otherr = otherr + numpy.sum((neb.curpos - nneb.curpos)**2)**.5
        #            summas = summas + (equips / mainnr) * (neb.mass - self.nodlst[i].mass * mainnr / (1 + otherr))
        #        delmas[i] = summas
        #    for i in range(len(self.nodlst)):
        #        self.nodlst[i].mass = delmas[i]
        #        curmas = curmas + delmas[i]
        #    print(curmas)
    
    def masssett(self, equips, speed):
        prvmas = len(self.nodlst)
        curmas = len(self.nodlst) / 2
        itrcnt = 0
        while numpy.abs(prvmas - curmas) / prvmas > .001:
            infupd = []
            #masupd = []
            for node in self.nodlst:
                ninfup = []
                totpul = 0
                #masscr = 1
                for neb in node.neibrs:
                    #masscr = masscr + neb.mass * equips / numpy.sum((neb.curpos - node.curpos)**2)
                    totpul = 1
                    gudvec = neb.curpos - node.curpos
                    for i in range(len(neb.neibrs)):
                        if numpy.dot((neb.neibrs[i].curpos - neb.curpos), gudvec) > 0:
                            totpul = totpul + neb.inflnc[i] * equips * numpy.dot((neb.neibrs[i].curpos - neb.curpos), gudvec) / (numpy.sum((neb.neibrs[i].curpos - neb.curpos)**2) * numpy.sum(gudvec**2)**.5)
                    ninfup.append(totpul)
                infupd.append(ninfup)
                #masupd.append(masscr)
            prvmas = curmas
            curmas = 0
            for i in range(len(self.nodlst)):
                #print(infupd[i])
                for j in range(len(self.nodlst[i].neibrs)):
                    self.nodlst[i].inflnc[j] = infupd[i][j]
                    curmas = curmas + infupd[i][j] - 1
            itrcnt = itrcnt + 1
            #print(curmas)
        #print(itrcnt)
        #for node in self.nodlst:
        #    i = 0
        #    while i < len(node.neibrs):
        #        j = 0
        #        while j < len(node.neibrs[i].neibrs):
        #            if node.neibrs[i].neibrs[j] == node:
        #                print([node.resist, node.inflnc[i], node.neibrs[i].resist, node.neibrs[i].inflnc[j]])
        #                j = len(node.neibrs[i].neibrs)
        #            else:
        #                j = j + 1
        #        i = i + 1
    
    def densty(self, equips):
        for node in self.nodlst:
            avgrad = 0
            for neb in node.neibrs:
                avgrad = avgrad + numpy.sum((neb.curpos - node.curpos)**2)**.5
            avgrad = avgrad / len(node.neibrs)
            node.mass = equips / avgrad
    
    def affpro(self, speed, length):
        k = 0
        while k < length:
            resupd = []
            avaupd = []
            for node in self.nodlst:
                nresup = []
                navaup = []
                similr = .5
                # The responsibility of each node, including itself, to another is 
                # the similarity of that node to the other, 1 / (1 - r), minus the 
                # highest sum of availability and similarity of the primary node 
                # to all other nodes 
                bigscr = 1 / (1 + numpy.sum((node.curpos - node.neibrs[0].curpos)**2)**.5) + node.availa[1]
                for i in range(len(node.neibrs)):
                    scr = 1 / (1 + numpy.sum((node.curpos - node.neibrs[i].curpos)**2)**.5) + node.availa[i + 1]
                    if scr > bigscr:
                        bigscr = scr
                nresup.append(similr - bigscr)
                for i in range(len(node.neibrs)):
                    similr = 1 / (1 + numpy.sum((node.curpos - node.neibrs[i].curpos)**2)**.5)
                    bigscr = .5 + node.availa[0]
                    for j in range(len(node.neibrs)):
                        if not i == j:
                            scr = 1 / (1 + numpy.sum((node.curpos - node.neibrs[j].curpos)**2)**.5) + node.availa[j + 1]
                            if scr > bigscr:
                                bigscr = scr
                    nresup.append(similr - bigscr)
                # The availability of one node to another is the responsibility of 
                # the other node to itself, plus the sum of the responsibility of 
                # each of the other node's neighbors to the other node if those 
                # responsibilities are positive
                scr = 0
                for neb in node.neibrs:
                    i = 0
                    while i < len(neb.neibrs):
                        if neb.neibrs[i] == node:
                            if neb.respon[i + 1] > 0:
                                scr = scr + neb.respon[i + 1]
                            i = len(neb.neibrs)
                        else:
                            i = i + 1
                navaup.append(scr)
                for neb in node.neibrs:
                    scr = neb.respon[0]
                    for soneb in neb.neibrs:
                        i = 0
                        while i < len(soneb.neibrs):
                            if soneb.neibrs[i] == neb:
                                if soneb.respon[i + 1] > 0:
                                    scr = scr + soneb.respon[i + 1]
                                i = len(soneb.neibrs)
                            else:
                                i = i + 1
                    if scr > 0:
                        navaup.append(0)
                    else:
                        navaup.append(scr)
                resupd.append(nresup)
                avaupd.append(navaup)
            for i in range(len(self.nodlst)):
                for j in range(len(self.nodlst[i].respon)):
                    self.nodlst[i].respon[j] = speed * self.nodlst[i].respon[j] + (1 - speed) * resupd[i][j]
                    self.nodlst[i].availa[j] = speed * self.nodlst[i].availa[j] + (1 - speed) * avaupd[i][j]
                self.nodlst[i].mass = self.nodlst[i].respon[0] + self.nodlst[i].availa[0]
            k = k + 1
    
    def edgestat(self):
        edglgn = []
        #vstlst = [] # A
        #nxtlst = [self.nodlst[0]] # A
        avglen = 0
        #while len(nxtlst) > 0: # A
        #    curnod = nxtlst.pop(0) # A
        #    if not curnod in vstlst: # A
        #        for neb in curnod.neibrs: # A
        #            if not neb in vstlst: # A
        #                edglgn.append(numpy.sum((neb.curpos - curnod.curpos)**2)**.5) # A
        #                nxtlst.append(neb) # A
        #                avglen = avglen + numpy.sum((neb.curpos - curnod.curpos)**2)**.5 # A
        #        vstlst.append(curnod) # A
        for node in self.nodlst: # B
            for neb in node.neibrs: # B
                avglen = avglen + numpy.sum((neb.curpos - node.curpos)**2)**.5 # B
                edglgn.append(numpy.sum((neb.curpos - node.curpos)**2)**.5) # B
        avglen = avglen / len(edglgn)
        edgstd = 0
        for edg in edglgn:
            edgstd = edgstd + (edg - avglen)**2
        edgstd = (edgstd / (len(edglgn) - 1))**.5
        return [avglen, edgstd]
    
    def edgsep(self, thresh):
        for node in self.nodlst:
            #change = True
            avgrad = 0
            for neb in node.neibrs:
                #if numpy.sum((neb.curpos - node.curpos)**2)**.5 > thresh:
                #    change = False
                avgrad = avgrad + numpy.sum((neb.curpos - node.curpos)**2)**.5
            avgrad = avgrad / len(node.neibrs)
            if avgrad < thresh:
                node.clasif = 0
        mpl.figure()
        vstlst = []
        scatar = [[], []]
        for node in self.nodlst:
            if node.clasif == -1:
                scatar[0].append(node.curpos[0])
                scatar[1].append(node.curpos[1])
            else:
                for neb in node.neibrs:
                    if node.clasif == 0 and neb.clasif == 0:
                        if not neb in vstlst:
                            mpl.plot([neb.curpos[0], node.curpos[0]], [neb.curpos[1], node.curpos[1]])
        mpl.scatter(scatar[0], scatar[1])
    
    def cluster(self, combfc = 1):
        self.clstst = []
        clstnd = 0
        edghep = []
        vstlst = []
        # First, compile a list of edges and sort them according to length
        for node in self.nodlst:
            for neb in node.neibrs:
                if not neb in vstlst:
                    edg = [numpy.sum((neb.curpos - node.curpos)**2)**.5, node, neb]
                    if len(edghep) == 0:
                        edghep.append(edg)
                    else:
                        chcknd = len(edghep) - 1
                        stpsze = len(edghep) / 2
                        #print(f'Sort value: {edg[0]}')
                        while stpsze > 0:
                            #print(f'{chcknd}: {edghep[chcknd][0]}')
                            if edg[0] < edghep[chcknd - int(stpsze)][0]:
                                chcknd = chcknd - int(stpsze)
                                if stpsze > 1:
                                    stpsze = numpy.ceil(stpsze)
                            stpsze = int(stpsze) / 2
                        #print(f'Final check {chcknd}: {edghep[chcknd][0]}')
                        if edg[0] > edghep[chcknd][0]:
                            edghep.insert(chcknd + 1, edg)
                        else:
                            edghep.insert(chcknd, edg)
            vstlst.append(node)
        print(f'Edges compiled: {len(edghep)} total')
        for edg in edghep:
            #print(edg[0])
            if edg[1].clasif == -1 and edg[2].clasif == -1:
                # If both are unlabeled, create a new cluster
                self.clstst.append(nodebucket([edg[1], edg[2]], clstnd))
                clstnd = clstnd + 1
            elif edg[1].clasif == -1:
                # If one is unlabeled, add it to the cluster of the other
                bdgnum = edg[2].clasif
                corcst = None
                for clst in self.clstst:
                    if clst.badge == bdgnum:
                        corcst = clst
                if corcst == None:
                    print(f'Cluster not found while adding')
                corcst.collct(edg[1])
            elif edg[2].clasif == -1:
                bdgnum = edg[1].clasif
                corcst = None
                for clst in self.clstst:
                    if clst.badge == bdgnum:
                        corcst = clst
                if corcst == None:
                    print(f'Cluster not found while adding')
                corcst.collct(edg[2])
            elif not edg[1].clasif == edg[2].clasif:
                # If both clusters have a label, test to see if they should be 
                # merged
                clusta = None
                clustb = None
                for clst in self.clstst:
                    if edg[1].clasif == clst.badge:
                        clusta = clst
                    elif edg[2].clasif == clst.badge:
                        clustb = clst
                if clusta == None or clustb == None:
                    print(f'Cluster not found while merging')
                    print([edg[1].clasif, edg[2].clasif])
                # Update each cluster's average and standard deviation, then 
                # compare them to the edge being considered
                amet = clusta.calavgedg()
                bmet = clustb.calavgedg()
                amet = clusta.calstdedg()
                bmet = clustb.calstdedg()
                if edg[0] < amet * combfc and edg[0] < bmet * combfc:
                    if clusta.badge > clustb.badge:
                        for oldnod in clusta.nodlst:
                            clustb.collct(oldnod)
                        self.clstst.remove(clusta)
                    else:
                        for oldnod in clustb.nodlst:
                            clusta.collct(oldnod)
                        self.clstst.remove(clustb)
    
    def searchcluster(self):
        mxnods = []
        for node in self.nodlst:
            locmax = True
            for neb in node.neibrs:
                if node.mass < neb.mass:
                    locmax = False
            if locmax:
                if len(mxnods) == 0:
                    mxnods.append(node)
                else:
                    chcknd = len(mxnods) - 1
                    stpsze = len(mxnods) / 2
                    #print(f'Sort value: {edg[0]}')
                    while stpsze > 0:
                        #print(f'{chcknd}: {edghep[chcknd][0]}')
                        if node.mass < mxnods[chcknd - int(stpsze)].mass:
                            chcknd = chcknd - int(stpsze)
                            if stpsze > 1:
                                stpsze = numpy.ceil(stpsze)
                        stpsze = int(stpsze) / 2
                    #print(f'Final check {chcknd}: {edghep[chcknd][0]}')
                    if node.mass > mxnods[chcknd].mass:
                        mxnods.insert(chcknd + 1, node)
                    else:
                        mxnods.insert(chcknd, node)
        clstin = -1
        nxtarr = []
        for i in range(len(mxnods) - 1, -1, -1):
            clstin = clstin + 1
            crclst = nodebucket([mxnods[i]], clstin)
            nxtarr = []
            for neb in mxnods[i].neibrs:
                if neb.clasif == -1:
                    nxtarr.append(neb)
                    crclst.collct(neb)
            while len(nxtarr) > 0:
                curnod = nxtarr.pop(0)
                for neb in curnod.neibrs:
                    if neb.clasif == -1 and neb.mass <= curnod.mass:
                        nxtarr.append(neb)
                        crclst.collct(neb)
            self.clstst.append(crclst)
    
    def edgecluster(self, thresh, hrdsft = -1, modeone = 'average', modetwo = 'average'):
        #clscnt = 0
        print(f'Threshold: {thresh}')
        self.clstst = []
        # First, create clusters based on nodes whose average separation 
        # distance with their neighbors are less than the threshold.
        k = 0
        for node in self.nodlst:
            if k % 100 == 0:
                print(f'The graph is {100 * k / len(self.nodlst):.2f}% initially classified.')
            k = k + 1
            # If the node is uncategorized...
            if node.clasif == -1:
                # ...check to see if it meets the criteria for being a core  
                # node for a cluster.
                iscent = False
                if hrdsft > 0:
                    # If using the harsh-soft dual criteria, use the threshold 
                    # defined by hrdsft
                    if modeone == 'average':
                        avgrad = 0
                        for neb in node.neibrs:
                            avgrad = avgrad + numpy.sum((neb.curpos - node.curpos)**2)**.5
                        avgrad = avgrad / len(node.neibrs)
                        if avgrad < hrdsft:
                            iscent = True
                    else:
                        iscent = True
                        for neb in node.neibrs:
                            if numpy.sum((neb.curpos - node.curpos)**2)**.5 > hrdsft:
                                iscent = False
                    if iscent:
                        print(node.curpos)
                        for neb in node.neibrs:
                            print(f'Neighbor: {neb.curpos}, {numpy.sum((node.curpos - neb.curpos)**2)**.5:.2f}')
                else:
                    # If using the single threshold, check to see if the 
                    # average distance to a neighbor falls below the 
                    # threshold.
                    if modeone == 'average':
                        avgrad = 0
                        for neb in node.neibrs:
                            avgrad = avgrad + numpy.sum((neb.curpos - node.curpos)**2)**.5
                        avgrad = avgrad / len(node.neibrs)
                        if avgrad < thresh:
                            iscent = True
                    else:
                        iscent = True
                        for neb in node.neibrs:
                            if numpy.sum((neb.curpos - node.curpos)**2)**.5 > thresh:
                                iscent = False
                    if iscent:
                        print(node.curpos)
                        for neb in node.neibrs:
                            print(f'Neighbor: {neb.curpos}, {numpy.sum((node.curpos - neb.curpos)**2)**.5:.2f}')
                # If the criteria is met,...
                if iscent:
                    # create a new cluster,...
                    newcls = nodebucket([node], len(self.clstst))
                    self.clstst.append(newcls)
                    # then begin searching neighboring nodes for other 
                    # nodes that meet the criteria.
                    nxtnds = []
                    for neb in node.neibrs:
                        if neb.clasif == -1:
                            nxtnds.append(neb)
                    while len(nxtnds) > 0:
                        curnod = nxtnds.pop(0)
                        # If the node still has not been categorized, because 
                        # it is possible for a node to be added to the up-next  
                        # queue multiple times before being visited,...
                        if curnod.clasif == -1:
                            # ...check to see if it meets the criteria.
                            if modetwo == 'average':
                                avgrad = 0
                                for curneb in curnod.neibrs:
                                    avgrad = avgrad + numpy.sum((curneb.curpos - curnod.curpos)**2)**.5
                                avgrad = avgrad / len(curnod.neibrs)
                                # If it does meet the criteria,...
                                if avgrad < thresh:
                                    # ...add it to the cluster,...
                                    newcls.collct(curnod)
                                    # ...then queue up its neighbors to be 
                                    # investigated as well
                                    for curneb in curnod.neibrs:
                                        if curneb.clasif == -1:
                                            nxtnds.append(curneb)
                            else:
                                shldad = True
                                for curneb in curnod.neibrs:
                                    if numpy.sum((curneb.curpos - curnod.curpos)**2)**.5 > thresh:
                                        shldad = False
                                # If it does meet the criteria
                                if shldad:
                                    # ...add it to the cluster,...
                                    newcls.collct(curnod)
                                    # ...then queue up its neighbors to be 
                                    # investigated as well
                                    for curneb in curnod.neibrs:
                                        if curneb.clasif == -1:
                                            nxtnds.append(curneb)
        # Next, categorize all uncategorized nodes.
        pnextq = []
        for clster in self.clstst:
            brdnds = clster.findborder()
            for node in brdnds:
                for neb in node.neibrs:
                    if neb.clasif == -1:
                        heapq.heappush(pnextq, [numpy.sum((neb.curpos - node.curpos)**2)**.5, neb])
        while len(pnextq) > 0:
            curcon = heapq.heappop(pnextq)
            if curcon[1].clasif == -1:
                corcls = None
                for neb in curcon[1].neibrs:
                    if neb.clasif != -1:
                        if corcls == None:
                            corcls = neb
                        else:
                            if numpy.sum((neb.curpos - curcon[1].curpos)**2) < numpy.sum((corcls.curpos - curcon[1].curpos)**2):
                                corcls = neb
                    else:
                        heapq.heappush(pnextq, [numpy.sum((neb.curpos - curcon[1].curpos)**2)**.5, neb])
                self.clstst[corcls.clasif].collct(curcon[1])
    
    def nnearestneighbordistance(self, sphere = 1, noisy = False):
        self.clstst = []
        if noisy:
            xdata = []
            ydata = []
            zdata = []
        averagemass = 0
        for i in range(len(self.nodlst)):
            visited = []
            uplist = []
            crust = [self.nodlst[i]]
            averagedistance = 0
            for j in range(sphere):
                uplist = crust
                crust = []
                while len(uplist) > 0:
                    curnod = uplist.pop(0)
                    visited.append(curnod)
                    for neb in curnod.neibrs:
                        srchng = True
                        for nod in visited:
                            if neb.curpos[0] == nod.curpos[0] and neb.curpos[1] == nod.curpos[1]:
                                srchng = False
                        for nod in uplist:
                            if neb.curpos[0] == nod.curpos[0] and neb.curpos[1] == nod.curpos[1]:
                                srchng = False
                        for nod in crust:
                            if neb.curpos[0] == nod.curpos[0] and neb.curpos[1] == nod.curpos[1]:
                                srchng = False
                        if srchng:
                            crust.append(neb)
            for nod in crust:
                averagedistance = averagedistance + ((nod.curpos[0] - self.nodlst[i].curpos[0])**2 + (nod.curpos[1] - self.nodlst[i].curpos[1])**2)**.5
            averagedistance = averagedistance / len(crust)
            self.nodlst[i].mass = averagedistance
            averagemass = averagemass + averagedistance
            if noisy:
                xdata.append(self.nodlst[i].curpos[0])
                ydata.append(self.nodlst[i].curpos[1])
                zdata.append(averagedistance)
        #print(numpy.average(numpy.array(zdata)))
        if noisy:
            dig = mpl.figure()
            axe = dig.add_subplot(111, projection = '3d')
            axe.scatter(xdata, ydata, zdata)
        #clstcn = []
        averagemass = averagemass / len(self.nodlst)
        stdmass = 0
        for i in range(len(self.nodlst)):
            stdmass = stdmass + (self.nodlst[i].mass - averagemass)**2
        stdmass = stdmass / (len(self.nodlst) - 1)
        for i in range(len(self.nodlst)):
            curnod = self.nodlst[i]
            if curnod.mass <= averagemass - stdmass and curnod.clasif == -1:
                newcls = nodebucket([curnod], len(self.clstst))
                self.clstst.append(newcls)
                nxtque = []
                for neb in curnod.neibrs:
                    if neb.mass <= averagemass - stdmass and neb.clasif == -1:
                        nxtque.append(neb)
                while len(nxtque) > 0:
                    upnode = nxtque.pop(0)
                    if upnode.clasif == -1:
                        newcls.collct(upnode)
                        for neb in upnode.neibrs:
                            if neb.mass <= averagemass - stdmass and neb.clasif == -1:
                                nxtque.append(neb)
        unclassified = []
        for i in range(len(self.clstst)):
            border = self.clstst[i].findborder()
            for brdnod in border:
                for neb in brdnod.neibrs:
                    if neb.clasif == -1:
                        if len(unclassified) > 0:
                            pivot = len(unclassified) - 1
                            pvtlen = len(unclassified) / 2
                            while pvtlen > 1:
                                if numpy.sum((neb.curpos - brdnod.curpos)**2)**.5 <= unclassified[pivot - int(numpy.floor(pvtlen))][1]:
                                    pivot = pivot - int(numpy.floor(pvtlen))
                                    pvtlen = numpy.ceil(pvtlen) / 2
                                else:
                                    pvtlen = numpy.floor(pvtlen) / 2
                            if numpy.sum((neb.curpos - brdnod.curpos)**2)**.5 <= unclassified[pivot - int(numpy.floor(pvtlen))][1]:
                                unclassified.insert(pivot, [neb, numpy.sum((neb.curpos - brdnod.curpos)**2)**.5])
                            else:
                                unclassified.append([neb, numpy.sum((neb.curpos - brdnod.curpos)**2)**.5])
                        else:
                            unclassified.append([neb, numpy.sum((neb.curpos - brdnod.curpos)**2)**.5])
        while len(unclassified) > 0:
            curnod = unclassified.pop(0)
            if curnod[0].clasif == -1:
                closest = None
                for neb in curnod[0].neibrs:
                    if neb.clasif != -1:
                        if closest == None:
                            closest = neb
                        else:
                            if numpy.sum((neb.curpos - curnod[0].curpos)**2) < numpy.sum((closest.curpos - curnod[0].curpos)**2):
                                closest = neb
                    else:
                        if len(unclassified) > 0:
                            pivot = len(unclassified) - 1
                            pvtlen = len(unclassified) / 2
                            while pvtlen > 1:
                                if numpy.sum((neb.curpos - curnod[0].curpos)**2)**.5 <= unclassified[pivot - int(numpy.floor(pvtlen))][1]:
                                    pivot = pivot - int(numpy.floor(pvtlen))
                                    pvtlen = numpy.ceil(pvtlen) / 2
                                else:
                                    pvtlen = numpy.floor(pvtlen) / 2
                            if numpy.sum((neb.curpos - curnod[0].curpos)**2)**.5 <= unclassified[pivot - int(numpy.floor(pvtlen))][1]:
                                unclassified.insert(pivot, [neb, numpy.sum((neb.curpos - curnod[0].curpos)**2)**.5])
                            else:
                                unclassified.append([neb, numpy.sum((neb.curpos - curnod[0].curpos)**2)**.5])
                        else:
                            unclassified.append([neb, numpy.sum((neb.curpos - curnod[0].curpos)**2)**.5])
                self.clstst[closest.clasif].collct(curnod[0])
    
    def dbscancluster(self, sepdst = -1, minsmp = 10, scaling = True):
        datlst = []
        if scaling:
            for node in self.nodlst:
                datlst.append(node.curpos)
        else:
            for node in self.nodlst:
                datlst.append(node.orgpos)
        if sepdst == -1:
            kdstar = numpy.zeros((len(self.nodlst), int(numpy.max([numpy.log10(len(self.nodlst)), minsmp]))))
            for i in range(kdstar.shape[0]):
                vstlst = [self.nodlst[i]]
                dstlst = []
                curnod = self.nodlst[i]
                nxtlst = []
                for neb in curnod.neibrs:
                    nxtlst.append(neb)
                while len(nxtlst) > 0:
                    curnod = nxtlst.pop(0)
                    vstlst.append(curnod)
                    if scaling:
                        heapq.heappush(dstlst, numpy.sum((self.nodlst[i].curpos - curnod.curpos)**2)**.5)
                    else:
                        heapq.heappush(dstlst, numpy.sum((self.nodlst[i].orgpos - curnod.orgpos)**2)**.5)
                    if len(dstlst) < kdstar.shape[1]:
                        for neb in curnod.neibrs:
                            if not (neb in vstlst or neb in nxtlst):
                                nxtlst.append(neb)
                for j in range(kdstar.shape[1]):
                    kdstar[i][j] = heapq.heappop(dstlst)
            #print(kdstar)
            kdstar = numpy.sort(kdstar, axis = 0)
            xarray = numpy.arange(kdstar.shape[0])
            mpl.figure()
            for q in range(kdstar.shape[1]):
                mpl.plot(xarray, kdstar[:, q], label = f'{q}')
            mpl.legend()
            bestk = 1
            beste = numpy.max(kdstar[:, -1])
            besti = -1
            totint = numpy.sum(kdstar, axis = 0)
            print(totint)
            for j in range(kdstar.shape[1]):
                prtint = kdstar[0][j]
                i = 0
                while prtint / totint[j] < .5:
                    i = i + 1
                    prtint = prtint + kdstar[i][j]
                if i > besti:
                    besti = i
                    bestk = j + 1
                    beste = kdstar[i - 1][j]
            print([bestk, beste])
            dbclst = DBSCAN(eps = beste, min_samples = bestk + 1)
        else:
            dbclst = DBSCAN(eps = sepdst, min_samples = minsmp + 1)
        dbclst.fit(datlst)
        nmclst = numpy.max(dbclst.labels_) + 1
        self.clstst = []
        for i in range(int(nmclst)):
            self.clstst.append(nodebucket([], i))
        for i in range(len(self.nodlst)):
            if dbclst.labels_[i] > -1:
                self.clstst[dbclst.labels_[i]].collct(self.nodlst[i])
    
    def kmeanscluster(self, nmclst, scaling = True):
        datlst = []
        if scaling:
            for node in self.nodlst:
                datlst.append(node.curpos)
        else:
            for node in self.nodlst:
                datlst.append(node.orgpos)
        kmclst = MiniBatchKMeans(n_clusters = nmclst)
        kmclst.fit(numpy.array(datlst))
        self.clstst = []
        for i in range(nmclst):
            self.clstst.append(nodebucket([], i))
        for i in range(len(self.nodlst)):
            if kmclst.labels_[i] > -1:
                self.clstst[kmclst.labels_[i]].collct(self.nodlst[i])
    
    def heirarchicalcluster(self, nmclst = 2, mxdist = None, scaling = True, mode = 'ward'):
        datlst = []
        if scaling:
            for node in self.nodlst:
                datlst.append(node.curpos)
        else:
            for node in self.nodlst:
                datlst.append(node.orgpos)
        hrclst = AgglomerativeClustering(n_clusters = nmclst, distance_threshold = mxdist)
        hrclst.fit(numpy.array(datlst))
        self.clstst = []
        for i in range(hrclst.n_clusters_):
            self.clstst.append(nodebucket([], i))
        for i in range(len(self.nodlst)):
            if hrclst.labels_[i] > -1:
                self.clstst[hrclst.labels_[i]].collct(self.nodlst[i])
    
    def centerofdensity(self, equips):
        self.densty(equips)
        codarr = []
        for clst in self.clstst:
            cod = numpy.zeros(self.nodlst[0].orgpos.shape)
            totdns = 0
            for node in clst.nodlst:
                cod = cod + node.orgpos * node.mass
                totdns = totdns + node.mass
            cod = cod / totdns
            codarr.append(cod)
        return codarr
    
    def disply(self):
        vstlst = []
        scatar = [[], []]
        colors = [(numpy.random.rand(), numpy.random.rand(), numpy.random.rand())]
        for n in range(len(self.clstst)):
            colors.append((numpy.random.rand(), numpy.random.rand(), numpy.random.rand()))
        mpl.figure()
        for node in self.nodlst:
            scatar[0].append(node.curpos[0])
            scatar[1].append(node.curpos[1])
            for neb in node.neibrs:
                if not neb in vstlst and node.clasif == neb.clasif:
                    mpl.plot([node.curpos[0], neb.curpos[0]], [node.curpos[1], neb.curpos[1]], color = colors[node.clasif])
            vstlst.append(node)
            #print(node.clasif)
        mpl.scatter(scatar[0], scatar[1], marker = '.')
        mpl.show()
    
    def masplt(self):
        poslst = numpy.zeros((len(self.nodlst), 2))
        maslst = numpy.zeros((len(self.nodlst)))
        #poslst = []
        #maslst = []
        for i in range(len(self.nodlst)):
            for j in range(len(self.nodlst[i].neibrs)):
                poslst[i][0] = self.nodlst[i].curpos[0]
                poslst[i][1] = self.nodlst[i].curpos[1]
                maslst[i] = self.nodlst[i].mass
                #poslst.append([self.nodlst[i].curpos[0] + (self.nodlst[i].neibrs[j].curpos[0] - self.nodlst[i].curpos[0]) / 3, self.nodlst[i].curpos[1] + (self.nodlst[i].neibrs[j].curpos[1] - self.nodlst[i].curpos[1]) / 3])
                #maslst.append(self.nodlst[i].inflnc[j])
        #poslst = numpy.array(poslst)
        #maslst = numpy.array(maslst)
        dig = mpl.figure()
        axe = dig.add_subplot(111, projection = '3d')
        axe.scatter(poslst[:, 0], poslst[:, 1], maslst)
    
    def labelrep(self):
        outarr = []
        for i in range(len(self.clstst)):
            for node in self.clstst[i].nodlst:
                nstnod = []
                for val in node.orgpos:
                    nstnod.append(val)
                nstnod.append(i)
                outarr.append(nstnod)
        return outarr

if __name__ == '__main__':
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    if ftype == 'csv':
        imgdat = numpy.genfromtxt(flname, delimiter = ',')
    elif ftype == 'jpg':
        imgdat = imageio.imread(flname)
    start = time.time()
    scatmp = pulltugger()
    scatmp.simmintrebfsbuild(imgdat, cubsze = 20, scalng = True)
    #scatmp.edgebuild(imgdat, scalng = True)
    #scatmp.disply()
    #scatmp.standardscale()
    #scatmp.settle(.5, .5, 20, numneb = 1)
    #scatmp.masssett(.002504617, .5)
    #scatmp.densty(.0016697)
    #scatmp.pacify(.75, 4, bias = 2)
    mid = time.time()
    avgtrs = scatmp.edgestat()
    #print(avgtrs)
    #scatmp.edgsep(avgtrs[0])
    scatmp.edgecluster(avgtrs[0])
    #scatmp.dbscancluster(sepdst = -1, minsmp = 4, scaling = True)
    #scatmp.kmeanscluster(3)
    #scatmp.heirarchicalcluster(nmclst = 4, mode = 'single')
    end = time.time()
    print(f'Forming the lattice took {mid - start} seconds')
    print(f'Clustering took {end - mid} seconds')
    #print(scatmp.centerofdensity(.0016697))
    #scatmp.masplt()
    #scatmp.cluster()
    #scatmp.disply()
    #scatmp.edgecluster(avgtrs[0], hrdsft = True)
    #scatmp.disply()
    #scatmp.nnearestneighbordistance(sphere = 2)
    #scatmp.disply()
# =============================================================================
#     scatmp.disply()
#     scatmp.dbscancluster(sepdst = .25, minsmp = 4, scaling = True)
#     scatmp.disply()
#     scatmp.dbscancluster(sepdst = .5, minsmp = 4, scaling = True)
#     scatmp.disply()
#     scatmp.dbscancluster(sepdst = .75, minsmp = 4, scaling = True)
#     scatmp.disply()
#     scatmp.kmeanscluster(4, scaling = True)
#     scatmp.disply()
#     scatmp.kmeanscluster(6, scaling = True)
#     scatmp.disply()
# =============================================================================
    #numpy.savetxt(f'{fname}_labeled.csv', scatmp.labelrep(), delimiter = ',')