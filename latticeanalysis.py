# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:32:55 2019

@author: shini
"""

import numpy
import lonelattice
import imageio
import levelgraph
import matplotlib.pyplot as mpl
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.optimize as sciopt

def findatomheight(lattice):
    vstlst = []
    hgtdel = []
    for node in lattice.nodlst:
        vstlst.append(node)
        for neibr in node.neibrs:
            if not neibr in vstlst:
                hgtdel.append(numpy.abs(neibr[0].height - node.height))
    upbond = numpy.max(hgtdel)
    lwbond = numpy.min(hgtdel)
    nestmt = numpy.zeros((len(hgtdel)))
    aestmt = lwbond
    ascr = len(hgtdel) * (upbond - lwbond)**2
    prescr = len(hgtdel) * (upbond - lwbond)**2
    for aval in numpy.linspace(lwbond, upbond, num = 50):
        for i in range(len(hgtdel)):
            nestmt[i] = numpy.round(hgtdel[i] / aval)
        curscr = 0
        for i in range(len(hgtdel)):
            curscr = curscr + (hgtdel[i] - nestmt[i] * aval)**2
        if curscr < ascr:
            aestmt = aval
            ascr = curscr
    while (prescr - ascr) / prescr > .0001:
        for i in range(len(hgtdel)):
            nestmt[i] = numpy.round(hgtdel[i] / aestmt)
        gradea = 0
        for i in range(len(hgtdel)):
            gradea = gradea + 2 * nestmt[i] * (hgtdel[i] - nestmt[i] * aestmt)
        aestmt = aestmt - .0001 * gradea
        prescr = ascr
        for i in range(len(hgtdel)):
            ascr = ascr + (hgtdel[i] - nestmt[i] * aestmt)
    return aestmt

def clusterheight(lattice, sepdst = -1, minsmp = -1):
    hgtlst = []
    vstlst = []
    for node in lattice.nodlst:
        hgtlst.append(node.height)
        vstlst.append(node)
        for neb in node.neibrs:
            if not neb[0] in vstlst:
                hgtlst.append(numpy.abs(neb[0].height - node.height))
    srtarr = numpy.sort(hgtlst)
    print(srtarr)
    if sepdst == -1:
        bestk = 1
        beste = len(srtarr)
        for k in range(1, int(numpy.ceil(len(srtarr) / 4)) + 1):
            kdslst = []
            for i in range(len(srtarr)):
                if i < k:
                    kdslst.append(srtarr[i + k] - srtarr[i])
                elif i >= len(srtarr) - k:
                    kdslst.append(srtarr[i] - srtarr[i - k])
                else:
                    if srtarr[i + k] - srtarr[i] > srtarr[i] - srtarr[i - k]:
                        kdslst.append(srtarr[i] - srtarr[i - k])
                    else:
                        kdslst.append(srtarr[i + k] - srtarr[i])
            kdslst = numpy.sort(kdslst)
            hghvec = -1
            espind = 0
            for i in range(1, len(kdslst) - 1):
                veca = [-i, kdslst[0] - kdslst[i]]
                vecb = [len(kdslst) - 1 - i, kdslst[len(kdslst) - 1] - kdslst[i]]
                vang = (veca[0] * vecb[0] + veca[1] * vecb[1]) / ((veca[0]**2 + veca[1]**2)**.5 * (vecb[0]**2 + vecb[1]**2)**.5)
                if vang > hghvec:
                    hghvec = vang
                    espind = i
            if beste > kdslst[espind]:
                beste = kdslst[espind]
                bestk = k
        dbclst = DBSCAN(eps = beste, min_samples = bestk + 1)
    elif minsmp == -1:
        dbclst = DBSCAN(eps = sepdst, min_samples = 3)
    else:
        dbclst = DBSCAN(eps = sepdst, min_samples = minsmp)
    dbclst.fit(numpy.array(srtarr).reshape(-1, 1))
    #print(dbclst.core_sample_indices_)
    #print(dbclst.labels_)
    #cores = dbclst.components_
    clstrs = numpy.zeros((int(numpy.max(dbclst.labels_)) + 1, 2))
    print(clstrs.shape)
    print(len(srtarr))
    for i in range(len(dbclst.labels_)):
        clstrs[dbclst.labels_[i]][0] = clstrs[dbclst.labels_[i]][0] + srtarr[i]
        clstrs[dbclst.labels_[i]][1] = clstrs[dbclst.labels_[i]][1] + 1
    centrs = []
    for clstr in clstrs:
        centrs.append(clstr[0] / clstr[1])
    return centrs

def updateheights(lattice, aguess):
    newhgt = []
    for i in range(len(lattice.nodlst)):
        newhgt.append(lattice.nodlst[i].height)
        nmain = numpy.round(newhgt[i] / aguess)
        nlist = []
        for j in range(len(lattice.nodlst[i].neibrs)):
            nlist.append(numpy.round((newhgt[i] - lattice.nodlst[i].neibrs[j][0].height) / aguess))
        nlist = []
        prescr = aguess**2 * (len(lattice.nodlst[i].neibrs) + 1)
        curscr = aguess**2 * len(lattice.nodlst[i].neibrs)
        while (prescr - curscr) / prescr > .0001:
            hgtdel = 2 * (newhgt[i] - nmain * aguess)
            for j in range(len(nlist)):
                hgtdel = hgtdel + 2 * (newhgt[i] - lattice.nodlst[i].neibrs[j][0].height - nlist[j] * aguess)
            newhgt[i] = newhgt[i] - .0001 * hgtdel
            nmain = numpy.round(newhgt[i] / aguess)
            nlist = []
            for j in range(len(lattice.nodlst[i].neibrs)):
                nlist.append(numpy.round((newhgt[i] - lattice.nodlst[i].neibrs[j][0].height) / aguess))
            prescr = curscr
            curscr = (newhgt[i] - nmain * aguess)**2
            for j in range(len(lattice.nodlst[i].neibrs)):
                curscr = curscr + (newhgt[i] - lattice.nodlst[i].neibrs[j][0].height - nlist[j] * aguess)**2
    for i in range(len(lattice.nodlst)):
        lattice.nodlst[i].base = lattice.nodlst[i].base + lattice.nodlst[i].height - newhgt[i]
        lattice.nodlst[i].height = newhgt[i]

def refinethickness(lattice):
    prea = findatomheight(lattice)
    updateheights(lattice, prea)
    cura = findatomheight(lattice)
    print([prea, cura])
    while numpy.abs(prea - cura) / prea > .0001:
        updateheights(lattice, cura)
        prea = cura
        cura = findatomheight(lattice)
        print(cura)
    return cura

def strainanalysis(edges, orientation = False, height = False, category = False, position = False, kthneb = 5):
    anlarr = []
    for edg in edges:
        lentry = [edg[8]]
        if orientation:
            if edg[8] == 0:
                cosang = 1
            else:
                cosang = (edg[4] - edg[0]) / edg[8]
            if edg[5] - edg[1] < 0:
                cosang = cosang * -1
            lentry.append(cosang)
        if height:
            if edg[2] < edg[6]:
                lentry.append(edg[2])
                lentry.append(edg[6] - edg[2])
            else:
                lentry.append(edg[6])
                lentry.append(edg[2] - edg[6])
        if category:
            if edg[2] < edg[6]:
                lentry.append(edg[3])
                lentry.append(edg[7] - edg[3])
            else:
                lentry.append(edg[7])
                lentry.append(edg[3] - edg[7])
        if position:
            lentry.append((edg[0] + edg[4]) / 2)
            lentry.append((edg[1] + edg[5]) / 2)
        anlarr.append(lentry)
    #stdscl = StandardScaler().fit_transform(anlarr)
    #knndst = []
    #for i in range(len(anlarr)):
    #    mindst = []
    #    for j in range(len(anlarr)):
    #        if not i == j:
    #            dst = (numpy.sum((stdscl[j] - stdscl[i])**2))**.5
    #            if dst > 0:
    #                srchng = True
    #                k = 0
    #                while k < len(mindst) and srchng:
    #                    if dst < mindst[k]:
    #                        k = k + 1
    #                    else:
    #                        srchng = False
    #                mindst.insert(k, dst)
    #                if len(mindst) > kthneb:
    #                    mindst.pop(0)
    #    knndst.append(mindst[0])
    #dstgss = numpy.sort(knndst)
    #smlang = -2
    #vback = [0, 0]
    #vforw = [0, 0]
    #angarr = []
    #okyeps = dstgss[len(dstgss) - 1]
    #for i in range(1, len(dstgss) - 1):
    #    vback = [dstgss[0] - dstgss[i], -1 * i * (numpy.max(dstgss) - numpy.min(dstgss)) / len(dstgss)]
    #    vforw = [dstgss[len(dstgss) - 1] - dstgss[i], (len(dstgss) - 1 - i) * (numpy.max(dstgss) - numpy.min(dstgss)) / len(dstgss)]
    #    curang = (vback[0] * vforw[0] + vback[1] * vforw[1]) / ((vback[0]**2 + vback[1]**2)**.5 * (vforw[0]**2 + vforw[1]**2)**.5)
    #    angarr.append(curang)
    #    if curang > smlang:
    #        smlang = curang
    #        okyeps = dstgss[i]
    #mpl.figure()
    #mpl.scatter(range(len(dstgss)), dstgss)
    #mpl.plot(range(len(dstgss)), okyeps * numpy.ones((len(dstgss))))
    #mpl.plot(range(len(angarr)), angarr)
    #print(f'Running OPTICS with {okyeps} as the max epsilon')
    #opclst = AgglomerativeClustering(n_clusters = kthneb)
    #opclst.fit(stdscl)
    #for i in range(len(anlarr)):
    #    anlarr[i].append(opclst.labels_[i])
    # This is the first custom attempt to do custom clustering. Doesn't work 
    # because apparently a sum of parabolas makes a parabola
    #datarr = []
    #maxs = []
    #for i in range(len(anlarr[0])):
    #    datarr.append([])
    #    maxs.append([])
    #for dat in anlarr:
    #    for i in range(len(datarr)):
    #        datarr[i].append(dat[i])
    #for i in range(len(datarr)):
    #    errs = []
    #    cutgss = numpy.linspace(numpy.min(datarr[i]), numpy.max(datarr[i]), num = 100)
    #    for k in cutgss:
    #        err = 0
    #        for p in datarr[i]:
    #            err = err + (p - k)**2
    #        errs.append(err)
    #    for k in range(1, len(errs) - 1):
    #        if errs[k] > errs[k - 1] and errs[k] >= errs[k + 1]:
    #            maxs[i].append(cutgss[k])
    #    maxs[i].append(numpy.max(datarr[i]))
    #    maxs[i] = numpy.sort(maxs[i])
    #print(maxs)
    #for ent in anlarr:
    #    binnms = []
    #    for i in range(len(ent)):
    #        j = 0
    #        srchng = True
    #        while j < len(maxs[i]) and srchng:
    #            if ent[i] <= maxs[i][j]:
    #                binnms.append(j)
    #                srchng = False
    #            else:
    #                j = j + 1
    #    finbin = binnms[0]
    #    for i in range(1, len(binnms)):
    #        finbin = finbin * len(maxs[i]) + binnms[i]
    #    ent.append(finbin)
    return anlarr #dumbclustering(anlarr)

def dumbclustering(datarr, level = 16):
    datrng = []
    datmin = []
    stpsze = []
    datarr = numpy.array(datarr)
    for i in range(datarr.shape[1]):
        datrng.append(numpy.max(datarr[:, i]) - numpy.min(datarr[:, i]))
        datmin.append(numpy.min(datarr[:, i]))
        stpsze.append(datrng[i] / level)
    lvlone = numpy.zeros((datarr.shape[1], level))
    lvltwo = numpy.zeros((datarr.shape[1], level * 2))
    lvlthr = numpy.zeros((datarr.shape[1], level * 4))
    lvlfor = numpy.zeros((datarr.shape[1], level * 8))
    for dat in datarr:
        for i in range(dat.shape[0]):
            if int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) == level:
                lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) - 1] = lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) - 1] + 1
                lvltwo[i][int(numpy.floor((dat[i] - datmin[i]) * 2 / stpsze[i])) - 1] = lvltwo[i][int(numpy.floor((dat[i] - datmin[i]) * 2 / stpsze[i])) - 1] + 1
                lvlthr[i][int(numpy.floor((dat[i] - datmin[i]) * 4 / stpsze[i])) - 1] = lvlthr[i][int(numpy.floor((dat[i] - datmin[i]) * 4 / stpsze[i])) - 1] + 1
                lvlfor[i][int(numpy.floor((dat[i] - datmin[i]) * 8 / stpsze[i])) - 1] = lvlfor[i][int(numpy.floor((dat[i] - datmin[i]) * 8 / stpsze[i])) - 1] + 1
            else:
                lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i]))] = lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i]))] + 1
                lvltwo[i][int(numpy.floor((dat[i] - datmin[i]) * 2 / stpsze[i]))] = lvltwo[i][int(numpy.floor((dat[i] - datmin[i]) * 2 / stpsze[i]))] + 1
                lvlthr[i][int(numpy.floor((dat[i] - datmin[i]) * 4 / stpsze[i]))] = lvlthr[i][int(numpy.floor((dat[i] - datmin[i]) * 4 / stpsze[i]))] + 1
                lvlfor[i][int(numpy.floor((dat[i] - datmin[i]) * 8 / stpsze[i]))] = lvlfor[i][int(numpy.floor((dat[i] - datmin[i]) * 8 / stpsze[i]))] + 1
    minbns = []
    for i in range(lvlone.shape[0]):
        dimmin = []
        for j in range(1, lvlone.shape[1] - 1):
            if lvlone[i][j] <= lvlone[i][j - 1] and lvlone[i][j] < lvlone[i][j + 1]:
                dimmin.append(j)
        print(dimmin)
        for j in range(len(dimmin)):
            if lvltwo[i][2 * dimmin[j]] <= lvltwo[i][2 * dimmin[j] + 1]:
                dimmin[j] = 2 * dimmin[j]
            else:
                dimmin[j] = 2 * dimmin[j] + 1
            if lvlthr[i][2 * dimmin[j]] <= lvlthr[i][2 * dimmin[j] + 1]:
                dimmin[j] = 2 * dimmin[j]
            else:
                dimmin[j] = 2 * dimmin[j] + 1
            if lvlfor[i][2 * dimmin[j]] <= lvlfor[i][2 * dimmin[j] + 1]:
                dimmin[j] = 2 * dimmin[j]
            else:
                dimmin[j] = 2 * dimmin[j] + 1
            print(stpsze[i] / 8)
            dimmin[j] = (dimmin[j] + 1) * stpsze[i] / 8 + datmin[i]
        dimmin.append(datmin[i] + datrng[i])
        minbns.append(dimmin)
    print(minbns)
    anlarr = []
    for dat in datarr:
        anlpnt = []
        binnms = []
        for i in range(dat.shape[0]):
            anlpnt.append(dat[i])
            srchng = True
            numbin = 0
            while numbin < len(minbns[i]) and srchng:
                if dat[i] <= minbns[i][numbin]:
                    binnms.append(numbin)
                    srchng = False
                else:
                    numbin = numbin + 1
        finbin = binnms[0]
        for i in range(1, len(binnms)):
            finbin = finbin * len(minbns[i]) + binnms[i]
        anlpnt.append(finbin)
        anlarr.append(anlpnt)
    return anlarr

def votingcentroid(datarr, minpts = 2, precis = 1000, numvot = 1000):
    #dims = data.shape[1]
    #proshp = []
    #for i in range(dims):
    #    proshp.append(precis)
    #prbmap = numpy.ones(tuple(proshp))
    #stepps = []
    #starts = []
    #for i in range(dims):
    #    stepps.append(numpy.max(data[:, i]) - numpy.min(data[:, i]))
    #    starts.append(numpy.min(data[:, i]))
    #stepps = stepps / precis
    #for dat in data:
    #    coords = []
    #    for stat in dat:
    #        coords.append(int(numpy.floor(stat / precis)))
    #    entry = []
    #    for cord in coords:
    #        entry = prbmap[cord]
    #    entry = entry + 1
    #prbmap = prbmap / numpy.sum(prbmap)
    data = numpy.array(datarr)
    prbarr = 10 * numpy.ones((data.shape[0])) / data.shape[0]
    smples = []
    for i in range(numvot):
        smples.append([])
        for j in range(data.shape[0]):
            if numpy.random.random() < prbarr[j]:
                smples[i].append(j)
        while len(smples[i]) < minpts:
            apoint = numpy.random.randint(0, data.shape[0])
            if not apoint in smples[i]:
                smples[i].append(apoint)
    print('Samples generated')
    weghts = []
    for i in range(len(smples)):
        clstrs = numpy.zeros((len(smples[i])))
        for j in range(data.shape[0]):
            shtdst = numpy.sum((data[j] - data[smples[i][0]])**2)
            clster = 0
            for k in range(1, len(smples[i])):
                if numpy.sum((data[j] - data[smples[i][k]])**2) < shtdst:
                    shtdst = numpy.sum((data[j] - data[smples[i][k]])**2)
                    clster = k
            clstrs[clster] = clstrs[clster] + shtdst
        weghts.append(clstrs)
    print([numpy.max(weghts), numpy.min(weghts)])
    weghts = (weghts - numpy.min(weghts)) / (numpy.max(weghts) - numpy.min(weghts))
    print('Weights calculated')
    pbupdt = numpy.ones((data.shape[0]))
    for i in range(len(smples)):
        for j in range(len(smples[i])):
            pbupdt[smples[i][j]] = pbupdt[smples[i][j]] + 1 / (1 + weghts[i][j])
    pbupdt = pbupdt / numpy.max(pbupdt)
    prbarr = (prbarr + pbupdt) / 2
    print('Probabilities updated')
    retarr = []
    for i in range(prbarr.shape[0]):
        lkpnt = []
        for cord in data[i]:
            lkpnt.append(cord)
        lkpnt.append(prbarr[i])
        retarr.append(lkpnt)
    return retarr

def maxprobcentroid(datarr, level = 16):
    datrng = []
    datmin = []
    stpsze = []
    datarr = numpy.array(datarr)
    for i in range(datarr.shape[1]):
        datrng.append(numpy.max(datarr[:, i]) - numpy.min(datarr[:, i]))
        datmin.append(numpy.min(datarr[:, i]))
        stpsze.append(datrng[i] / level)
    lvlone = numpy.zeros((datarr.shape[1], level))
    for dat in datarr:
        for i in range(dat.shape[0]):
            if int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) == level:
                lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) - 1] = lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i])) - 1] + 1
            else:
                lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i]))] = lvlone[i][int(numpy.floor((dat[i] - datmin[i]) / stpsze[i]))] + 1
    maxbns = []
    for i in range(lvlone.shape[0]):
        dimmax = []
        for j in range(lvlone.shape[1]):
            if j == 0:
                if lvlone[i][j] > lvlone[i][j + 1]:
                    dimmax.append((j + 1) * stpsze[i] + datmin[i])
            elif j == lvlone.shape[1] - 1:
                if lvlone[i][j] >= lvlone[i][j - 1]:
                    dimmax.append((j + 1) * stpsze[i] + datmin[i])
            else:
                if lvlone[i][j] >= lvlone[i][j - 1] and lvlone[i][j] > lvlone[i][j + 1]:
                    dimmax.append((j + 1) * stpsze[i] + datmin[i])
        #dimmax.append(datmin[i] + datrng[i])
        maxbns.append(dimmax)
    #print(minbns)
    cennum = 1
    centck = []
    #cenind = []
    for bn in maxbns:
        cennum = cennum * len(bn)
        centck.append(0)
        #cenind.append(len(bn))
    centrs = []
    covars = []
    for i in range(cennum):
        centr = []
        covar = []
        for bn in range(len(maxbns)):
            centr.append(maxbns[bn][centck[bn]] - stpsze[bn] / 2)
            covar.append(stpsze[bn] / 4)
        centrs.append(centr)
        covars.append(covar)
        tckind = len(maxbns) - 1
        while tckind > -1:
            centck[tckind] = centck[tckind] + 1
            if centck[tckind] >= len(maxbns[tckind]):
                centck[tckind] = 1
                tckind = tckind - 1
            else:
                tckind = -1
    #print(maxbns)
    #print(covars)
    fedarr = []
    bndarr = []
    for i in range(cennum):
        for j in range(len(centrs[i])):
            fedarr.append(centrs[i][j])
            bndarr.append((datmin[j], datmin[j] + datrng[j]))
        for j in range(len(covars[i])):
            for k in range(j + 1):
                if j == k:
                    fudge = 1
                    bndarr.append((10**-9, datrng[j]**2))
                else:
                    fudge = 0
                    bndarr.append((-1 * 10**-10, 10**-10))
                fedarr.append(fudge * covars[i][j] * covars[i][k])
    #print(fedarr)
    #print(bndarr)
    optcls = sciopt.minimize(multivargauss, fedarr, args = (datarr.shape[1], datarr), method = 'TNC', bounds = bndarr)
    return optcls.x

def multivargauss(params, *args):
    dims = args[0]
    pnts = args[1]
    pstep = args[0]
    for i in range(dims):
        pstep = pstep + i + 1
    centrs = numpy.zeros((int(len(params) / pstep), dims))
    covars = numpy.zeros((int(len(params) / pstep), dims, dims))
    p = 0
    for c in range(int(len(params) / pstep)):
        cp = 0
        for i in range(dims):
            centrs[c][i] = params[c * pstep + i]
        geocnt = 0
        for i in range(dims):
            for j in range(i + 1):
                covars[c][i][j] = params[c * pstep + dims + geocnt]
                covars[c][j][i] = params[c * pstep + dims + geocnt]
                geocnt = geocnt + 1
        #if numpy.isnan(numpy.sum(centrs[c]) + numpy.sum(covars[c])):
        #    print([centrs[c], covars[c]])
        #if numpy.linalg.det(covars[c]) == 0:
        #    print(covars[c])
        try:
            invcov = numpy.linalg.inv(covars[c])
            prodon = numpy.matmul((pnts - centrs[c]), invcov)
            prodtw = numpy.sum(prodon * (pnts - centrs[c]), axis = 1)
            dennom = ((2 * numpy.pi)**dims * numpy.linalg.det(covars[c]))**.5
            if numpy.isnan(dennom):
                print(covars[c])
                print(numpy.linalg.det(covars[c]))
            pcp = numpy.exp(-.5 * prodtw) / dennom
            #pcp = numpy.exp(-.5 * numpy.sum(numpy.matmul((pnts - centrs[c]), numpy.linalg.inv(covars[c])) * (pnts - centrs[c]), axis = 1)) / ((2 * numpy.pi)**dims * numpy.linalg.det(covars[c]))**.5
            cp = numpy.sum(pcp)
            p = p + cp
        except:
            print(covars[c])
    #print(p)
    return p

def centroidtransform(data, center, extens):
    nrmdat = data - center
    justce = StandardScaler()
    nrmdat = justce.fit_transform(nrmdat)
    print(justce.scale_)
    trnsmr = PCA(n_components = center.shape[0])
    trnsmr.fit(nrmdat)
    trnsdt = trnsmr.transform(nrmdat)
    mpl.figure()
    mpl.scatter(trnsdt[:, 0], trnsdt[:, 1], marker = '.')
    print(trnsmr.components_)
    print(justce.inverse_transform(trnsmr.components_))
    sigsqr = numpy.zeros(center.shape)
    for pnt in trnsdt:
        sigsqr = sigsqr + pnt**2
    sigsqr = (sigsqr / (trnsdt.shape[0] - 1))**.5
    return sigsqr

def findcentroid(data, multis = False):
    if multis:
        data = numpy.array(data)
        whtarr = []
        lngdst = numpy.sum((numpy.max(data) - numpy.min(data))**2)
        for i in range(data.shape[0]):
            mindst = lngdst
            for j in range(data.shape[0]):
                if not i == j:
                    if mindst > numpy.sum((data[i][0:(data.shape[1] - 1)] - data[j][0:(data.shape[1] - 1)])**2)**.5:
                        mindst = numpy.sum((data[i][0:(data.shape[1] - 1)] - data[j][0:(data.shape[1] - 1)])**2)**.5
            whtarr.append(1 / (.1 + mindst))
        numser = numpy.max(data[:, -1])
        print(numser)
        cntrds = numpy.zeros((int(numser + 1), data.shape[1]))
        for i in range(data.shape[0]):
            #print([data[i][-1], range(0, (data.shape[1] - 1))])
            cntrds[int(data[i][-1])][0:(data.shape[1] - 1)] = cntrds[int(data[i][-1])][0:(data.shape[1] - 1)] + data[i][0:(data.shape[1] - 1)] * whtarr[i]
            cntrds[int(data[i][-1])][(data.shape[1] - 1)] = cntrds[int(data[i][-1])][(data.shape[1] - 1)] + whtarr[i]
        #cntrds = cntrds[:, 0:(cntrds.shape[1] - 1)] / cntrds[:, cntrds.shape[1] - 1]
        centrs = numpy.zeros((cntrds.shape[0], cntrds.shape[1] - 1))
        for j in range(cntrds.shape[0]):
            centrs[j] = cntrds[j][0:(cntrds.shape[1] - 1)] / cntrds[j][cntrds.shape[1] - 1]
        spreds = numpy.zeros((cntrds.shape[0], 2))
        for i in range(data.shape[0]):
            spreds[int(data[i][-1])][0] = spreds[int(data[i][-1])][0] + numpy.sum((data[i][0:(data.shape[1] - 1)] - centrs[int(data[i][-1])])**2)
            spreds[int(data[i][-1])][1] = spreds[int(data[i][-1])][1] + 1
        spreds = (spreds[:, 0] / spreds[:, 1])**.5
    else:
        data = numpy.array(data)
        whtarr = []
        lngdst = numpy.sum((numpy.max(data) - numpy.min(data))**2)
        for i in range(data.shape[0]):
            mindst = lngdst
            for j in range(data.shape[0]):
                if not i == j:
                    if mindst > numpy.sum((data[i] - data[j])**2)**.5:
                        mindst = numpy.sum((data[i] - data[j])**2)**.5
            whtarr.append(1 / (1 + mindst))
        cntrod = numpy.zeros((data.shape[1] + 1))
        for i in range(data.shape[0]):
            cntrod[0:(data.shape[1] - 1)] = cntrod[0:(data.shape[1] - 1)] + data[i][0:(data.shape[1] - 1)] * whtarr[i]
            cntrod[data.shape[1] - 1] = cntrod[data.shape[1] - 1] + whtarr[i]
        centrs = cntrod[0:(cntrod.shape[0] - 1)] / cntrod[cntrod.shape[1] - 1]
        spreds = numpy.zeros((2))
        for i in range(data.shape[0]):
            spreds[0] = spreds[0] + numpy.sum((data[i][0:(data.shape[1] - 1)] - cntrod)**2)
            spreds[1] = spreds[1] + 1
        spreds = (spreds[:, 0] / spreds[:, 1])**.5
    return [centrs, spreds]

def gravmove(points, numitr = 200, evhorz = 10**-5, tmestp = 10**-2):
    newpts = numpy.array(points)
    itrnum = 0
    while itrnum < numitr:
        print(f'{itrnum}')
        delvcs = numpy.zeros(newpts.shape)
        for i in range(newpts.shape[0]):
            for j in range(newpts.shape[0]):
                if not i == j:
                    delvec = newpts[j] - newpts[i]
                    if numpy.sum(delvec**2) >= evhorz:
                        delmag = 1 / numpy.sum(delvec**2)
                        delvec = delmag * delvec / numpy.sum(delvec**2)**.5
                        delvcs[i] = delvcs[i] + delvec
            delvcs[i] = delvcs[i] / numpy.sum(delvcs[i]**2)**.5
        newpts = newpts + delvcs * tmestp
        itrnum = itrnum + 1
    mpl.figure()
    mpl.scatter(points[:, 0], points[:, 1], marker = '.')
    mpl.scatter(newpts[:, 0], newpts[:, 1], marker = '.')

if __name__ == '__main__':
    flname = input('Please enter the name of your file: ')
    [fname, ftype] = flname.split('.')
    if ftype == 'csv':
        imgdat = numpy.genfromtxt(flname, delimiter = ',')
    elif ftype == 'jpg':
        imgdat = imageio.imread(flname)
    #imgrph = levelgraph.intensgraph()
    #imgrph.bfsnncnst(imgdat)
    #imgrph.hubevl()
    #imgrph.hubcmp(3.1, 9.95)
    #pntlst = imgrph.getcoords()
    #crystl = lonelattice.lonelattice()
    #crystl.qntbuild(pntlst)
    #crystl.refinepoints(imgdat)
    #crystl.restore(f'{fname}.hdf5')
    #orghgt = crystl.heightstat(binning = 'dynamic')
    clstdt = strainanalysis(imgdat, orientation = True, position = True)
    #gravmove(imgdat, numitr = 200, tmestp = 10**-2)
    #print(centroidtransform(imgdat, numpy.array([7.5, -.54]), (1, 1, 1, 1)))
    #print(findcentroid(imgdat, multis = True))
    numpy.savetxt(f'{fname}_analyzed.csv', clstdt, delimiter = ',')
    #numpy.savetxt(f'{fname}_strain_custom_2.csv', clstdt, delimiter = ',')
    #print(clstdt)
    #print(clusterheight(crystl))
    #print(refinethickness(crystl))
    #finhgt = crystl.heightstat()
    #print(orghgt)
    #print(finhgt)
    #f, axearr = mpl.subplots(1, 2)
    #axearr[0].imshow(imgdat)
    #axearr[1].imshow(crystl.pntreplct(imgdat.shape))