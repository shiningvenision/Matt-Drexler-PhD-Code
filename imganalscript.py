# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:04:23 2019

@author: shini
"""

import fftcompression
import rsicompression
import sandwhichsmooth
#import levelgraph
import pointfitting
import lonelattice
import latticeanalysis
import numpy
import time
import imageio
import matplotlib.pyplot as mpl

flname = input('Please enter the name of your file: ')
[fname, ftype] = flname.split('.')
if ftype == 'csv':
    imgdat = numpy.genfromtxt(flname, delimiter = ',')
else:
    imgdat = imageio.imread(flname)
start = time.time()
# imprad = fftcompression.findcompressionradius(imgdat) # A
imprad = int(numpy.ceil(fftcompression.findcharacteristicperiod(imgdat, noisereductionfactor = 2, vectortolerance = 1, mode = 'average') / 2)) # B
print(imprad)
smthed = sandwhichsmooth.cycleinterp(imgdat, imprad)
iddimg = rsicompression.imageevaluation(smthed, imprad) # A
atmpos = rsicompression.pointdetection(iddimg, imprad) # A
# atmpos = rsicompression.spotatoms(smthed, imprad) # B
lonlat = lonelattice.lonelattice()
lonlat.djkbuild(atmpos)
preref = lonlat.disply(imgdat.shape)
lonlat.refinepoints(smthed)
lonlat.redjkbuild()
posref = lonlat.disply(imgdat.shape)
lonlat.heightest(imgdat)
#numpy.savetxt(f'{fname}_heighttestdata.csv', lonlat.heightstat(), delimiter = ',')
hgtfit = pointfitting.emforsize(lonlat.heightstat(), tstnum = 20)
lonlat.catheight(hgtfit)
edgsts = lonlat.edgrep()
proces = latticeanalysis.strainanalysis(edgsts, orientation = True, category = True, position = True)
end = time.time()
print(end - start)
numpy.savetxt(f'{fname}_r_edge_list.csv', edgsts, delimiter = ',')
numpy.savetxt(f'{fname}_r_length_height.csv', proces, delimiter = ',')
mpl.figure()
mpl.imshow(imgdat)
mpl.figure()
mpl.imshow(smthed)
mpl.figure() # A
mpl.imshow(rsicompression.visualizecategoryimage(iddimg)) # A
mpl.figure()
mpl.imshow(rsicompression.markpositions(imgdat, atmpos))
mpl.figure()
mpl.imshow(lonlat.disply(imgdat.shape))
mpl.figure()
mpl.imshow(preref)
mpl.figure()
mpl.imshow(posref)