# Matt-Drexler-PhD-Code

This repository is used to store the code I wrote and used at various different times during my PhD. There are three main categories that this code falls into:
 1. HAADF STEM image analysis
 2. Clustering algorithms
 3. Miscellaneous

A breif description of the files according to their category are below:
 HAADF STEM image analysis
  1. fftcompression: Used to find the approximate size of atoms within the image in pixels using the Fourier transform. Used before rsicompression and sandwhichsmooth.
  2. imganalscript: Contains the general workflow of methods needed to find the atoms of an image and get useful information about their relative positions. Shows the appropriate order of the files.
  3. latticeanalysis: Takes a network of points and calculates the distance and orientation of the connections. Used after lonelattice.
  4. lonelattice: Connects a set of identified features to its nearest neighbors in a network by process of elimination. Used after rsicompression, in conjunction with pointfitting, and before latticeanalysis.
  5. pointfitting: Refines the positions of identified features that have been connected in a network. Used in conjunction with lonelattice.
  6. rsicompression: Identifies atoms in an image by comparing subsets of the image to theoretical standards for types of features. Used after fftcomparison and before lonelattice.
  7. sandwhichsmooth: Reduces image noise by iteratively averaging interpolated images. Used optionally after fftcompression and before rsicompression.

 Clustering algorithms
  1. gravitycluster: Clusters points by connecting each data point to its nearest neighbors to form a graph then making cuts to that graph based on connection length.
  2. visualclustering: Creates a grayscale visual representation of the data, then forms clusters based on image derivatives.

 Miscellaneous
  1. expectmaxlocalize: Contains a method that calculates the lowest average difference between graphene bonding sites and a metallic lattice for a given range of angles.
  2. semanalysis: Contains code for simulating the ORR reaction, normalizing XPS spectra, and simulating HAADF STEM images for different materials.
