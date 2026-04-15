#!/usr/bin/env python3

"""
Rangsiman Ketkaew
May 3rd, 2020

Randomly sample N geometries (conformers) from every classified clusters

Normal usage:
1. Edit the user-defined parameters below
2. Execute the script at the directory containing xyz files. If your input cluster files are
    cluster_1.xyz, cluster_2.xyz, ..., the input file for this script should be 'cluster_*.xyz'
    $ python step_4_sample_geom_randomly.py 'cluster_*.xyz' 10
"""

import os
import sys
import glob
import numpy as np

##############################
outputs = 'cluster_sampled'
ext = '.xyz'
##############################

try:
    inputs = str(sys.argv[1])
except IndexError:
    exit("Error: Please specify a representative name of all cluster files, such as cluster_*.xyz")

try:
    sample_size = int(sys.argv[2])
except IndexError:
    exit("Error: Please specify the size for sampling geometries")

try:
    files = glob.glob(inputs)
    a = files[0]
except IndexError:
    exit("Error: Can't open the extxyz files with the argument you entered : {0}".format(inputs))

prefix, suffix = inputs.split('*')

print("Equally sample multiple geometries from the classified cluster (extended xyz)")

# loop over input files (*.xyz)
for n in range(len(files)):
    # nfile = int(f.replace(prefix, "").replace(suffix, ""))
    # Read file and split 1D array of whole lines into 2D array containing separate geometries
    f = prefix + str(n + 1) + suffix
    strcts = np.loadtxt(f, delimiter='\n', dtype=str)
    nlines = strcts.size
    natoms = strcts[0].astype(np.int)
    mols = np.array(np.split(strcts, nlines / (natoms + 2)))
    
    print("Splitting file no. " + str(n + 1) + " : " + f)
    print("Total number of geometries : " + str(mols.shape[0]))

    nsplit = 1
    split = True
    while split:
        # check if user-defined sample size is over number of geometries
        samsize = sample_size if sample_size <= mols.shape[0] else mols.shape[0]
        
        # get index of the randomly sampled geometries
        idx = np.random.choice(mols.shape[0], samsize, replace=False)
        new_mols = np.array(mols[idx])
        mols = np.array(np.delete(mols, idx, axis=0))

        # flatten 2D array to 1D in order not to write the output files using two for-loops
        new_mols = new_mols.flatten()
        
        # name the output files (*.xyz)
        outname = outputs + '_cluster-' + str(n + 1) + '_split-' + str(nsplit) + '_size-' + str(samsize) + ext
        out = open(outname, 'w+')
        for i in range(len(new_mols)):
            out.write(new_mols[i]+'\n')
        out.close()

        print("Successfully written \"{0}\" with sample size of {1}".format(outname, samsize))
        
        nsplit += 1
        split = True if mols.shape[0] > 0 else False

    print("----------------- DONE -----------------")
        
    ## combine all output files into one file
    # output = glob.glob(outputs + '_cluster-' + str(n + 1) + '_split-' + '*' + ext)
    # oneoutname = outputs + '_all_geoms_cluster-' + str(n + 1) + ext
    # with open(oneoutname, 'w+') as oneout:
    #     for o in output:
    #         with open(o, 'r') as inp:
    #             for line in inp:
    #                 oneout.write(line)

    # print("All output files were also combined into \"{0}\"".format(oneoutname))
