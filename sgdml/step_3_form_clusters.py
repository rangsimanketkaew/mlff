#!/usr/bin/env python3

"""
Rangsiman Ketkaew
May 3rd, 2020

This script will form different clusters that based on a list of index of 
classified cluster for each geometries.

Note:
A list of classified cluster index can be created using 'step_2_rmsd_clustering.py'.

Normal usage:
$ python THIS_SCRIPT.py YOUR-ExtXYZ.xyz CLUSTER_LIST_file
"""

import sys
import numpy as np


def save_clusters_extxyz(trajfile, clusters):
    all_lines = open(trajfile).read().splitlines()
    natoms = int(all_lines[0])
    onexyz = natoms + 2
    energies, symbols, coords, forces = [], [], [], []
    i = 0
    while i < len(all_lines):
        energies.append(all_lines[i+1])
        start = i+2
        end = start + natoms
        symbol, coord, force = [], [], []
        for j in range(start, end):
            line = all_lines[j].split("\t")
            symbol.append(line[0])
            coord.append(list(map(float, line[1:4])))
            force.append(list(map(float, line[4:7])))
        symbols.append(symbol)
        coords.append(coord)
        forces.append(force)
        i += onexyz

    energies = np.array(energies)
    symbols = np.array(symbols)
    coords = np.array(coords)
    forces = np.array(forces)

    index_clusters = np.array(
        open(clusters).read().splitlines()).astype(np.int)
    num_clusters = np.sort(np.unique(index_clusters))

    # write file over clusters
    num_all_geom = 0
    for c in num_clusters:
        name = clusters.strip('.' + clusters.split('.')
                              [-1]) + '_' + str(c) + '.xyz'
        out = open(name, 'w+')
        # loop over geometries/molecules in a cluster
        count = 0
        for geom, index in enumerate(index_clusters):
            if index == c:
                out.write(str(natoms)+'\n')
                out.write(energies[geom]+'\n')
                # loop over atoms in a molecule
                for x in range(natoms):
                    out.write(symbols[geom][x] + '\t' +
                              str(coords[geom][x][0]) + '\t' +
                              str(coords[geom][x][1]) + '\t' +
                              str(coords[geom][x][2]) + '\t' +
                              str(forces[geom][x][0]) + '\t' +
                              str(forces[geom][x][1]) + '\t' +
                              str(forces[geom][x][2]) + '\t' + '\n'
                              )
                count += 1
                num_all_geom += 1
        print("Written " + name + " --> there are " + str(count) + " geometries")
        out.close()
    print("==> Total number of geometries = " + str(num_all_geom))


if __name__ == "__main__":
    try:
        # A Complete extended xyz file
        inp = sys.argv[1]
    except IndexError:
        exit("Error: Please specify a complete extended xyz file")
    
    try:
        # A file that contains a list of the classified clusters
        cluster = sys.argv[2]
    except IndexError:
        exit("Error: Please specify a file that contains a list of the classified clusters")

    print("==> Start: writing extxyz files for each cluster")
    save_clusters_extxyz(inp, cluster)
    print("==> Stop: All Done!!")
