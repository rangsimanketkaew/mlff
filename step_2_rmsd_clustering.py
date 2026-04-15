#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 25th, 2020

Cluster geometries from extended xyz (.xyz) and classify into different clusters 
using minimum RMSD with the Kabsch algorithm (https://github.com/charnley/rmsd)

This script performs agglomerative clustering as suggested in
https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

Even though pybel supports several file formats to write the output file for each clusters, 
but it does not support an extended xyz file. So I then will use the step_3_form_clusters.py 
to form the clusters by reading a cluster.dat file which printed by this script.

Prerequisites:
# Please use conda create a new environment and install the following packages.
conda update --all -y
conda install numpy
conda install -c openbabel openbabel (this will also install pybel)
conda install scipy
conda install scikit-learn
conda install matplotlib
conda install pip
conda install -c rangsiman rmsd

# Only rmsd that must be installed using pip.
pip install rmsd

Normal Usage:
$ python step_2_rmsd_clustering.py YOUR.xyz RMSD_CUTOFF -od dist.dat -oc cluster.dat 
"""

import argparse
import itertools
import multiprocessing
import os
import sys

import numpy as np
import openbabel
import pybel
import rmsd
import scipy.cluster.hierarchy as hcl
from sklearn import manifold
from scipy.spatial.distance import squareform

import matplotlib as mpl
import matplotlib.pyplot as plt


def get_mol_coords(mol):
    """
    Extract all cartesian coordinates from molecule.
    """
    q_all = []
    for atom in mol:
        q_all.append(atom.coords)

    return np.asarray(q_all)


def get_mol_info(mol):
    """
    Find information of molecule.
    """
    # table to convert atomic number to symbols
    etab = openbabel.OBElementTable()

    q_atoms = []
    q_all = []
    for atom in mol:
        q_atoms.append(etab.GetSymbol(atom.atomicnum))
        q_all.append(atom.coords)

    return np.asarray(q_atoms), np.asarray(q_all)


def compute_distmat_line(idx1, 
                         q_info, 
                         traj_file, 
                         noh 
                         ):
    """
    Compute all distances in parallel (on request)
    """
    # unpack q_info tuple
    q_atoms, q_all = q_info

    # initialize distance matrix
    distmat = []

    for idx2, mol2 in enumerate(pybel.readfile(os.path.splitext(traj_file)[1][1:], traj_file)):
        # skip if it's not an element from the superior diagonal matrix
        if idx1 >= idx2:
            continue

            # arrays for second molecule
        p_atoms, p_all = get_mol_info(mol2)
        
        if noh:
            not_hydrogens = np.where(p_atoms != 'H')
            P = p_all[not_hydrogens]
            Q = q_all[not_hydrogens]
            Pa = p_atoms[not_hydrogens]
            Qa = q_atoms[not_hydrogens]
            pcenter = rmsd.centroid(P)
            qcenter = rmsd.centroid(Q)
        else:
            P = p_all
            Q = q_all
            Pa = p_atoms
            Qa = q_atoms
            pcenter = rmsd.centroid(P)
            qcenter = rmsd.centroid(Q)

        # center the coordinates at the origin
        P -= pcenter
        Q -= qcenter
        
        # get the RMSD and store it
        distmat.append(rmsd.kabsch_rmsd(P, Q))

    return distmat


def build_distance_matrix(trajfile, 
                          noh, 
                          nprocs
                          ):
    """
    Compute all distances in parallel (on request)
    """
    # create iterator containing information to compute a line of the distance matrix
    input_iterator = zip(itertools.count(),
                         map(lambda x: 
                             get_mol_info(x), 
                             pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)
                             ), 
                         itertools.repeat(trajfile), 
                         itertools.repeat(noh)
                         )

    # create the pool with nprocs processes to compute the distance matrix in parallel
    p = multiprocessing.Pool(processes=nprocs)

    # build the distance matrix in parallel
    ldistmat = p.starmap(compute_distmat_line, input_iterator)

    return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


def save_clusters_config(trajfile, 
                         clusters, 
                         distmat, 
                         noh, 
                         outbasename, 
                         outfmt, 
                         ):
    """
    Save all info of classified clusters
    """
    # create a distance square matrix
    sqdistmat = squareform(distmat)

    for cnum in range(1, max(clusters) + 1):
        # create object to output the configurations
        outfile = pybel.Outputfile(
            outfmt, outbasename + "_" + str(cnum) + "." + outfmt)

        # creates mask with True only for the members of cluster number cnum
        mask = np.array([1 if i == cnum else 0 for i in clusters], dtype=bool)

        # gets the member with smallest sum of distances from the submatrix
        idx = np.argmin(sum(sqdistmat[:, mask][mask, :]))

        # get list with the members of this cluster only and store medoid
        sublist = [num for (num, cluster) in enumerate(
            clusters) if cluster == cnum]
        medoid = sublist[idx]

        # get the medoid coordinates
        for idx, mol in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):
            if idx != medoid:
                continue

            # medoid coordinates
            tnatoms = len(mol.atoms)
            q_atoms, q_all = get_mol_info(mol)
                
            if noh:
                not_hydrogens = np.where(q_atoms != 'H')
                Q = np.copy(q_all[not_hydrogens])
                qcenter = rmsd.centroid(Q)
                Qa = np.copy(q_atoms[not_hydrogens])
            else:
                Q = np.copy(q_all)
                qcenter = rmsd.centroid(Q)
                Qa = np.copy(q_atoms)

            # center the coordinates at the origin
            Q -= qcenter

            # write medoid configuration to file (molstring is a xyz string used to generate pybel mol)
            molstring = str(tnatoms) + "\n" + mol.title.rstrip() + "\n"
            for i, coords in enumerate(q_all - qcenter):
                molstring += q_atoms[i] + "\t" + str(coords[0]) + "\t" + str(
                    coords[1]) + "\t" + str(coords[2]) + "\n"
            rmol = pybel.readstring("xyz", molstring)
            outfile.write(rmol)

            break

        # rotate all the cluster members into the medoid and print them to the .xyz file
        for idx, mol in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):
            if not mask[idx] or idx == medoid:
                continue

            # config coordinates
            p_atoms, p_all = get_mol_info(mol)

            if noh:
                not_hydrogens = np.where(p_atoms != 'H')
                P = np.copy(p_all[not_hydrogens])
                pcenter = rmsd.centroid(P)
                Pa = np.copy(p_atoms[not_hydrogens])
            else:
                P = np.copy(p_all)
                pcenter = rmsd.centroid(P)
                Pa = np.copy(p_atoms)

            # center the coordinates at the origin
            P -= pcenter
            p_all -= pcenter

            # generate rotation matrix
            U = rmsd.kabsch(P, Q)

            # rotate whole configuration (considering hydrogens even with noh)
            p_all = np.dot(p_all, U)

            # write rotated configuration to file (molstring is a xyz string used to generate pybel mol)
            molstring = str(tnatoms) + "\n" + mol.title.rstrip() + "\n"
            for i, coords in enumerate(p_all):
                molstring += p_atoms[i] + "\t" + str(coords[0]) + "\t" + str(
                    coords[1]) + "\t" + str(coords[2]) + "\n"
            rmol = pybel.readstring("xyz", molstring)
            outfile.write(rmol)

        outfile.close()


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a clustering analysis on a trajectory based on the minimal RMSD obtained with a Kabsch superposition.')
    parser.add_argument('trajectory_file',
                        help='path to the trajectory containing the conformations to be classified')
    parser.add_argument('min_rmsd', type=float,
                        help='value of RMSD used to classify structures as similar')
    parser.add_argument('-np', '--nprocesses', metavar='NPROCS', type=check_positive, default=2,
                        help='defines the number of processes used to compute the distance matrix and multidimensional representation '
                             '(default = 2)')
    parser.add_argument('-n', '--no-hydrogen', action='store_true',
                        help='ignore hydrogens when doing the Kabsch superposition and calculating the RMSD')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='enable the multidimensional scaling and dendrogram plot saving the figures in pdf format '
                             '(filenames use the same basename of the -oc option)')
    parser.add_argument('-m', '--method', metavar='METHOD', default='average',
                        help="method used for clustering (see valid methods at "
                             "https://docs.scipy.org/doc/scipy-1.4.1/reference/generated/scipy.cluster.hierarchy.linkage.html) "
                             "(default: average)")
    parser.add_argument('-cc', '--clusters-configurations', metavar='EXTENSION',
                        help='save superposed configurations for each cluster in EXTENSION format '
                             '(basename based on -oc option)')
    parser.add_argument('-oc', '--outputclusters', default='clusters.dat', metavar='FILE',
                        help='file to store the clusters (default: clusters.dat)')

    io_group = parser.add_mutually_exclusive_group()
    io_group.add_argument('-i', '--input', type=argparse.FileType('rb'), metavar='FILE',
                          help='file containing input distance matrix in condensed form')
    io_group.add_argument('-od', '--outputdistmat', metavar='FILE',
                          help='file to store distance matrix in condensed form (default: distmat.dat)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # check input consistency manually
    if args.method not in ["single", 
                           "complete", 
                           "average", 
                           "weighted", 
                           "centroid", 
                           "median", 
                           "ward"
                           ]:
        exit("Error: The method you selected with -m (%s) is not valid." 
             % args.method)

    if args.clusters_configurations:
        # pybel does not support extended xyz, so I then use the step_3_split_to_clusters.py to
        # write the extxyz files for each cluster
        if args.clusters_configurations not in \
                ["acr", "adf", "adfout", "alc", "arc", "bgf", "box", "bs", "c3d1",
                 "c3d2", "cac", "caccrt", "cache", "cacint", "can", "car", "ccc", "cdx",
                 "cdxml", "cht", "cif", "ck", "cml", "cmlr", "com", "copy", "crk2d",
                 "crk3d", "csr", "cssr", "ct", "cub", "cube", "dmol", "dx", "ent", "fa",
                 "fasta", "fch", "fchk", "fck", "feat", "fh", "fix", "fpt", "fract",
                 "fs", "fsa", "g03", "g92", "g94", "g98", "gal", "gam", "gamin",
                 "gamout", "gau", "gjc", "gjf", "gpr", "gr96", "gukin", "gukout",
                 "gzmat", "hin", "inchi", "inp", "ins", "jin", "jout", "mcdl", "mcif",
                 "mdl", "ml2", "mmcif", "mmd", "mmod", "mol", "mol2", "molden",
                 "molreport", "moo", "mop", "mopcrt", "mopin", "mopout", "mpc", "mpd",
                 "mpqc", "mpqcin", "msi", "msms", "nw", "nwo", "outmol", "pc", "pcm",
                 "pdb", "png", "pov", "pqr", "pqs", "prep", "qcin", "qcout", "report",
                 "res", "rsmi", "rxn", "sd", "sdf", "smi", "smiles", "sy2", "t41", "tdd",
                 "test", "therm", "tmol", "txt", "txyz", "unixyz", "vmol", "xed", "xml",
                 "xyz", "yob", "zin"]:
            exit("Error: The format you selected to save the clustered superposed configurations (%s) is not valid."
                % args.clusters_configurations)

    # name an cluster output file
    if len(os.path.splitext(args.outputclusters)[1]) == 0:
        args.outputclusters += ".dat"

    if not args.input:
        if not args.outputdistmat:
            args.outputdistmat = "distmat.dat"

        if os.path.exists(args.outputdistmat):
            exit("Error: File %s already exists, please specify a new filename with -od command option"
                 % args.outputdistmat)
        else:
            # os.system('rm %s' % args.outputdistmat)
            args.outputdistmat = open(args.outputdistmat, 'wb')

    if os.path.exists(args.outputclusters):
        exit("Error: File %s already exists, please specify a new filename with -oc command option" 
             % args.outputclusters)
    else:
        # os.system('rm %s' % args.outputclusters)
        args.outputclusters = open(args.outputclusters, 'wb')

    print("=============== START ===============")

    # check if distance matrix will be read from input or calculated
    if args.input:
        print("Reading condensed distance matrix from %s" 
              % args.input.name)
        distmat = np.loadtxt(args.input)
    # build a distance matrix already in the condensed form
    else:
        print("Calculating distance matrix using %d threads" 
              % args.nprocesses)
        distmat = build_distance_matrix(args.trajectory_file, 
                                        args.no_hydrogen,
                                        args.nprocesses
                                        )
        print("Saving condensed distance matrix to %s" 
              % args.outputdistmat.name)
        np.savetxt(args.outputdistmat, distmat, fmt='%.18f')
        # args.outputdistmat.close()

    # linkage
    print("Starting clustering using '%s' method to join the clusters" 
          % args.method)
    Z = hcl.linkage(distmat, args.method)

    # build the clusters and print them to file
    clusters = hcl.fcluster(Z, args.min_rmsd, criterion='distance')
    print("Saving clustering classification to %s" 
          % args.outputclusters.name)
    np.savetxt(args.outputclusters, clusters, fmt='%d')
    args.outputclusters.close()

    # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
    if args.clusters_configurations:
        print("Writing superposed configurations per cluster to files %s" % (
            os.path.splitext(args.outputclusters.name)[
                0] + "_confs" + "_*" + "." + args.clusters_configurations))
        save_clusters_config(args.trajectory_file, 
                             clusters, 
                             distmat, 
                             args.no_hydrogen, 
                             os.path.splitext(args.outputclusters.name)[0] + "_confs", 
                             args.clusters_configurations,
                             )

    if args.plot:
        # plot evolution with cluster in trajectory
        plt.figure(figsize=(25, 10))
        plt.plot(range(1, len(clusters) + 1), clusters, "o-", markersize=4)
        plt.title('Cluster Classification Evolution')
        plt.xlabel('Geometry Index')
        plt.ylabel('Cluster classification')
        # plt.xticks(list(range(1, len(clusters) + 1, 5)))
        # plt.show()
        plt.savefig(os.path.splitext(args.outputclusters.name)[0] + "_evo.png", bbox_inches='tight')

        # plot the dendrogram
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Geometry Index')
        plt.ylabel('RMSD (Ångström)')
        hcl.dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.axhline(args.min_rmsd, linestyle='--')
        # plt.show()
        plt.savefig(os.path.splitext(args.outputclusters.name)[
                    0] + "_dendrogram.png", bbox_inches='tight')

        # finds the 2D representation of the distance matrix (multidimensional scaling) and plot it
        plt.figure()
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=666, n_init=3, max_iter=200,
                           eps=1e-3, n_jobs=args.nprocesses)
        coords = mds.fit_transform(squareform(distmat))
        # plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                        # left=False, right=False, labelbottom=False, labelleft=False)
        plt.scatter(coords[:, 0], coords[:, 1], marker='o', 
                    c=clusters, cmap=plt.cm.get_cmap("nipy_spectral"))
        plt.title('Multidimensionality Scaling (dissimilarity)')
        plt.xlabel('coordinate 1')
        plt.ylabel('coordinate 2')
        # plt.show()
        plt.savefig(os.path.splitext(args.outputclusters.name)
                    [0] + "_mds.png", bbox_inches='tight')

    # print the cluster sizes
    print("\nA total of %d cluster(s) was(were) found." % max(clusters))
    print("A total of %d structures were read from the trajectory." % len(clusters))
    print("The cluster sizes are:\n")
    print("Cluster\tSize")
    print("=======\t====")
    
    labels, sizes = np.unique(clusters, return_counts=True)
    for label, size in zip(labels, sizes):
        print("%d\t%d" % (label, size))

    # save summary
    with open("summary_" + args.outputclusters.name, "w") as f:
        f.write("Clusterized %d structures from file %s with a minimum RMSD of %f\n" 
                % (len(clusters), args.trajectory_file, args.min_rmsd))
        f.write("Method: %s\nIgnoring hydrogens?: %s\n" 
                % (args.method, args.no_hydrogen))

        if args.input:
            f.write("\nDistance matrix was read from: %s\n" 
                    % args.input.name)
        else:
            f.write("\nDistance matrix was written in: %s\n" %
                    args.outputdistmat.name)
        f.write("The classification of each configuration was written in: %s\n" 
                % args.outputclusters.name)
        if args.clusters_configurations:
            f.write("\nThe superposed structures for each cluster were saved at: %s\n" 
                    % (os.path.splitext(args.outputclusters.name)[0] + 
                       "_confs" + "_*" + "." + args.clusters_configurations))
        if args.plot:
            f.write("\nPlotting the dendrogram to: %s\n" 
                    % (os.path.splitext(args.outputclusters.name)[0] + "_dendrogram.png"))
            f.write("Plotting the multidimensional scaling to 2D to: %s\n" 
                    % (os.path.splitext(args.outputclusters.name)[0] + ".png"))
            f.write("Plotting the evolution of the classification with the trajectory to: %s\n" 
                    % (os.path.splitext(args.outputclusters.name)[0] + "_evo.png"))

        f.write("\nThe following %d clusters were found:\n" % max(clusters))
        f.write("Cluster\tSize\n")
        for label, size in zip(labels, sizes):
            f.write("%d\t%d\n" % (label, size))

    print("=============== DONE ===============")
