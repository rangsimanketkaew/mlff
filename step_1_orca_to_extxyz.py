#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 26th, 2020

Extract the cartesian coordinates of all optimized geometries and gradients 
from MP2 ORCA output files, compute the forces from the gradients, 
and save all as an extended xyz file.

Note that the extended xyz output file will be written at the directory where 
this script is executed.

Normal usage:
$ python step_1_orca_to_extxyz.py /path/to/folder/containing/MP2_outputs
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np


def getVector(nline, start, end):
    vector = []
    for line in nline[start + 1:end]:
        dat = line.split()
        vector.append(dat)

    return vector


def getMol(f):
    orca_file = open(f, "r")
    nline = orca_file.readlines()
    orca_file.close()

    all_atom = []
    all_energy = []
    all_coord = []
    all_grad = []

    # <Number of atoms>
    # <Energy>
    # <Atomic symbol> <coord_x> <coord_y> <coord_z> <force_x> <force_y> <force_z>
    # ...

    # Extract the final single point energies
    for i in range(len(nline)):
        if "FINAL SINGLE POINT ENERGY" in nline[i]:
            energy = nline[i].split(' ')[-1].rstrip("\n\r")
            energy = float(energy) * Eh2kcalmol
            all_energy.append(energy)

    for i in range(len(nline)):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in nline[i]:
            start = i + 1
            for j in range(start + 2, len(nline)):
                if "---" in nline[j]:
                    end = j - 1
                    coord = getVector(nline, start, end)
                    all_coord.append(coord)
                    all_atom.append([coord[a][0] for a in range(len(coord))])
                    break

    for i in range(len(nline)):
        if "The final MP2 gradient" in nline[i]:
            start = i
            for j in range(start + 2, len(nline)):
                if "---" in nline[j]:
                    end = j - 1
                    grad = getVector(nline, start, end)
                    all_grad.append(grad)
                    break

    # Remove the first column (atomic number) from the inner most of 3D array
    # and then convert all element to float

    if len(all_coord) == 0:
        return [], [], [], []
    else:
        all_coord = np.asarray(all_coord)[:, :, 1:].astype(float)
        all_grad = np.asarray(all_grad)[:, :, 1:].astype(float)
        # Convert gradient (Eh/bohr) to force (kcal/mol/Angstroms)
        all_force = all_grad * Eh2kcalmol / bohr2Angstroms * -1

        return all_atom, all_energy, all_coord, all_force


if __name__ == '__main__':

    ### User-defined parameters ###
    Eh2kcalmol = float(627.5094740631)
    bohr2Angstroms = float(0.529177249)
    # sGDML_convert_script = 'step_2_extxyz_to_npz.py'
    ###############################

    # if not shutil.which(sGDML_convert_script):
    #     if os.path.exists(os.getcwd() + '/' + sGDML_convert_script):
    #         sGDML_convert_script = 'python3 ' + sGDML_convert_script
    #     else:
    #         sys.stderr.write("Error: \'{0}\' not found in PATH environment variable!\n"
    #                          .format(sGDML_convert_script))
    #         exit()

    parser = argparse.ArgumentParser(
        description='Creates a dataset (.npz) from MP2-ORCA output (.out) files.'
    )
    parser.add_argument(
        'mp2_dir',
        metavar='MP2-output-directory',
        type=str,
        help='path to directory storing MP2 ORCA output files',
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar='name-of-output',
        action='store',
        help='name of extended xyz and numpy zip files',
    )

    args = parser.parse_args()
    mp2dir = args.mp2_dir

    files = glob.glob(str(mp2dir) + '/' + '*.out')

    if not files:
        sys.stderr.write("Error: Cannot find any MP2 ORCA output file in the directory you entered!\n")
        exit()

    ######### Extract atomic symbols, coordinates, energies and forces #########

    sys.stdout.write("===> Starting extracing cartesian coordinates from MP2 output files\n")

    A, E, C, F = [], [], [], []
    for no, file in enumerate(files):
        a, e, c, f = getMol(file)
        A.append(a)
        E.append(e)
        C.append(c)
        F.append(f)
        sys.stdout.write("File {0} : {1}\n".format(no + 1, file))

    ######### Write extended XYZ file #########

    sys.stdout.write("===> Writing output files\n")
    
    name_out = args.output
    if name_out is None:
        name_out = 'sGDML_dataset_from_MP2'
        sys.stdout.write("===> Output name not specified, default name '%s' will be used\n"
                         % name_out
                         )
    basename = name_out.split('.')[-1]
    extxyz_out = basename + '.xyz'
    energies_out = basename + '_energies.dat'
    npz_out = basename + '.npz'
    
    extxyz_out = mp2dir + '/' + extxyz_out
    energies_out = mp2dir + '/' + energies_out

    o = open(extxyz_out, 'w+')
    oe = open(energies_out, 'w+')
    # loop over files
    for i in range(len(files)):
        # loop over conformers
        for j in range(len(E[i])):
            o.write('{0}\n'.format(len(A[i][j])))
            o.write('{0:16.12f}\n'.format(E[i][j]))
            oe.write('{0:16.12f}\n'.format(E[i][j]))
            # loop over geometries
            for k in range(len(C[i][j])):
                o.write('{0}\t{1:9.8f}\t{2:9.8f}\t{3:9.8f}\t{4:9.8f}\t{5:9.8f}\t{6:9.8f}\n'
                        .format(A[i][j][k],
                                C[i][j][k][0],
                                C[i][j][k][1],
                                C[i][j][k][2],
                                F[i][j][k][0],
                                F[i][j][k][1],
                                F[i][j][k][2]))
    o.close()
    oe.close()
    sys.stdout.write("===> Extended XYZ has been saved to '%s'\n" % extxyz_out)
    sys.stdout.write("===> MP2 energies has been saved to '%s'\n" % energies_out)
    sys.stdout.write("===> All tasks completed!\n")

    # ######### Convert extended xyz to numpy zip file #########

    # sys.stdout.write("[/] Start converting extxyz to numpy zip\n")
    # os.system(sGDML_convert_script + ' ' + mp2dir + '/' + extxyz_out + ' -o')
    # sys.stdout.write("[/] Stop converting extended xyz to numpy zip\n")

    # ######### Move dataset (.npz) file to target directory #########

    # cwd_mp2dir = os.path.abspath(mp2dir)
    # cwd_run = os.getcwd()
    # if not cwd_mp2dir == cwd_run:
    #     os.system('mv ' + cwd_run + '/' + npz_out + ' ' + cwd_mp2dir)
