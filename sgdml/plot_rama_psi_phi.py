#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 24th, 2020

Plot Ramachandra Psi/Phi correlation plot

Prerequisites:
pip install git+https://github.com/mdtraj/mdtraj@master
pip install matplotlib
pip install numpy

Normal usage:
    1. Convert extxyz --> pdb using openbabel v.3.x 
    2. Convert pdb --> h5 using mdconvert which is a module of mdtraj
    3. Edit the user-defined parameters below
    4. Execute this script

Notes:
1. OpenBabel v.3.x can be installed by using Ubuntu's apt, like this:
    $ sudo apt install openbabel
2. mdtraj can be installed by using pip & git commands, like this:
    $ pip install git+https://github.com/mdtraj/mdtraj@master
"""

import numpy as np
import mdtraj
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar

############### User-defined ###############
inp = 'AA.h5'
name = 'Alanine Dipeptide'
energies = open('sGDML_dataset_from_MP2_energies.dat', 'r')
# Specify the four atoms that together parameterize the phi and psi dihedral angles.
psi_indices, phi_indices = [2, 4, 5, 6], [6, 5, 7, 9]
############################################

all_ener = energies.read().splitlines()
all_ener = list(map(float, all_ener))
# print(all_ener)

# Load up the trajectory
# traj = mdtraj.load(name + '.h5')
traj = mdtraj.load(inp)

atoms, bonds = traj.topology.to_dataframe()

angles = mdtraj.geometry.compute_dihedrals(traj, [phi_indices, psi_indices])
# print(angles)
angles = np.asarray(angles) * 180.0 / np.pi

# Let's plot our dihedral angles in a scatter plot using matplotlib.
# What conformational states of Alanine dipeptide did we sample?

fig = plt.figure()
plt.title('Dihedral Map: ' + name)
plt.scatter(angles[:, 0], angles[:, 1], marker='o', c=all_ener)
cbar = colorbar()
cbar.set_label('Energy')
plt.xlabel(r'$\Phi$ Angle [degrees]')
plt.xlim(-180.0, 180.0)
plt.ylabel(r'$\Psi$ Angle [degrees]')
plt.ylim(-180.0, 180.0)
plt.show()
