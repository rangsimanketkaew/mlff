# !/usr/bin/env python3

"""
Rangsiman Ketkaew
May 11st, 2020

Normal usage:
$ python dihedral_pca.py
"""

import os
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
# from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.ipython_useSVG = True

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_nonstd_ext_xyz(f):
    n_atoms = None

    R, z, E, F = [], [], [], []
    for i, line in enumerate(f):
        line = line.strip()
        if not n_atoms:
            n_atoms = int(line)
            # print('Number atoms per geometry: {:,}'.format(n_atoms))

        file_i, line_i = divmod(i, n_atoms + 2)

        if line_i == 1:
            try:
                e = float(line)
            except ValueError:
                pass
            else:
                E.append(e)

        cols = line.split()
        if line_i >= 2:
            R.append(list(map(float, cols[1:4])))
            if file_i == 0:  # first molecule
                z.append(cols[0])
            F.append(list(map(float, cols[4:7])))

    # print('Number geometries found so far: {:,}'.format(file_i + 1))

    R = np.array(R).reshape(-1, n_atoms, 3)
    z = np.array(z)
    E = None if not E else np.array(E)
    F = np.array(F).reshape(-1, n_atoms, 3)

    f.close()
    return (R, z, E, F)


# from https://sourceforge.net/p/rdkit/mailman/message/34554502/ (Paulo Tosco)
def enumerateTorsions(mol):
    torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = mol.GetSubstructMatches(torsionQuery)
    torsionList = []
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = mol.GetBondBetweenAtoms(idx2, idx3)
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)
        if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
             and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
            or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
            continue
        for b1 in jAtom.GetBonds():
            if (b1.GetIdx() == bond.GetIdx()):
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if ((b2.GetIdx() == bond.GetIdx())
                        or (b2.GetIdx() == b1.GetIdx())):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if (idx4 == idx1):
                    continue
                torsionList.append((idx1, idx2, idx3, idx4))
    return torsionList


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber',
                                        str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def cal_dihedral(p0, p1, p2, p3):
    """Praxeolitic formula
    1 sqrt, 1 cross product
    ref: https://stackoverflow.com/questions/20305272
    """
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

if __name__ == "__main__":
    #### Read extended xyz file ####
    inp = 'sGDML_dataset_from_MP2_plus_re.xyz'
    with open(inp) as f:
        R, z, E, F = read_nonstd_ext_xyz(f)
        
    #### Define SMILES ####
    # folder = '/home/nutt/AA/'
    # xyzname = 'AA_struct-3640_UFFmin_phi_135.003080083_psi_134.998144891.xyz'
    # xyzpath = folder + xyzname
    # xyz2sml = os.popen('obabel ' + xyzpath + ' -o smiles').read()
    # sml = xyz2sml.split()[0]

    sml = 'CC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NC'  ## alanine tripeptide
    mol = Chem.MolFromSmiles(sml)
    mol = Chem.RemoveHs(mol)
    # DrawingOptions.includeAtomNumbers = True
    
    # Find all dihedral backbone
    torsionList = enumerateTorsions(mol)
    
    # Compute cosine and sine of each dihedral angles
    dih_cos = np.zeros((R.shape[0], len(torsionList)))   
    dih_sin = np.zeros((R.shape[0], len(torsionList)))   
    for i in range(R.shape[0]):
        for j, t in enumerate(torsionList):
            dih_angle = cal_dihedral(R[i][t[0]], R[i][t[1]], R[i][t[2]], R[i][t[3]])
            dih_cos[i][j] = np.cos(dih_angle * np.py / 180.0)
            dih_sin[i][j] = np.sin(dih_angle * np.py / 180.0)
    
    # Reshape and transpose energy matrix from 1D array to 2D array
    E_re = np.reshape(E, (-1, 1))
    
    # Combine cosine and sin together
    # mat = np.concatenate((dih_cos, dih_sin, E_re), axis=1)
    mat = np.concatenate((dih_cos, dih_sin), axis=1)
    
    #### Running PCA ####
    # standardize the data
    mat_std = StandardScaler().fit_transform(mat)
    
    # Set the number of components for PCA
    pca = PCA(n_components=3)
    reduced_mat = pca.fit_transform(mat_std)
    print(np.cumsum(reduced_mat.explained_variance_ratio))
    
    # the principal components are available as `pca.components_`
    # print(pca.components_)
    
    #### Plot the data in this projection ####

    # 2 components plot
    # plt.figure()
    # plt.scatter(reduced_mat[:, 0], reduced_mat[:,1], reduced_mat[:,2], marker='x')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.zlabel('PC3')
    # plt.title('Dihedral PCA: alanine tripeptide')
    # cbar = plt.colorbar()
    # cbar.set_label('Energy (kcal/mol)')
    # plt.show()
    
    # 3 components plot
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(*zip(*reduced_mat), c=E, marker='o')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    cbar = fig.colorbar(p)
    cbar.set_label('Energy (kcal/mol)')
    plt.show()
