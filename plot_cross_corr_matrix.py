# Rangsiman Ketkaew

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# read file
fx = 'train_mean_rmsd'
fy = 'n_train'
ecorr = 'e_corrcoef'
figname = fx + '_' + fy + '_' + ecorr + '.png'

x = np.loadtxt(fx, delimiter='\n', dtype=np.float)
y = np.loadtxt(fy, delimiter='\n', dtype=np.float)
e = np.loadtxt(ecorr, delimiter='\n', dtype=np.float)

plt.scatter(x, y, c=e, s=80, vmin=-1, vmax=1)
cb = plt.colorbar()
cb.set_label('energy_corr_coeff')
plt.xlabel(fx)
plt.ylabel(fy)
plt.savefig(figname)
plt.show()
