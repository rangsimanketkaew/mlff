#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 27th, 2020

Train the energies- and forces-based force field model 
using sGDML trainer module

Normal command-line:
sgdml all TRAINING.npz N_training N_test 
    --test_dataset TEST.npz \
    --task_dir TASK_OUT \
    --model_file MODEL_OUT \
        > STDOUT.out 2>&1 &

Normal usage:
$ python step_6_sgdml_train_model.py
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

# from sgdml.train import GDMLTrain


if __name__ == '__main__':

    ######### Select molecules and dataset split ratio #########

    parser = argparse.ArgumentParser(
        description='Train, validate and test model using sGDML.'
    )
    parser.add_argument(
        'npz_dir',
        metavar='npz-directory',
        type=str,
        help='path to directory storing NumPy zip (npz) files',
    )

    args = parser.parse_args()
    npz_dir = args.npz_dir
    if not os.path.exists(npz_dir):
        sys.stderr.write("Error: directory you entered does not exist\n")
        exit()

    # Studied molecules
    # MP2_set_1_p53 = [
    #     'EG', 'FK', 'GQ', 'HK', 'HL', 'KG', 'KK', 'KL',
    #     'KS', 'KT', 'LK', 'LM', 'MF', 'QS', 'RH', 'SH',
    #     'SK', 'SR', 'ST', 'TE', 'TS', 'ZZ'
    # ]
    #
    # MP2_set_1_AR = [
    #     'AA', 'AQ', 'AS', 'EE', 'EQ', 'GA', 'KP', 'KQ',
    #     'KY', 'LE', 'LL', 'LQ', 'PG', 'QE', 'QK', 'QQ',
    #     'SA', 'SL'
    # ]

    # Split ratio (%): [A, X]
    #   where A and X are data points weight (%) for training and actual training sets.
    ratio = {
        '80-80': [80, 80],
        # '75-80': [75, 80],
        # '70-80': [70, 80],
    }

    all_dataset = []
    npzs = glob.glob(npz_dir + '/' + '*.npz')
    if npzs:
        for npz in npzs:
            for item in ratio.keys():
                coll = {'dataset': npz, 'ptrain': ratio[item][0], 'pactrain': ratio[item][1]}
                all_dataset.append(coll)
    else:
        sys.stderr.write("Error: no dataset (.npz) file in directory you entered\n")

    for dataset in all_dataset:
        ######### Prepare dataset for sGDML #########

        lst = np.load(dataset['dataset'])
        ndata = np.int(len(lst['R']))
        ptrain = np.float(dataset['ptrain'])
        ptest = np.float(100.0 - ptrain)
        pactrain = np.float(dataset['pactrain'])
        pvalid = np.float(100.0 - pactrain)

        # split dataset into training set + test set
        ntrain = np.int(np.ceil(ndata * ptrain / 100.0))
        ntest = np.int(ndata - ntrain)

        # split training set into actual training + validation set
        nactrain = np.int(np.ceil(ntrain * pactrain / 100))
        nvalid = np.int(ntrain - nactrain)

        sys.stdout.write("\n===========================================\n")
        sys.stdout.write("Info Summary for Training Model using sGDML\n")
        sys.stdout.write("===========================================\n")
        sys.stdout.write("Dataset samples      = {0}\n".format(ndata))
        sys.stdout.write("Training set samples = {0}\n".format(ntrain))
        sys.stdout.write("###########################################\n")
        sys.stdout.write("Actual training set samples = {0}\n".format(nactrain))
        sys.stdout.write("Validating set samples      = {0}\n".format(nvalid))
        sys.stdout.write("Test set samples            = {0}\n".format(ntest))
        sys.stdout.write("###########################################\n")

        ######### Start training, validating and testing model #########

        sGDML_trainer = 'sgdml'
        if shutil.which(sGDML_trainer) is None:
            sys.stderr.write("Error: can't call \'sgdml\', please check your PATH environment variable\n")

        # Take samples from dataset as validation and test sets
        samples = dataset['dataset']
        valid_set = samples
        test_set = samples

        model_name = npz_dir + '/' + os.path.basename(samples.split('.npz')[0]) + '_trained_' + \
                     'ptrain_' + str(dataset['ptrain']) + '%_' + \
                     'pactrain_' + str(dataset['pactrain']) + '%'

        split_ratio = str(nactrain) + ' ' + str(nvalid) + ' ' + str(ntest)

        os.system(sGDML_trainer + ' all ' + samples + ' ' + split_ratio +
                  ' --validation_dataset ' + valid_set + ' --test_dataset ' + test_set +
                  ' --task_dir ' + model_name + ' --model_file ' + model_name)

        ######### Train, validate and test model via Python API #########

        # train_set = npz
        # valid_set = train_set
        #
        # gdml = GDMLTrain()
        # task = gdml.create_task(train_set, ntrain,
        #                         valid_set, nvalid,
        #                         sig=10, lam=1e-15)
        # try:
        #     model = gdml.train(task)
        # except Exception:
        #     sys.stderr.write('Error')
        #     sys.exit()
        # else:
        #     np.savez_compressed("model.npz", **model)
