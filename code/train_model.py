import os

from HighLevelFeatures import HighLevelFeatures as HLF
import numpy as np
import h5py
import matplotlib.pyplot as plt

from XMLHandler import XMLHandler

import my_dcgan_mnist as dcgan_mnist
#import dcgan_mnist as dcgan_mnist

dataset_dir = "/eos/user/b/brfranco/caloChallenge/"

# creating instance of HighLevelFeatures class to handle geometry based on binning file
#HLF_1_photons = HLF('photon', filename='binning_dataset_1_photons.xml')

# loading the .hdf5 datasets
#photon_file = h5py.File(os.path.join(dataset_dir, 'dataset_1_photons_1.hdf5'), 'r')
photon_file = h5py.File('/afs/cern.ch/user/b/brfranco/work/public/Fellow/caloChallenge2022/homepage/code/5k_photons_1024_withEnergySum.hdf5', 'r')

# each file contains one dataset for the incident energy and one for the showers.
#for dataset in photon_file:
#    # name of the datasets:
#    print("dataset name: ", dataset)
#    print("dataset shape:", photon_file[dataset][:].shape)
#print('\n')

#print(photon_file['showers'])
#for shower in photon_file['showers']:
#    print(shower.shape[0])
#    print(shower.shape[1])
#    print(shower.shape[2])

#xmlhandler = XMLHandler('photon', filename = 'binning_dataset_1_photons.xml')

#print(xmlhandler.GetRelevantLayers())
#print(xmlhandler.GetTotalNumberOfBins())
#print(xmlhandler.r_edges)

#print(np.array(photon_file['showers'][:]).shape)

# retrieve a 3D array with shower in 3D
#showers_3d = np.array()
#for shower_1d in photon_file['showers']:
#    shower_1d
#energy_separated_showers = {}
#    if energy not in energy_separated_showers.keys():
#        energy_separated_showers[energy] = []
#    energy_separated_showers[energy].append(photon_file['showers'][shower_index])
#print(energy_separated_showers.keys())
#training_showers = np.asarray(energy_separated_showers['256'])

#for shower in photon_file['showers']:
#    np.insert(shower, 0, shower.sum())
#    print(shower.shape)

my_gan = dcgan_mnist.MNIST_DCGAN(photon_file['showers'][:], photon_file['n_cells'])
timer = dcgan_mnist.ElapsedTimer()
my_gan.train()
timer.elapsed_time()
