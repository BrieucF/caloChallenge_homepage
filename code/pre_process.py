import os

from HighLevelFeatures import HighLevelFeatures as HLF
import numpy as np
import h5py
import matplotlib.pyplot as plt

from XMLHandler import XMLHandler

add_total_energy = True

dataset_dir = "/eos/user/b/brfranco/caloChallenge/"
photon_file = h5py.File(os.path.join(dataset_dir, 'dataset_1_photons_1.hdf5'), 'r')
xmlhandler = XMLHandler('photon', filename = 'binning_dataset_1_photons.xml')
output_file_name = '5k_photons_1024_withEnergySum.hdf5'
max_evt = 10000
current_evt = 0
print(photon_file['showers'][0].shape[0])

n_cells = photon_file['showers'][0].shape[0]
showers = []
incident_energies = []
print("Starting loop")
for shower_index in range(photon_file['showers'].shape[0]):
    energy = photon_file['incident_energies'][shower_index][0]
    if energy == 1024.:
        print(current_evt)
        current_evt += 1
        if add_total_energy:
            tmp_list = photon_file['showers'][shower_index].tolist()
            tmp_list.append(photon_file['showers'][shower_index].sum())
            tmp_list.append(energy)
            showers.append(tmp_list)
            incident_energies.append(energy)
        else:
            showers.append(photon_file['showers'][shower_index])
    if current_evt == max_evt:
        break
shower_array = np.asarray(showers)
incident_energies_array = np.asarray(incident_energies)
h5f = h5py.File(output_file_name, 'w')
h5f.create_dataset('showers', data=shower_array)
h5f.create_dataset('incident_energies', data=incident_energies_array)
h5f.create_dataset('n_cells', data=photon_file['showers'][0].shape[0])
h5f.close()

#    if energy not in energy_separated_showers.keys():
#        energy_separated_showers[energy] = []
#    energy_separated_showers[energy].append(photon_file['showers'][shower_index])

#print(xmlhandler.GetRelevantLayers())
#print(xmlhandler.GetTotalNumberOfBins())
#print(xmlhandler.r_edges)

#print(np.array(photon_file['showers'][:]).shape)

# retrieve a 3D array with shower in 3D
#showers_3d = np.array()
#for shower_1d in photon_file['showers']:
#    shower_1d
#energy_separated_showers = {}
#print(energy_separated_showers.keys())
#training_showers = np.asarray(energy_separated_showers['256'])
