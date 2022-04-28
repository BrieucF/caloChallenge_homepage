import keras
import numpy as np
import os
import h5py

model_path = "models/generator.tf"
n_showers = 5000

shower_path = 'generated_showers'

energy = 1024.0

model = keras.models.load_model(model_path)
n_input = model.layers[0].input.shape[1]
noise = np.random.uniform(-1.0, 1.0, size=[n_showers, n_input])

showers = model.predict(noise)
energies = energy*np.ones((showers.shape[0], 1))
print(energies)
print(showers)
print(energies.shape)
print(showers.shape)
gen_shower_file = os.path.join(shower_path, 'exo_1_photon.h5')
hf = h5py.File(gen_shower_file, 'w')
hf.create_dataset('incident_energies', data=energies)
hf.create_dataset('showers', data=showers)
hf.close()

os.system("python evaluate.py -i %s -r source/dataset_1_photons_2.pkl  -d 1-photons"%(gen_shower_file))
