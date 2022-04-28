import numpy as np
import time
import tensorflow as tf
input_data = tf.keras.datasets.mnist.load_data()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Normalization
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import activations

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, n_cells_output, img_cols, channel):

        self.n_cells_output = n_cells_output
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        dropout = 0.4
        self.D = Sequential()
        #input_shape = (self.n_cells_output, self.img_cols, self.channel)
        #print(input_shape)
        #self.D.add(Normalization(axis = None))
        self.D.add(Dense(2*self.n_cells_output, input_shape=(self.n_cells_output,)))
        self.D.add(Dense(2*self.n_cells_output, activation='relu'))
        #self.D.add(Dropout(dropout))
        self.D.add(Dense(2*self.n_cells_output, activation='relu'))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        #self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        self.G.add(Dense(self.n_cells_output*2, input_dim=self.n_cells_output))
        # Out: dim x dim x depth
        self.G.add(Dense(self.n_cells_output*2, activation='relu'))
        #self.G.add(Dropout(dropout))
        self.G.add(Dense(self.n_cells_output*2, activation='relu'))
        self.G.add(Dense(self.n_cells_output*2, activation='relu'))
        self.G.add(Dropout(dropout))
        self.G.add(Dense(self.n_cells_output))
        self.G.add(Activation('relu'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(learning_rate=0.0002, decay=6e-8)
        #optimizer = RMSprop()
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        #self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
        self.DM.compile(loss='minimax_discriminator_loss', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(learning_rate=0.0008, decay=3e-8)
        #optimizer = RMSprop()
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        #self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
        self.AM.compile(loss='minimax_generator_loss', optimizer=optimizer,\
            metrics=['accuracy'])
        self.AM.summary()
        return self.AM

    def make_discriminator_trainable(self, val):
        self.D.trainable = val
        for l in self.D.layers:
            l.trainable = val



class MNIST_DCGAN(object):
    def __init__(self, training_data, n_cells_output):

        self.n_cells_output = training_data.shape[1]
        self.n_random_input = training_data.shape[1]
        self.img_cols = 1
        self.channel = 1

        self.x_train = training_data
        print("Train on %d events"%self.x_train.shape[0])
        #self.x_train = self.x_train.reshape(-1, self.n_cells_output,\
        #    self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN(self.n_cells_output, self.img_cols, self.channel)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=1000, batch_size=500, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, self.n_random_input])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :]
            # generate fake images
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.n_random_input])
            images_fake = self.generator.predict(noise)
            #print(images_train[0])
            #print(images_fake[0])
            print(round(images_train[0].sum()))
            print(round(images_fake[0].sum()))
            # train the discriminator to recognize them as fake comparing with real images
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            self.DCGAN.make_discriminator_trainable(True)
            #self.adversarial.summary()
            d_loss = self.discriminator.train_on_batch(x, y)

            # generate only fake images and train the generator to fool the discriminator (whose weights have to be frozen)
            self.DCGAN.make_discriminator_trainable(False)
            #self.adversarial.summary()
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.n_random_input])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1]) # the adversarial performance reflects how well the Generator succeeds to fool the discriminator (try to have the discri output one for fake showers)
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
        self.adversarial.save('models/adversarial.tf')
        self.generator.save('models/generator.tf')
        self.generator.save('models/discriminator.tf')

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, self.n_random_input])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.n_cells_output, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

#if __name__ == '__main__':
    #mnist_dcgan = MNIST_DCGAN()
    #timer = ElapsedTimer()
    #mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    #timer.elapsed_time()
    #mnist_dcgan.plot_images(fake=True)

