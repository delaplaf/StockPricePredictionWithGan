import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import GRU, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, ELU, ReLU
from tensorflow.keras.metrics import Mean
from tensorflow.keras import Sequential, regularizers
from tensorflow.python.client import device_lib
from tensorflow.keras import Model
from tensorflow.keras import callbacks


# Define the generator
def Generator(input_dim, output_dim, feature_size):
    model = Sequential()
    model.add(GRU(units=256,
                  return_sequences=True,
                  input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.02,
                  recurrent_regularizer=regularizers.l2(1e-3)))
    model.add(GRU(units=128,
                  recurrent_dropout=0.02,
                  recurrent_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(64, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(32, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(units=output_dim))
    return model


# Define the discriminator
def Discriminator():
    model = Sequential()
    model.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Flatten())
    model.add(Dense(220, use_bias=True))
    model.add(LeakyReLU())
    model.add(Dense(220, use_bias=True))
    model.add(ReLU())
    model.add(Dense(1))
    return model


# Train WGAN-GP model
class WGAN_GP(Model):
    def __init__(self, generator, discriminator, d_steps=1, g_steps=3, gp_weight=10):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.gp_weight = gp_weight
        self.disc_loss_tracker = Mean(name="discriminator_loss")
        self.gen_loss_tracker = Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_output, fake_output):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interpolated data
        alpha = tf.random.normal([batch_size, tf.shape(real_output)[1], 1], 0.0, 1.0)
        diff = fake_output - tf.cast(real_output, tf.float32)
        interpolated = tf.cast(real_output, tf.float32) + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))

        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_input, real_price, yc = data

        # cast in float32 to concat with generator output
        yc = tf.cast(yc, tf.float32)
        real_price = tf.cast(real_price, tf.float32)

        batch_size = tf.shape(real_input)[0]

        # Train the discriminator
        for _ in range(self.d_steps):
            with tf.GradientTape() as d_tape:
                # generate fake output
                generated_data = self.generator(real_input, training=True)

                generated_shape = tf.shape(generated_data)
                generated_data_reshape = tf.reshape(generated_data, [generated_shape[0], generated_shape[1], 1])
                fake_output = tf.concat([generated_data_reshape, yc], axis=1)

                # get real output
                real_price_shape = tf.shape(real_price)
                real_price_reshape = tf.reshape(real_price, [real_price_shape[0], real_price_shape[1], 1])
                real_output = tf.concat([real_price_reshape, yc], axis=1)

                # Get the logits for the fake images
                D_real = self.discriminator(real_output, training=True)
                # Get the logits for real images
                D_fake = self.discriminator(fake_output, training=True)
                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(D_real, D_fake)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_output, fake_output)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train the generator
        for _ in range(self.g_steps):
            with tf.GradientTape() as g_tape:
                # generate fake output
                generated_data = self.generator(real_input, training=True)

                generated_shape = tf.shape(generated_data)
                generated_data_reshape = tf.reshape(generated_data, [generated_shape[0], generated_shape[1], 1])
                fake_output = tf.concat([generated_data_reshape, yc], axis=1)

                # Get the discriminator logits for fake images
                G_fake = self.discriminator(fake_output, training=True)
                # Calculate the generator loss
                g_loss = self.g_loss_fn(G_fake)

            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Monitor loss.
        self.disc_loss_tracker.update_state(d_loss)
        self.gen_loss_tracker.update_state(g_loss)

        return {
            "d_loss": self.disc_loss_tracker.result(),
            "g_loss": self.gen_loss_tracker.result(),
        }


class SaveModel(callbacks.Callback):
      def __init__(self, save_path: str, epoch_model_save: int):
          self.save_path = save_path
          self.epoch_model_save = epoch_model_save
     
      def on_epoch_end(self, epoch: int, logs=None):
          if (epoch + 1) % self.epoch_model_save == 0:
              save_path = os.path.join(self.save_path,
                                       "wgan_gp_*epoch*epoch.h5")
              # Save the generator model                  
              self.model.generator.save(save_path.replace("*epoch*",
                             "{:04d}".format(epoch + 1))
                             )


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(D_real, D_fake):
    # Calculate discriminator loss using fake and real logits
    real_loss = tf.cast(tf.reduce_mean(D_real), tf.float32)
    fake_loss = tf.cast(tf.reduce_mean(D_fake), tf.float32)
    return fake_loss-real_loss


# Define the loss functions for the generator.
def generator_loss(G_fake):
    return -tf.reduce_mean(G_fake)


if __name__ == '__main__':
    # Load data
    path = r'Data\dataPreprocessed'
    X_train = np.load(os.path.join(path, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path, "y_test.npy"), allow_pickle=True)
    yc_train = np.load(os.path.join(path, "yc_train.npy"), allow_pickle=True)
    yc_test = np.load(os.path.join(path, "yc_test.npy"), allow_pickle=True)

    input_generator_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_generator_dim = y_train.shape[1]

    # Hyperparameter
    EPOCHS = 5
    BATCH_SIZE = 128
    D_STEPS = 1
    G_STEPS = 3
    GP_WEIGHT = 10
    D_LEARNING_RATE = 0.0001
    G_LEARNING_RATE = 0.0001

    # Instantiate the optimizer for both networks
    discriminator_optimizer = tf.keras.optimizers.Adam(D_LEARNING_RATE)
    generator_optimizer = tf.keras.optimizers.Adam(G_LEARNING_RATE)

    generator = Generator(input_generator_dim, output_generator_dim, feature_size)
    generator.compile()
    discriminator = Discriminator()
    wgan_gp = WGAN_GP(generator, discriminator, D_STEPS, G_STEPS, GP_WEIGHT)

    # Compile the WGAN model.
    wgan_gp.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )

    data = X_train, y_train, yc_train
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, yc_train))
    dataset = dataset.batch(BATCH_SIZE)

    save_path = r'Models\trying'
    EPOCH_MODEL_SAVE = 1
    callback = [SaveModel(save_path, EPOCH_MODEL_SAVE)]

    history = wgan_gp.fit(dataset, epochs=EPOCHS, callbacks=callback)

    # summarize history for loss
    plt.plot(history.history['d_loss'])
    plt.plot(history.history['g_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['d_loss', 'g_loss'], loc='upper left')
    plt.show()