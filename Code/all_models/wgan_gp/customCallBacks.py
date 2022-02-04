import os
import tensorflow as tf

from tensorflow.keras import callbacks


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


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_best_metric='val_d_loss'):
        self.save_path = os.path.join(save_path, "wgan_gp_best.h5")
        self.save_best_metric = save_best_metric
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if metric_value < self.best:
            self.best = metric_value
            self.model.generator.save(self.save_path)
            print("Current best epoch: ", epoch + 1)