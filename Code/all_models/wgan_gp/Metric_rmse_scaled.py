import tensorflow as tf


class Metric_rmse_scaled(tf.keras.metrics.Metric):

    def __init__(self, y_scaler_function, **kwargs):
        super(Metric_rmse_scaled, self).__init__(**kwargs)
        self.rmse_scaled = self.add_weight(name='my_metric', initializer='zeros')
        self.y_scaler_function = y_scaler_function

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = (y_true - tf.constant(self.y_scaler_function.min_)) / tf.constant(self.y_scaler_function.scale_)
        y_pred = (y_pred - tf.constant(self.y_scaler_function.min_)) / tf.constant(self.y_scaler_function.scale_)
        rmse = tf.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))
        self.rmse_scaled.assign(tf.cast(rmse, tf.float32))

    def result(self):
        return self.rmse_scaled

    def reset_state(self):
        self.rmse_scaled.assign(0.)