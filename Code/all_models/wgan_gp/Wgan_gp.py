import tensorflow as tf

from tensorflow.keras.metrics import Mean
from tensorflow.keras import Model

# Train WGAN-GP model
class WGAN_GP(Model):
    def __init__(self, generator, discriminator, custom_metric, d_steps=1, g_steps=3, gp_weight=10):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.gp_weight = gp_weight
        self.disc_loss_tracker = Mean(name="discriminator_loss")
        self.gen_loss_tracker = Mean(name="generator_loss")
        self.metric_rmse_scaled = custom_metric

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.metric_rmse_scaled]
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, inputs):
        return self.generator(inputs)

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

    def d_loss(self, real_input, real_price, yc, batch_size, withTraining):
        # generate fake output
        generated_data = self.generator(real_input, training=withTraining)

        generated_shape = tf.shape(generated_data)
        generated_data_reshape = tf.reshape(generated_data, [generated_shape[0], generated_shape[1], 1])
        fake_output = tf.concat([yc, generated_data_reshape], axis=1)

        # get real output
        real_price_shape = tf.shape(real_price)
        real_price_reshape = tf.reshape(real_price, [real_price_shape[0], real_price_shape[1], 1])
        real_output = tf.concat([yc, real_price_reshape], axis=1)

        # Get the logits for the fake images
        D_real = self.discriminator(real_output, training=withTraining)
        # Get the logits for real images
        D_fake = self.discriminator(fake_output, training=withTraining)
        # Calculate discriminator loss using fake and real logits
        d_cost = self.d_loss_fn(D_real, D_fake)
        # Calculate the gradient penalty
        gp = self.gradient_penalty(batch_size, real_output, fake_output)
        # Add the gradient penalty to the original discriminator loss
        d_loss = d_cost + gp * self.gp_weight
        return d_loss

    def g_loss(self, real_input, yc, withTraining):
        # generate fake output
        generated_data = self.generator(real_input, training=withTraining)

        generated_shape = tf.shape(generated_data)
        generated_data_reshape = tf.reshape(generated_data, [generated_shape[0], generated_shape[1], 1])
        fake_output = tf.concat([yc, generated_data_reshape], axis=1)

        # Get the discriminator logits for fake images
        G_fake = self.discriminator(fake_output, training=withTraining)
        # Calculate the generator loss
        g_loss = self.g_loss_fn(G_fake)
        return g_loss, generated_data
        
    def train_step(self, data):
        real_input, real_price, yc = data

        # cast in float32 to concat with generator output
        yc = tf.cast(yc, tf.float32)
        real_price = tf.cast(real_price, tf.float32)

        batch_size = tf.shape(real_input)[0]

        # Train the discriminator
        for _ in range(self.d_steps):
            with tf.GradientTape() as d_tape:
                d_loss = self.d_loss(real_input, real_price, yc, batch_size, True)

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train the generator
        for _ in range(self.g_steps):
            with tf.GradientTape() as g_tape:
                g_loss, generated_data = self.g_loss(real_input, yc, True)

            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Monitor loss.
        self.disc_loss_tracker.update_state(d_loss)
        self.gen_loss_tracker.update_state(g_loss)

        # Monitor metric
        self.metric_rmse_scaled.update_state(real_price, generated_data)

        return {
            "d_loss": self.disc_loss_tracker.result(),
            "g_loss": self.gen_loss_tracker.result(),
            "rmse": self.metric_rmse_scaled.result()
        }

    def test_step(self, data):
        real_input, real_price, yc = data

        # cast in float32 to concat with generator output
        yc = tf.cast(yc, tf.float32)
        real_price = tf.cast(real_price, tf.float32)

        batch_size = tf.shape(real_input)[0]

        # discriminator loss
        with tf.GradientTape() as d_tape:
            d_loss = self.d_loss(real_input, real_price, yc, batch_size, False)

        # Train the generator
        with tf.GradientTape() as g_tape:
            g_loss, generated_data = self.g_loss(real_input, yc, False)

        # Monitor loss.
        self.disc_loss_tracker.update_state(d_loss)
        self.gen_loss_tracker.update_state(g_loss)

        # Monitor metric
        self.metric_rmse_scaled.update_state(real_price, generated_data)

        return {
            "d_loss": self.disc_loss_tracker.result(),
            "g_loss": self.gen_loss_tracker.result(),
            "rmse": self.metric_rmse_scaled.result()
        }


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