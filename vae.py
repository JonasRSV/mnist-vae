import tensorflow as tf
from typing import Callable, Tuple
import numpy as np

Mapping = Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


class VAE:
    def __init__(self, encoder: Mapping, decoder: Mapping, C=1.0):
        self.C = C
        self.encoder = encoder
        self.decoder = decoder

    def sample_given_z(self, z_batch: tf.Tensor):
        return self.decoder(z_batch)

    @tf.function
    def map_to_latent(self, x_batch: tf.Tensor):
        return self.encoder(x_batch)

    def map_from_latent(self, z_batch: tf.Tensor):
        return self.decoder(z_batch)

    @tf.function
    def loss_op(self, x_batch: tf.Tensor) -> tf.Tensor:
        lat_mu, lat_sigma = self.encoder(x_batch)

        # re-parametrization trick
        sample = lat_mu + tf.random.normal(lat_mu.shape, mean=0, stddev=1) * lat_sigma

        dat_mu, dat_sigma = self.decoder(sample)

        # We wish to maximize Elbo
        # Elbo = E[ \log p(x, z) / q(z)] = E[ \log (p(x | z) p(z)) / q(z) ]
        # = E[\log p(x | z)] + E[ \log p(z) / q(z)] = E [\log p(x | z)] - KL (p || q)

        # From this we see that we have 2 terms to optimize
        # E [\log p(x | z)] is often called reconstruction error, in this case we have assumed p(x | z) to be normal
        # giving E [\log p(x | z)] = - 1 / (dat_sigma * 2) (x - dat_mu)^T (x - dat_mu)
        # maximizing this is the same as minimizing 1 / 2 (x - dat_mu)^T (I @ dat_sigma)^{-1} (x - dat_mu)
        # which is pretty much the same as mean-squared-error between the prediction of the decoder and the data.

        reconstruction_error = (1 / 2) * tf.reduce_sum(tf.multiply(x_batch - dat_mu, x_batch - dat_mu) / (dat_sigma + 1e-10),
                                                       axis=1, keepdims=True)

        # Then we have the KL-divergence term.
        # - KL(p | | q) = E[ \log p(z) / q(z)]
        # For this we see that we need to make an assumption on the form of p(z)
        # we know q(z) since we assume q(z) = p(z | x) = N(0, 1)((z - lat_mu) /  lat_sigma) (given by encoder)
        # A common assumption is that p(z) is normal with mean 0 and spherical covariance C * I, giving
        # E[ \log p(z) / q(z)] = E[- (1 / (C * 2)) z^T z + (1 / 2) (z - lat_mu)^T (I @ lat_sigma)^{-1} (z - lat_mu)]
        # We estimate this expectation by sampling from q(z), typically we just use 1 sample since combining it
        # with mini-batch appears to work fine empirically, giving
        # E[ \log p(z) / q(z)] =  -(1 / (C * 2)) z^T z + (1 / 2) (z - lat_mu)^T (I @ lat_sigma)^{-1} (z - lat_mu)
        # maximizing the term above is the same as minimizing
        # (1 / (C * 2)) z^T z - (1 / 2) (z - lat_mu)^T (I @ lat_sigma)^{-1} (z - lat_mu)
        # = (1 / 2)((1/C) z^T z - (z - lat_mu)^T (I @ lat_sigma)^{-1} (z - lat_mu))

        kl_error = (1 / 2) * ((1 / self.C) * tf.reduce_sum(tf.multiply(sample, sample), 1, keepdims=True)
                              - tf.reduce_sum(tf.multiply(sample - lat_mu, sample - lat_mu) / (lat_sigma + 1e-10),
                                              axis=1, keepdims=True))

        #print("Recon", reconstruction_error, "KL", kl_error)
        #tf.print(reconstruction_error)
        #tf.print(kl_error)

        # this is total error
        return tf.reduce_mean(reconstruction_error) + tf.reduce_mean(kl_error)

    @tf.function
    def train_op(self, x_batch: tf.Tensor, optimizer: tf.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss_op(x_batch)

        variables = tape.watched_variables()
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            zip(gradient,
                variables)
        )

        return loss

    def fit(self, data, batch_size=1, epochs= 3):
        data = data.astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data). \
            shuffle(buffer_size=10000). \
            batch(batch_size=batch_size)

        optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        for epoch in range(epochs):
            error, iter = 0, 0
            for batch, x in enumerate(dataset.as_numpy_iterator()):
                loss = self.train_op(x, optimizer)
                error, iter = error + loss, iter + 1

                print(f"Error {error / iter}", end="        \r")
            print(f"epoch {epoch} - {error / iter}")


def dense_encoder_decoder(data_dim: int,
                          latent_dim: int,
                          layers: int,
                          hidden_units: int,
                          activation: tf.function) -> (Mapping, Mapping):
    assert layers > 0

    encoder_hidden_layers = []
    for _ in range(layers):
        encoder_hidden_layers.append(
            tf.keras.layers.Dense(hidden_units, activation=activation)
        )

    decoder_hidden_layers = []
    for _ in range(layers):
        decoder_hidden_layers.append(
            tf.keras.layers.Dense(hidden_units, activation=activation)
        )

    encoder_mu_layer, encoder_sigma_layer = tf.keras.layers.Dense(latent_dim, activation=None),\
        tf.keras.layers.Dense(latent_dim, activation=None)

    decoder_mu_layer, decoder_sigma_layer = tf.keras.layers.Dense(data_dim, activation=None), \
        tf.keras.layers.Dense(data_dim, activation=None)

    @tf.function
    def encoder(data: tf.Tensor):
        x = data
        for i in range(layers):
            x = encoder_hidden_layers[i](x)

        encoder_mu = encoder_mu_layer(x)
        log_encoder_sigma = encoder_sigma_layer(x)
        return encoder_mu, tf.exp(log_encoder_sigma)

    @tf.function
    def decoder(latent: tf.Tensor):
        x = latent
        for i in range(layers):
            x = decoder_hidden_layers[i](x)

        decoder_mu = decoder_mu_layer(x)
        #log_decoder_sigma = decoder_sigma_layer(x)

        return decoder_mu, 1.0 #tf.exp(log_decoder_sigma)

    return encoder, decoder
