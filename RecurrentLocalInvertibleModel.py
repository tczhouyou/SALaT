import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')

reprod_loss_tracker = keras.metrics.Mean(name="reprod_loss")
latent_loss_tracker = keras.metrics.Mean(name="latent_loss")

single_ent_tracker = keras.metrics.Mean(name="single_ent")
total_ent_tracker = keras.metrics.Mean(name="total_ent")

kernel_initializer = initializers.RandomUniform(minval=-1e-3, maxval=1e-3)
kernel_regularizer = None

def cov_f(x0, x1, l, sigf):
    return np.square(sigf) * np.exp(- np.square(x0 - x1) / (2 * np.square(l)))


def constructGP(timesteps, l, sigf, sign):
    K = np.zeros(shape=(timesteps, timesteps))
    timestamps = np.linspace(1, timesteps, timesteps)
    for i in range(timesteps):
        for j in range(timesteps):
            x0 = timestamps[i]
            x1 = timestamps[j]
            K[i,j] = cov_f(x0, x1, l, sigf)

    K = K + sign * np.identity(timesteps)
    mu = np.zeros(shape=timesteps)
    scale = tf.linalg.cholesky(K)
    mvn = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale)
    return mvn


def get_gru(dim, act=None, is_bidirectional=False, merge_mode="sum"):
    layer = layers.GRU(dim, activation=act, return_sequences=True,
                       kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if is_bidirectional:
        layer = layers.Bidirectional(layer, merge_mode=merge_mode)

    return layer


class RecurrentNVP2D(keras.layers.Layer):
    def __init__(self, struct, name="RecurrentNVP2D", scope="RIL", **kwargs):
        super(RecurrentNVP2D, self).__init__(name=name, **kwargs)
        keras.backend.name_scope(scope)
        s1_layers=struct['s1_layers']
        t1_layers=struct['t1_layers']
        s2_layers=struct['s2_layers']
        t2_layers=struct['t2_layers']
        s_act=struct['s_act']
        t_act=struct['t_act']
        is_bidirectional = struct['is_bidirectional']

        self.s1_layers = []
        for i in range(len(s1_layers)):
            if s_act == "LeakyReLU":
                self.s1_layers.append(get_gru(s1_layers[i], act=None, is_bidirectional=is_bidirectional))
                self.s1_layers.append(layers.LeakyReLU())
            else:
                self.s1_layers.append(get_gru(s1_layers[i], act=s_act, is_bidirectional=is_bidirectional))

        self.s2_layers = []
        for i in range(len(s2_layers)):
            if s_act == "LeakyReLU":
                self.s2_layers.append(get_gru(s2_layers[i], act=None, is_bidirectional=is_bidirectional))
                self.s2_layers.append(layers.LeakyReLU())
            else:
                self.s2_layers.append(get_gru(s2_layers[i], act=s_act, is_bidirectional=is_bidirectional))

        self.t1_layers = []
        for i in range(len(t1_layers)):
            if t_act == "LeakyReLU":
                self.t1_layers.append(get_gru(t1_layers[i], act=None, is_bidirectional=is_bidirectional))
                self.t1_layers.append(layers.LeakyReLU())
            else:
                self.t1_layers.append(get_gru(t1_layers[i], act=t_act, is_bidirectional=is_bidirectional))

        self.t2_layers = []
        for i in range(len(t2_layers)):
            if t_act == "LeakyReLU":
                self.t2_layers.append(get_gru(t2_layers[i], act=None, is_bidirectional=is_bidirectional))
                self.t2_layers.append(layers.LeakyReLU())
            else:
                self.t2_layers.append(get_gru(t2_layers[i], act=t_act, is_bidirectional=is_bidirectional))

    def apply_layers(self, inputs, clayers):
        outputs = inputs
        for i in range(len(clayers)):
            outputs = clayers[i](outputs)

        return outputs

    def s1(self, inputs):
        return self.apply_layers(inputs, self.s1_layers)

    def s2(self, inputs):
        return self.apply_layers(inputs, self.s2_layers)

    def t1(self, inputs):
        return self.apply_layers(inputs, self.t1_layers)

    def t2(self, inputs):
        return self.apply_layers(inputs, self.t2_layers)

    def forward(self, u1, u2):
        logit_1 = self.s1(u2)
        v1 = tf.multiply(u1, tf.exp(logit_1)) + self.t1(u2)
        logit_2 = self.s2(v1)
        v2 = tf.multiply(u2, tf.exp(logit_2)) + self.t2(v1)
        z = tf.concat([v1,v2], axis=-1)
        jacobi_log_loss = tf.reduce_sum(logit_1, axis=-1) + tf.reduce_sum(logit_2, axis=-1)
        return z, jacobi_log_loss

    def backward(self, v1, v2):
        u2 = tf.multiply(v2 - self.t2(v1), tf.exp(-self.s2(v1)))
        u1 = tf.multiply(v1 - self.t1(u2), tf.exp(-self.s1(u2)))
        y = tf.concat([u1,u2], axis=-1)
        return y

    def jacobi_log_loss(self, u1, u2):
        v1 = tf.multiply(u1, tf.exp(self.s1(u2))) + self.t1(u2)
        logit = tf.reduce_sum(self.s1(u2), axis=-1) + tf.reduce_sum(self.s2(v1), axis=-1)
        return logit

    def jacobi_loss(self, u1, u2):
        v1 = tf.multiply(u1, tf.exp(self.s1(u2))) + self.t1(u2)
        logit = tf.reduce_sum(self.s1(u2), axis=-1) + tf.reduce_sum(self.s2(v1), axis=-1)
        return tf.exp(logit)


class RecurrentDecisionMaker(layers.Layer):
    def __init__(self, struct, name="RecurrentDecisionMaker", activation="relu", out_activation="softmax", **kwargs):
        super(RecurrentDecisionMaker, self).__init__(name=name, **kwargs)
        gru_layers = struct['decision_layers']
        out_dim = struct['num_frame']
        self.activation = struct['d_act']
        is_bidirectional = struct['is_bidirectional']

        self.layers = []

        for i in range(len(gru_layers)):
            ldim = gru_layers[i]
            if self.activation == "LeakyReLU":
                self.layers.append(get_gru(ldim, act=None, is_bidirectional=is_bidirectional, merge_mode="concat"))
            else:
                self.layers.append(get_gru(ldim, act=activation, is_bidirectional=is_bidirectional, merge_mode="concat"))

        self.post_layer = layers.Dense(out_dim, activation=out_activation)
        self.beta = struct['beta']

    def call(self, inputs, **kwargs):
        outputs = inputs
        for i in range(len(self.layers)):
            outputs = self.layers[i](outputs)
            if self.activation == "LeakyReLU":
                outputs = layers.LeakyReLU()(outputs)

        outputs = self.post_layer(outputs)
        return outputs



