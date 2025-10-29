import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
import numpy as np

_EPSILON = 1e-08


def log(x):
    return tf.math.log(x + _EPSILON)


def div(x, y):
    return tf.math.divide(x, (y + _EPSILON))


def create_fcnet(x, num_layers, hidden_dim, activation, keep_prob, reg):
    """Replaces utils.create_FCNet"""
    out = x
    for _ in range(num_layers):
        out = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=reg)(
            out
        )
        out = layers.Dropout(1.0 - keep_prob)(out)
    return out


class DeepHit(Model):
    def __init__(self, input_dims, network_settings):
        super(DeepHit, self).__init__()

        # Input dimensions
        self.x_dim = input_dims["x_dim"]
        self.num_Event = input_dims["num_Event"]
        self.num_Category = input_dims["num_Category"]

        # Network hyperparameters
        self.h_dim_shared = network_settings["h_dim_shared"]
        self.h_dim_CS = network_settings["h_dim_CS"]
        self.num_layers_shared = network_settings["num_layers_shared"]
        self.num_layers_CS = network_settings["num_layers_CS"]
        self.active_fn = network_settings["active_fn"]
        self.keep_prob = network_settings["keep_prob"]

        self.reg_W = regularizers.l2(1e-4)
        self.reg_W_out = regularizers.l1(1e-4)

        self.build_network()

    def build_network(self):
        # Shared network
        self.shared_layers = [
            layers.Dense(
                self.h_dim_shared,
                activation=self.active_fn,
                kernel_regularizer=self.reg_W,
            )
            for _ in range(self.num_layers_shared)
        ]

        # Cause-specific subnetworks
        self.cs_nets = []
        for _ in range(self.num_Event):
            cs_layers = [
                layers.Dense(
                    self.h_dim_CS,
                    activation=self.active_fn,
                    kernel_regularizer=self.reg_W,
                )
                for _ in range(self.num_layers_CS)
            ]
            self.cs_nets.append(cs_layers)

        # Output layer
        self.out_layer = layers.Dense(
            self.num_Event * self.num_Category,
            activation="softmax",
            kernel_regularizer=self.reg_W_out,
        )

    def call(self, x, training=False):
        # Shared network
        h = x
        for layer in self.shared_layers:
            h = layer(h, training=training)
            h = layers.Dropout(1.0 - self.keep_prob)(h, training=training)

        shared_out = tf.concat([x, h], axis=1)

        # Cause-specific networks
        cs_outs = []
        for cs_layers in self.cs_nets:
            h_cs = shared_out
            for layer in cs_layers:
                h_cs = layer(h_cs, training=training)
            cs_outs.append(h_cs)

        out = tf.concat(cs_outs, axis=1)
        out = layers.Dropout(1.0 - self.keep_prob)(out, training=training)
        out = self.out_layer(out)
        out = tf.reshape(out, [-1, self.num_Event, self.num_Category])
        return out

    # === LOSS FUNCTIONS ===
    def loss_log_likelihood(self, k, t, fc_mask1, fc_mask2, out):
        # Ensure correct dtype and shape
        k = tf.cast(k, tf.float32)
        I_1 = tf.sign(k)  # 1 for uncensored, 0 for censored

        tmp1 = tf.reduce_sum(fc_mask1 * out, axis=[1, 2], keepdims=True)
        tmp1 = tf.clip_by_value(tmp1, 1e-8, 1.0)  # prevent log(0)
        tmp1 = I_1 * tf.math.log(tmp1)

        tmp2 = tf.reduce_sum(fc_mask2 * out, axis=[1, 2], keepdims=True)
        tmp2 = tf.clip_by_value(tmp2, 1e-8, 1.0)
        tmp2 = (1.0 - I_1) * tf.math.log(tmp2)

        loss = -tf.reduce_mean(tmp1 + tmp2)
        return loss

    def loss_ranking(self, k, t, fc_mask2, out):
        sigma1 = tf.constant(0.1, dtype=tf.float32)
        num_Event = self.num_Event
        num_Category = self.num_Category

        eta = []
        for e in range(num_Event):
            one_vector = tf.ones_like(t, dtype=tf.float32)  # ✅ ensure float32
            I_2 = tf.cast(tf.equal(k, e + 1), dtype=tf.float32)
            I_2 = tf.linalg.diag(tf.squeeze(I_2))

            tmp_e = tf.reshape(out[:, e, :], [-1, num_Category])
            R = tf.matmul(tmp_e, tf.transpose(fc_mask2))

            diag_R = tf.expand_dims(tf.linalg.diag_part(R), axis=1)
            R = tf.transpose(one_vector * tf.transpose(diag_R) - R)  # ✅ fixed dtype

            T = tf.nn.relu(
                tf.sign(
                    tf.matmul(one_vector, tf.transpose(tf.cast(t, tf.float32)))
                    - tf.matmul(tf.cast(t, tf.float32), tf.transpose(one_vector))
                )
            )
            T = tf.matmul(I_2, T)

            tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), axis=1, keepdims=True)
            eta.append(tmp_eta)

        eta = tf.stack(eta, axis=1)
        eta = tf.reduce_mean(eta, axis=1, keepdims=True)
        loss = tf.reduce_sum(eta)
        return loss

    def loss_calibration(self, k, t, fc_mask2, out):
        eta = []
        for e in range(self.num_Event):
            I_2 = tf.cast(tf.equal(k, e + 1), tf.float32)
            tmp_e = out[:, e, :]
            r = tf.reduce_sum(tmp_e * fc_mask2, axis=0)
            tmp_eta = tf.reduce_mean(tf.square(r - I_2), axis=0, keepdims=True)
            eta.append(tmp_eta)
        eta = tf.concat(eta, axis=1)
        eta = tf.reduce_mean(eta, axis=1, keepdims=True)
        return tf.reduce_sum(eta)

    def compute_loss(self, x, k, t, fc_mask1, fc_mask2, alpha, beta, gamma):
        out = self.call(x, training=True)
        loss1 = self.loss_log_likelihood(k, t, fc_mask1, fc_mask2, out)
        loss2 = self.loss_ranking(k, t, fc_mask2, out)
        loss3 = self.loss_calibration(k, t, fc_mask2, out)
        total_loss = alpha * loss1 + beta * loss2 + gamma * loss3
        total_loss += tf.reduce_sum(self.losses)  # add regularization losses
        return total_loss, (loss1, loss2, loss3)
