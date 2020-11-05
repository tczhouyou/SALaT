import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from RecurrentLocalInvertibleModel import RecurrentNVP2D, RecurrentDecisionMaker, constructGP

def get_transformation_mat(queries):
    transform_to_global = np.zeros(shape=(np.shape(queries)[0], 3, 3))
    transform_to_local = np.zeros(shape=(np.shape(queries)[0], 3, 3))

    for i in range(np.shape(queries)[0]):
        x0 = queries[i, 0]
        y0 = queries[i, 1]
        a = queries[i, 2]
        mat = np.array([[np.cos(a), -np.sin(a), x0], [np.sin(a), np.cos(a), y0], [0, 0, 1]])
        inv_mat = np.linalg.inv(mat)
        transform_to_global[i, :, :] = mat
        transform_to_local[i, :, :] = inv_mat

    return transform_to_global, transform_to_local

def get_local_trajs(transform_to_local, global_trajs):
    batch_size = np.shape(global_trajs)[0]
    timestamps = np.shape(global_trajs)[1]
    local_trajs = []
    for ni in range(batch_size):
        local_points = []
        for ti in range(timestamps):
            local_point = []
            tpoint = tf.concat([global_trajs[ni, ti, :], [1]], axis=-1)
            lpoint = tf.linalg.matvec(transform_to_local[ni, :, :], tpoint)
            local_point.append(lpoint[:-1])

            local_point = tf.concat(local_point, axis=-1)
            local_points.append(local_point)

        latent_traj = tf.stack(local_points, axis=0)
        local_trajs.append(latent_traj)

    local_trajs = tf.stack(local_trajs, axis=0)
    return local_trajs


def get_global_trajs(transform_to_global, local_trajs):
    global_trajs = []
    batch_size = np.shape(local_trajs)[0]
    timestamps = np.shape(local_trajs)[1]
    for ni in range(batch_size):
        global_points = []
        for ti in range(timestamps):
            global_point = []
            tpoint = tf.concat([local_trajs[ni, ti, :], [1]], axis=-1)
            gpoint = tf.linalg.matvec(transform_to_global[ni, :, :], tpoint)
            global_point.append(gpoint[:-1])
            global_point = tf.concat(global_point, axis=-1)
            global_points.append(global_point)

        global_traj = tf.stack(global_points, axis=0)
        global_trajs.append(global_traj)

    global_trajs = tf.stack(global_trajs, axis=0)
    return global_trajs


class RecurrentNVP2DModel(keras.Model):
    def __init__(self, struct, latent_dist, ldim=2, timestamps=10, scope='my_nvp', name="RecurrentNVP2DModel",
                 reprod_ratio=0, tmpId=0, **kwargs):
        super(RecurrentNVP2DModel, self).__init__(name=name, **kwargs)
        self.generators = RecurrentNVP2D(struct, scope=scope)
        self.batch_size = 10
        self.ldim = ldim
        self.xdim = int(ldim/2)
        self.latent_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.reprod_loss_tracker = keras.metrics.Mean(name="reprod_loss")

        self.timestamps = timestamps
        self.latent_dist = latent_dist
        self.reprod_ratio = reprod_ratio
        self.tmpId = tmpId

    def compile(self, latent_opt):
        super(RecurrentNVP2DModel, self).compile()
        self.latent_opt = latent_opt


    def sample_latent_trajs(self, num, latent_dist):
        latent_trajs = []
        for i in range(self.ldim):
            latent_traj = latent_dist.sample(sample_shape=num)
            latent_traj = tf.expand_dims(latent_traj, axis=-1)
            latent_trajs.append(latent_traj)

        latent_trajs = tf.concat(latent_trajs, axis=-1)
        return latent_trajs

    def get_latent_trajs(self, local_trajs):
        latent_trajs = []

        u1 = local_trajs[:, :, :self.xdim]
        u2 = local_trajs[:, :, self.xdim:]

        if len(np.shape(u1)) == 2:
            u1 = tf.expand_dims(u1, axis=-1)

        if len(np.shape(u2)) == 2:
            u2 = tf.expand_dims(u2, axis=-1)

        ltraj, jacobi_cost = self.generators.forward(u1, u2)
        latent_trajs.append(ltraj)
        latent_trajs = tf.concat(latent_trajs, axis=-1)
        return latent_trajs, jacobi_cost

    def __revert_local_trajs__(self, latent_trajs):
        local_trajs = []
        v1 = latent_trajs[:, :, :self.xdim]
        v2 = latent_trajs[:, :, self.xdim:]
        if len(np.shape(v1)) == 2:
            v1 = tf.expand_dims(v1, axis=-1)

        if len(np.shape(v2)) == 2:
            v2 = tf.expand_dims(v2, axis=-1)

        ltraj = self.generators.backward(v1, v2)
        local_trajs.append(ltraj)
        local_trajs = tf.concat(local_trajs, axis=-1)
        return local_trajs

    def get_latent_cost(self, trtrajs):
        latent_trajs, jacobi_cost = self.get_latent_trajs(trtrajs)

        z = latent_trajs

        log_pz = 0
        for i in range(self.ldim):
            log_pz += self.latent_dist.log_prob(z[:,:,i])

        log_pz = tf.expand_dims(log_pz, axis=-1)
        loss = -tf.reduce_mean(log_pz + jacobi_cost)

        mean_latent_traj = tf.reduce_mean(latent_trajs, axis=0, keepdims=True)
        local_trajs = self.__revert_local_trajs__(mean_latent_traj)
        template = trtrajs[self.tmpId,:,:]
        reprod_loss = tf.reduce_mean(tf.norm(template - local_trajs, axis=-1))

        return loss, reprod_loss

    def train_step(self, data):
        trqueries, localtrajs = data

        with tf.GradientTape(persistent=True) as tape:
            latent_loss, reprod_loss = self.get_latent_cost(localtrajs)
            if self.reprod_ratio != 0:
                loss = latent_loss + self.reprod_ratio * reprod_loss
            else:
                loss = latent_loss

        grad_latent = tape.gradient(loss, self.trainable_variables)
        self.latent_opt.apply_gradients(zip(grad_latent, self.trainable_variables))
        self.latent_loss_tracker.update_state(latent_loss)
        self.reprod_loss_tracker.update_state(reprod_loss)
        return {"latent_loss": self.latent_loss_tracker.result(), "reprod_loss": self.reprod_loss_tracker.result()}


class RecurrentAttentionModel(keras.Model):
    def __init__(self, nframe, struct, single_ent_weight=0, total_ent_weight=0, smooth_loss_weight=0, ldim=2, use_variance_weight=False,
                 adjust_ratio=1, name="RecurrentAttentionModel", **kwargs):
        super(RecurrentAttentionModel, self).__init__(name=name, **kwargs)
        self.nframe = nframe

        self.decider = RecurrentDecisionMaker(struct)
        self.ldim = ldim
        self.useCovWeight = use_variance_weight
        self.adjust_ratio = adjust_ratio
        self.single_ent_weight = single_ent_weight
        self.total_ent_weight = total_ent_weight
        self.smooth_loss_weight = smooth_loss_weight

        self.max_total_ent = 0
        ratio = np.divide(1, self.nframe)
        for i in range(self.nframe):
            self.max_total_ent += ratio * np.math.log(ratio)

        self.max_total_ent = - self.max_total_ent

        self.reprod_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.single_ent_tracker = keras.metrics.Mean(name="single_ent")
        self.total_ent_tracker = keras.metrics.Mean(name="total_ent")
        self.smooth_loss_tracker = keras.metrics.Mean(name="smooth_loss")

    def compile(self, opt):
        super(RecurrentAttentionModel, self).compile()
        self.opt = opt

    def get_combined_trajs(self, global_trajs, latent_trajs, training=True):
        outs = self.decider.call(latent_trajs)

        if not training and self.adjust_ratio != 0:
            outs = tf.nn.softmax(outs * self.adjust_ratio, axis=-1)

        weights = outs
        trajs = 0
        for fi in range(self.nframe):
            ws = weights[:,:,fi]
            ws = tf.expand_dims(ws, axis=-1)
            ws = tf.tile(ws, multiples=[1, 1, self.ldim])
            trajs += tf.multiply(global_trajs[:,:,fi*self.ldim:(fi+1)*self.ldim], ws)

        return trajs, outs

    def get_reprod_cost(self, trtrajs, local_trajs, global_trajs, latent_trajs):
        current_states = []
        for i in range(len(self.decider.layers)):
            current_states.append(None)

        trajs, outs = self.get_combined_trajs(global_trajs, latent_trajs)
        batch_size = trtrajs.get_shape().as_list()[0]
        # entropy cost
        single_ent = tf.negative(tf.reduce_sum(tf.multiply(outs, tf.math.log(outs + 1e-20)), axis=-1))
        total_ent = tf.reduce_mean(outs, axis=1)
        total_ent = tf.negative(tf.reduce_sum(tf.multiply(total_ent, tf.math.log(total_ent + 1e-20)), axis=-1))
        single_ent = tf.reduce_mean(single_ent)
        total_ent = self.max_total_ent - tf.reduce_mean(total_ent)

        cfvars = tfp.stats.variance(local_trajs)
        cfvars = tf.math.reduce_min(cfvars, axis=-1, keepdims=True)
        cfvarsmin = tf.math.reduce_min(cfvars, axis=0, keepdims=True)
        cfvarsmax = tf.math.reduce_max(cfvars, axis=0, keepdims=True)
        cweights = tf.math.divide_no_nan(cfvars - cfvarsmin, cfvarsmax - cfvarsmin)

        # reproduction cost
        if self.useCovWeight:
            vweights = tf.math.reciprocal_no_nan(cweights + 0.01) - 1
            vweights = tf.expand_dims(vweights, axis=0)
            vweights = tf.tile(vweights, multiples=(batch_size, 1, 1))
            reprod_loss = tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.norm(trajs - trtrajs, axis=-1, keepdims=True), vweights), axis=-1))
        else:
            reprod_loss = tf.reduce_mean(tf.reduce_mean(tf.norm(trajs - trtrajs, axis=-1), axis=-1))

        smooth_loss = tf.reduce_mean(tf.norm(trajs[:,:-1,:] - trajs[:,1:,:], axis=-1))
        return reprod_loss, single_ent, total_ent, smooth_loss

    def train_step(self, data):
        latent_local_global_trajs, trtrajs = data
        latent_trajs = latent_local_global_trajs[:, :, :self.nframe*self.ldim]
        local_trajs = latent_local_global_trajs[:, :, self.nframe*self.ldim:2*self.nframe*self.ldim]
        global_trajs = latent_local_global_trajs[:, :, 2*self.nframe*self.ldim:]

        with tf.GradientTape(persistent=True) as tape:
            reprod_loss, sent, tent, sloss = self.get_reprod_cost(trtrajs, local_trajs, global_trajs, latent_trajs)
            loss = reprod_loss + self.single_ent_weight * sent + self.total_ent_weight * tent + self.smooth_loss_weight * sloss

        grad_latent = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grad_latent, self.trainable_variables))
        self.reprod_loss_tracker.update_state(reprod_loss)
        self.single_ent_tracker.update_state(sent)
        self.total_ent_tracker.update_state(tent)
        self.smooth_loss_tracker.update_state(sloss)

        return {"reprod_loss": self.reprod_loss_tracker.result(),
                "single_ent": self.single_ent_tracker.result(),
                "total_ent": self.total_ent_tracker.result(),
                "smooth_loss": self.smooth_loss_tracker.result()}


def train_model(model, query, trtrajs, latent_dist=None, train_epochs=1000, ax=None, obj=None, model_file=None, save_file=None, plot_title='Local Frame'):
    batch_size = np.shape(query)[0]
    transform_to_global, transform_to_local = get_transformation_mat(query)
    local_traj = get_local_trajs(transform_to_local, trtrajs)

    if model_file is not None:
        print('load the model')
        model.load_weights(model_file)
    else:
        model.fit(query, local_traj, shuffle=False, verbose=1, epochs=train_epochs, batch_size=np.shape(query)[0])
        if save_file is not None:
            model.save_weights(save_file)

    mean_local_traj = tf.reduce_mean(local_traj, axis=0, keepdims=True)
    latent_traj, _ = model.get_latent_trajs(local_traj)
    mean_latent_traj = tf.reduce_mean(latent_traj, axis=0, keepdims=True)
    mean_latent_traj = tf.tile(mean_latent_traj, multiples=[batch_size, 1, 1])

    gen_local_traj = model.__revert_local_trajs__(mean_latent_traj)
    gen_global_traj = get_global_trajs(transform_to_global, gen_local_traj)
    if ax is not None and latent_dist is not None:
        sampled_latent_traj = mean_latent_traj + model.sample_latent_trajs(batch_size, latent_dist)
        sampled_local_traj = model.__revert_local_trajs__(sampled_latent_traj)
        if obj is not None:
            obj.plot(ax)

        for i in range(20):
            ax.plot(sampled_local_traj[i, :, 0], sampled_local_traj[i, :, 1])

        gmtraj, = ax.plot(gen_local_traj[0, :, 0], gen_local_traj[0, :, 1], 'r-')
        mtraj, = ax.plot(mean_local_traj[0, :, 0], mean_local_traj[0,:,1], 'k-')
        ax.legend((mtraj, gmtraj), ('average', 'mean (RealNVP)'))
        ax.set_title(plot_title)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')

    return mean_latent_traj, local_traj, gen_global_traj


def use_model(model, tequeries, latent_traj):
    transform_to_global, transform_to_local = get_transformation_mat(tequeries)
    local_traj = model.__revert_local_trajs__(latent_traj)
    global_traj = get_global_trajs(transform_to_global, local_traj)
    return global_traj
