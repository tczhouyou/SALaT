import os, inspect, sys

use_cpu = True
if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from tensorflow import keras
import tensorflow as tf
from SALAT import RecurrentNVP2DModel, RecurrentAttentionModel, train_model, use_model, constructGP
import numpy as np
from obs_avoid_envs import ObsExp, checkDocking, ObsSet2D
import matplotlib.pyplot as plt
from numpy.random import seed
seed(10)
tf.random.set_seed(10)

timesteps = 30
dim = 2

queries = np.loadtxt('data/docker_obs/train_queries.csv', delimiter=',')
n_data = np.shape(queries)[0]
tequeries = np.loadtxt('data/docker_obs/test_queries.csv', delimiter=',')
qattend = np.zeros(shape=(np.shape(queries)[0], 1))
queries = np.concatenate([queries, qattend], axis=-1)
trqueries = queries.copy()

shape = (n_data, timesteps, dim)
trtrajs = np.zeros(shape=shape)
for i in range(np.shape(queries)[0]):
    fname = 'data/docker_obs/processedTraj_' + str(i+1)
    traj = np.loadtxt(fname, delimiter=',')
    trtrajs[i,:,:] = traj


latent_dist = constructGP(timesteps, 2, 1, .1)
struct = {'s1_layers': [20, 1],
          't1_layers': [20, 1],
          's2_layers': [20, 1],
          't2_layers': [20, 1],
          's_act': 'tanh',
          't_act': 'tanh',
          'is_bidirectional': True}
model1 = RecurrentNVP2DModel(scope='docker1', struct=struct, ldim=2, latent_dist=latent_dist, timestamps=timesteps, reprod_ratio=0)
model1.compile(latent_opt=keras.optimizers.Adam(learning_rate=0.001))

struct = {'s1_layers': [20, 1],
          't1_layers': [20, 1],
          's2_layers': [20, 1],
          't2_layers': [20, 1],
          's_act': 'tanh',
          't_act': 'tanh',
          'is_bidirectional': True}
model2 = RecurrentNVP2DModel(scope='docker2', struct=struct, ldim=2, latent_dist=latent_dist, timestamps=timesteps, reprod_ratio=0)
model2.compile(latent_opt=keras.optimizers.Adam(learning_rate=0.001))

struct = {'s1_layers': [20, 1],
          't1_layers': [20, 1],
          's2_layers': [20, 1],
          't2_layers': [20, 1],
          's_act': 'tanh',
          't_act': 'tanh',
          'is_bidirectional': True}
model3 = RecurrentNVP2DModel(scope='obs', struct=struct, ldim=2, latent_dist=latent_dist, timestamps=timesteps, reprod_ratio=0)
model3.compile(latent_opt=keras.optimizers.Adam(learning_rate=0.001))

docker = ObsSet2D()
docker.add_openRec_obs([0,0], 0, 3, 4)
obs = ObsSet2D()
obs.add_circle_obs(origin=[0,0], radius=4)

fig, axes = plt.subplots(1,3)
latent_dist_to_sample = constructGP(timesteps, 1, 0.01, 0.01)
fig.suptitle('Recurrent RealNVP Generated Local Trajectories', fontsize=16)

qobj3 = trqueries[:,6:9]
latent_traj3, local_traj3, global_traj3 = train_model(model3, qobj3, trtrajs, train_epochs=10000,
                                                      ax=axes[0], latent_dist=latent_dist_to_sample, obj=obs,
                                                      model_file ='models/salat_docker_obs/obs', plot_title='obstacle')

plt.pause(0.1)

qobj1 = trqueries[:, 0:3]
latent_traj1, local_traj1, global_traj1 = train_model(model1, qobj1, trtrajs, train_epochs=10000,
                                                      ax=axes[1], latent_dist=latent_dist_to_sample, obj=docker,
                                                      model_file='models/salat_docker_obs/docker1', plot_title='start Docker')

plt.pause(0.1)

qobj2 = trqueries[:,3:6]
latent_traj2, local_traj2, global_traj2 = train_model(model2, qobj2, trtrajs, train_epochs=10000,
                                                      ax=axes[2], latent_dist=latent_dist_to_sample, obj=docker,
                                                      model_file ='models/salat_docker_obs/docker2', plot_title='end Docker')

plt.pause(0.1)


latent_traj = np.concatenate([latent_traj1, latent_traj2, latent_traj3], axis=-1)
local_traj = np.concatenate([local_traj1, local_traj2, local_traj3], axis=-1)
global_traj = np.concatenate([global_traj1, global_traj2, global_traj3], axis=-1)
latent_local_global_traj = np.concatenate([latent_traj, local_traj, global_traj], axis=-1)


struct={'decision_layers':[10],
          'num_frame': 3,
          'd_act': 'LeakyReLU',
          'beta': 1,
          'is_bidirectional': True}
amodel = RecurrentAttentionModel(nframe=3, struct=struct, single_ent_weight=100, total_ent_weight=1000, ldim=2,
                                 use_variance_weight=True, adjust_ratio=8, smooth_loss_weight=100)
#amodel.compile(opt=keras.optimizers.Adam(learning_rate=0.0003))
#amodel.fit(latent_local_global_traj, trtrajs, shuffle=False, verbose=1, epochs=20000, batch_size=np.shape(trqueries)[0])
amodel.load_weights('models/salat_docker_obs/attention')

### test ###
latent_traj1 = np.tile(np.expand_dims(latent_traj1[0,:,:], axis=0), reps=[np.shape(tequeries)[0], 1, 1])
latent_traj2 = np.tile(np.expand_dims(latent_traj2[0,:,:], axis=0), reps=[np.shape(tequeries)[0], 1, 1])
latent_traj3 = np.tile(np.expand_dims(latent_traj3[0,:,:], axis=0), reps=[np.shape(tequeries)[0], 1, 1])

test_global_traj1 = use_model(model1, tequeries[:,:3], latent_traj1)
test_global_traj2 = use_model(model2, tequeries[:,3:6], latent_traj2)
test_global_traj3 = use_model(model3, tequeries[:,6:9], latent_traj3)

test_latent_trajs = np.concatenate([latent_traj1,latent_traj2,latent_traj3], axis=-1)
test_global_trajs = np.concatenate([test_global_traj1,test_global_traj2, test_global_traj3], axis=-1)

ttrajs, outs = amodel.get_combined_trajs(test_global_trajs, test_latent_trajs, training=False)

fig = plt.figure()
fig.suptitle('Attention Model', fontsize=16)
plt.plot(outs[0, :, 0], 'r-')
plt.plot(outs[0, :, 1], 'b-')
plt.plot(outs[0, :, 2], 'g-')
plt.pause(0.1)

print(outs[0,:,:])
obsExp = ObsExp(exp_name="DockingWithObs")
envs = obsExp.get_envs(tequeries)
fig, axes = plt.subplots(4, 5)
fig.suptitle('First 20 Examples, Green: Success, Red: Failure', fontsize=16)

ri = 0
ci = 0
idx = np.arange(0, np.shape(tequeries)[0],5)
for id in range(20):
    ax = axes[ri, ci]
    i = idx[id]
    env = envs[i]
    env.plot(ax)
    ci = ci + 1
    if ci == 5:
        ri = ri + 1
        ci = 0

    tetraj = ttrajs[i,:,:]
    tetraj = np.expand_dims(tetraj, axis=0)
    cq = np.expand_dims(tequeries[i,:], axis=0)
    failure, _ = checkDocking(trajs=tetraj, tequeries=cq, samples=1000, orig_samples=timesteps, exp_name="DockingWithObs")
    if failure > 0:
        ax.plot(tetraj[0, :,0], tetraj[0, :,1], 'r-')
    else:
        ax.plot(tetraj[0, :,0], tetraj[0, :,1], 'g-')

    ax.set_xlim(-5, 35)
    ax.set_ylim(-5, 35)
    ax.set_aspect('equal')

plt.show()

failure, _ = checkDocking(trajs=ttrajs, tequeries=tequeries, samples=1000, orig_samples=timesteps, exp_name="DockingWithObs")
print('success rate: {}'.format(1 - failure/np.shape(tequeries)[0]))
