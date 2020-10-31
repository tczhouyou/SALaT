import os, inspect, sys

use_cpu = True
if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from tensorflow import keras
import tensorflow as tf
from SALAT import RecurrentNVP2DModel, train_model, use_model, RecurrentAttentionModel, constructGP
import numpy as np
from obs_avoid_envs import ObsExp, checkDocking
import matplotlib.pyplot as plt
from numpy.random import seed
seed(10)
tf.random.set_seed(10)

n_data = 20
timesteps = 30
dim = 2
shape = (n_data, timesteps, dim)
trtrajs = np.zeros(shape=shape)
queries = np.loadtxt('data/docker/train_queries.csv', delimiter=',')
queries = queries[:n_data,:]
tequeries = np.loadtxt('data/docker/test_queries.csv', delimiter=',')
trqueries = queries.copy()

for i in range(n_data):
    fname = 'data/docker/processedTraj_' + str(i+1)
    traj = np.loadtxt(fname, delimiter=',')
    trtrajs[i,:,:] = traj


latent_dist = constructGP(timesteps, 2, 1, 0.1)
struct = {'s1_layers': [20, 1],
          't1_layers': [20, 1],
          's2_layers': [20, 1],
          't2_layers': [20, 1],
          's_act': 'LeakyReLU',
          't_act': 'LeakyReLU',
          'is_bidirectional': True}
model1 = RecurrentNVP2DModel(scope='obj1', struct=struct, ldim=2, latent_dist=latent_dist, timestamps=timesteps)
model1.compile(latent_opt=keras.optimizers.Adam(learning_rate=0.003))
model2 = RecurrentNVP2DModel(scope='obj2', struct=struct, ldim=2, latent_dist=latent_dist, timestamps=timesteps)
model2.compile(latent_opt=keras.optimizers.Adam(learning_rate=0.003))

qobj1 = trqueries[:, 0:3]
latent_traj1, local_traj1, global_traj1 = train_model(model1, qobj1, trtrajs, train_epochs=2000,
                                                      model_file ='models/salat_docker/docker1')

qobj2 = trqueries[:,3:6]
latent_traj2, local_traj2, global_traj2 = train_model(model2, qobj2, trtrajs, train_epochs=2000,
                                                      model_file ='models/salat_docker/docker2')


latent_traj = np.concatenate([latent_traj1, latent_traj2], axis=-1)
local_traj = np.concatenate([local_traj1, local_traj2], axis=-1)
global_traj = np.concatenate([global_traj1, global_traj2], axis=-1)
latent_local_global_traj = np.concatenate([latent_traj, local_traj, global_traj], axis=-1)

struct={'decision_layers':[20],
          'num_frame': 2,
          'd_act': 'LeakyReLU',
          'beta': 1,
          'is_bidirectional': True}
amodel = RecurrentAttentionModel(nframe=2, struct=struct, single_ent_weight=1, total_ent_weight=10, ldim=2,
                                 use_variance_weight=True, adjust_ratio=20, smooth_loss_weight=25)
#amodel.compile(opt=keras.optimizers.Adam(learning_rate=0.003))
#amodel.fit(latent_local_global_traj, trtrajs, shuffle=False, verbose=1, epochs=3000, batch_size=np.shape(trqueries)[0])
amodel.load_weights('models/salat_docker/attention')

### test ###
latent_traj1 = np.tile(np.expand_dims(latent_traj1[0,:,:], axis=0), reps=[np.shape(tequeries)[0], 1, 1])
latent_traj2 = np.tile(np.expand_dims(latent_traj2[0,:,:], axis=0), reps=[np.shape(tequeries)[0], 1, 1])

test_global_traj1 = use_model(model1, tequeries[:,:3], latent_traj1)
test_global_traj2 = use_model(model2, tequeries[:,3:6], latent_traj2)

test_latent_trajs = np.concatenate([latent_traj1,latent_traj2], axis=-1)
test_global_trajs = np.concatenate([test_global_traj1,test_global_traj2], axis=-1)

ttrajs, outs = amodel.get_combined_trajs(test_global_trajs, test_latent_trajs)

obsExp = ObsExp(exp_name="Docking")
envs = obsExp.get_envs(tequeries)
fig, axes = plt.subplots(4, 5)
ri = 0
ci = 0
for i in range(20):
    ax = axes[ri, ci]
    env = envs[i]
    env.plot(ax)
    ci = ci + 1
    if ci == 5:
        ri = ri + 1
        ci = 0

    tetraj = ttrajs[i,:,:]
    tetraj = np.expand_dims(tetraj, axis=0)
    cq = np.expand_dims(tequeries[i,:], axis=0)
    failure, _ = checkDocking(trajs=tetraj, tequeries=cq, samples=1000, orig_samples=timesteps, exp_name="Docking")
    if failure > 0:
        ax.plot(tetraj[0, :,0], tetraj[0, :,1], 'r-')
    else:
        ax.plot(tetraj[0, :,0], tetraj[0, :,1], 'g-')

    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 25)
    ax.set_aspect('equal')

plt.show()

failure, _ = checkDocking(trajs=ttrajs, tequeries=tequeries, samples=1000, orig_samples=timesteps, exp_name="Docking")
print('success rate: {}'.format(1 - failure/np.shape(tequeries)[0]))
