# SALaT

We created a model that lets robot learn how to shift attention from one object to others to accomplish the task. The model consists of latent transformations and an attention model. We use real-valued volume-preserving mapping (RealNVP) to model the mappings and use gated recurrent units for the attention model.

The code is associated with the paper "Learning to Shift Attention for Motion Generation"

The code was tested with tensorflow 2.0. Install the tensorflow environment by following https://www.tensorflow.org/install/pip

For setting up other necessary packages, run:
`python setup.py install`

For the docker experiment, run:
`python testSALATOnDocker.py`

For the docker-obstacle experiment, run:
`python testSALATOnDockerObs.py`

For the docker-obstacle-tunnel experiment, run:
`python testSALATOnDockerObsTunnel.py`