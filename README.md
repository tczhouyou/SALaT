# SALaT

## Introduction
We created a model to meet two challenges in robot Learning from Demonstrations (LfD): multiple modes and extrapolation.

The model is called _**S**hift **A**ttention **La**tent **T**ransformation_ (**SALaT**). 

It consists of _latent transformations_ and an _attention model_. 

We use _real valued non volume-preserving mapping_ (RealNVP)[[1]](#1) to model the mappings and gated recurrent units for the attention model.

The code provides supplementary materials for the paper "_Learning to Shift Attention for Motion Generation_"

## Setup
The code was tested with tensorflow 2.0. 
Install the tensorflow environment by following https://www.tensorflow.org/install/pip

For setting up other necessary packages, run:
`python setup.py install`

## Experiments

The repo contains the simulated experiments described in Sec.IV of the paper. 

With the pre-trained models, we can reproduce the result listed in TABLE 1 of the paper.

For the docker experiment, run:
`python testSALATOnDocker.py`

For the docker-obstacle experiment, run:
`python testSALATOnDockerObs.py`

For the docker-obstacle-tunnel experiment, run:
`python testSALATOnDockerObsTunnel.py`

## References
<a id="1">[1]</a> 
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio. "Density estimation using Real NVP" (https://arxiv.org/abs/1605.08803)

