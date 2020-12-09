# SALaT

## Introduction
We created a model to meet two challenges in robot Learning from Demonstrations (LfD): multiple modes and extrapolation.

The model is called _**S**hift **A**ttention **La**tent **T**ransformation_ (**SALaT**). 

It consists of _latent transformations_ and an _attention model_. 

We use _real valued non volume-preserving mapping_ (RealNVP)[[1]](#1) to model the mappings and gated recurrent units for the attention model.

The code provides supplementary materials for the paper "_Learning to Shift Attention for Motion Generation_"

## Setup
The code requires **tensorflow 2.2 or later**. (**Note: Because of recent update of tensorflow-probability, the newest tensorflow-probability (0.12) requires tensorflow 2.4. We update the setup.py with tensorflow-probability==0.11**)

Install the tensorflow environment (https://www.tensorflow.org/install/pip) by following steps:

Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:

`$ python3 -m venv --system-site-packages ./venv`

Activate the virtual environment using a shell-specific command:

`$ source ./venv/bin/activate` 

When the virtual environment is active, your shell prompt is prefixed with (venv).
Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip:

`(venv) $ pip install --upgrade pip`

`(venv) $ pip install --upgrade tensorflow`


For setting up other necessary packages, run:

`(venv) $ python setup.py install`

## Experiments

The repo contains the simulated experiments described in Sec.IV of the paper. 

With the pre-trained models, we can reproduce the result listed in TABLE 1 of the paper.

For the docker experiment, run:
`(venv) $ python testSALATOnDocker.py`

For the docker-obstacle experiment, run:
`(venv) $ python testSALATOnDockerObs.py`

For the docker-obstacle-tunnel experiment, run:
`(venv) $ python testSALATOnDockerObsTunnel.py`

## References
<a id="1">[1]</a> 
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio. "Density estimation using Real NVP" (https://arxiv.org/abs/1605.08803)

