#!/bin/bash
#
# Install all necessary packages to run experiments on GCE.
#
# Before running this script, you must have created a GCE instance.
# For instance, to create a VM with a TPUv3-32 Pod slice run this command:
#
# $ gcloud alpha compute tpus tpu-vm create instance-name --zone=europe-west4-a --accelerator-type=v3-32 --version=v2-alpha

# Run all commands relative to $HOME.
cd "$HOME";
mkdir -p "$HOME/src";
export PATH="$PATH:$HOME/.bin";

# Install virtualenv and create a new virtual environment.
pip3 install -q virtualenv;
[[ -d env ]] || python3 -m virtualenv env;
source env/bin/activate;

if ( which nvidia-smi &> /dev/null ); then
  # This assumes CUDA 11 and cuDNN 8.2.
  # Check https://github.com/google/jax#pip-installation-gpu-cuda for alternatives.
  pip install -q 'jax[cuda]' -f https://storage.googleapis.com/jax-releases/jax_releases.html;
else
  # Since pjit does not work on CPUs, if nvidia-smi is not found, we assume that
  # we will run on TPUs.
  pip install -q 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
fi;

# Upgrade pip.
python3 -m pip install -q --upgrade pip;
# Upgrade the following packages from GIT, since we use some features not part
# of any release yet.
pip install -q --upgrade git+https://github.com/google/jax.git;
pip install -q --upgrade git+https://github.com/google/flax.git;
pip install -q --upgrade git+https://github.com/google/CommonLoopUtils.git;
# Install the rest of necessary packages from PyPi.
pip install -q absl-py cachetools chex einops ml_collections numpy optax pandas scipy tensorflow-cpu tfds-nightly;
# Download vision_transformer codebase.
[[ -d src/vision_transformer ]] || git clone https://github.com/google-research/vision_transformer.git src/vision_transformer;
# Download vmoe codebase.
[[ -d src/vmoe ]] || git clone https://github.com/google-research/vmoe.git src/vmoe;
# This should print the total number of devices in your GCE instance.
# For instance, if you used TPUv3-32 this should print 32.
python -c 'import jax; print(jax.device_count())';
