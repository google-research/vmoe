# Scaling Vision with Sparse Mixture of Experts

**You can find the implementation of our V-MoE model under the `vmoe/nn/`
directory. We are currently working on open sourcing all the remaining pieces
to train from scratch and evaluate existing checkpoints. Stay tuned.**

This repository contains the code for training and fine-tuning Sparse MoE models
for vision (V-MoE) on ImageNet-21k, reproducing the results presented in
the paper:

- [Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974), by
  Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann,
  Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby.

The project uses, which you'll have to install:

- [JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax)
- [Tensorflow Datasets](https://www.tensorflow.org/datasets)
- [Vision Transformer](https://github.com/google-research/vision_transformer)


## Disclaimers

This is not an officially supported Google product.
