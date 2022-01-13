# Scaling Vision with Sparse Mixture of Experts

This repository contains the code for training and fine-tuning Sparse MoE models
for vision (V-MoE) on ImageNet-21k, reproducing the results presented in
the paper:

- [Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974), by
  Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann,
  Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby.


We will soon provide a colab analysing one of the models that we have released,
as well as "config" files to train from scratch and fine-tune checkpoints. Stay
tuned.

## Installation

Simply clone this repository.

The file `requirements.txt` contains the requirements that can be installed
via PyPi. However, we recommend installing `jax`, `flax` and `optax`
directly from GitHub, since we use some of the latest features that are not part
of any release yet.

In addition, you also have to clone the
[Vision Transformer](https://github.com/google-research/vision_transformer)
repository, since we use some parts of it.

If you want to use RandAugment to train models (which we recommend if you train
on ImageNet-21k or ILSVRC2012 from scratch), you must also clone the
[Cloud TPU](https://github.com/tensorflow/tpu) repository, and name it
`cloud_tpu`.

## Checkpoints

We release the checkpoints containing the weights of some models that we trained
on ImageNet (either ILSVRC2012 or ImageNet-21k). All checkpoints contain an
index file (with `.index` extension) and one or multiple data files (
with extension `.data-nnnnn-of-NNNNN`, called *shards*). In the following
list, we indicate *only the prefix* of each checkpoint.
We recommend using [gsutil](https://cloud.google.com/storage/docs/gsutil) to
obtain the full list of files, download them, etc.

- V-MoE S/32, 8 experts on the last two odd blocks, trained from scratch on
  ILSVRC2012 with RandAugment: `gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_medium`.
- V-MoE B/16, 8 experts on every odd block, trained from scratch on ImageNet-21k
  with RandAugment: `gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong`.
  - Fine-tuned on ILSVRC2012:
    `gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012`

## Disclaimers

This is not an officially supported Google product.
