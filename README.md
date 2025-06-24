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

We also provide checkpoints, a notebook, and a config for Efficient Ensemble of
Experts (E<sup>3</sup>), presented in the paper:

- [Sparse MoEs meet Efficient Ensembles](https://openreview.net/forum?id=i0ZM36d2qU&noteId=Rtlnlx5PzY), by
  James Urquhart Allingham, Florian Wenzel, Zelda E Mariet, Basil Mustafa,
  Joan Puigcerver, Neil Houlsby, Ghassen Jerfel, Vincent Fortuin,
  Balaji Lakshminarayanan, Jasper Snoek, Dustin Tran,Carlos Riquelme Ruiz,
  and Rodolphe Jenatton.

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
  ILSVRC2012 with RandAugment for 300 epochs:
  `gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1`.
  - Fine-tuned on ILSVRC2012 with a resolution of 384 pixels:
    `gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012`
- V-MoE S/32, 8 experts on the last two odd blocks, trained from scratch on
  ILSVRC2012 with RandAugment for 1000 epochs:
  `gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_medium`.
- V-MoE B/16, 8 experts on every odd block, trained from scratch on ImageNet-21k
  with RandAugment: `gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong`.
  - Fine-tuned on ILSVRC2012 with a resolution of 384 pixels:
    `gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012`
- E<sup>3</sup> S/32, 8 experts on the last two odd blocks, with two ensemble
  members (i.e., the 8 experts are partitioned into two groups), trained from
  scratch on ILSVRC2012 with RandAugment for 300 epochs:
  `gs://vmoe_checkpoints/eee_s32_last2_ilsvrc2012`
  - Fine-tuned on CIFAR100:
    `gs://vmoe_checkpoints/eee_s32_last2_ilsvrc2012_ft_cifar100`

## Disclaimers

This is not an officially supported Google product.
