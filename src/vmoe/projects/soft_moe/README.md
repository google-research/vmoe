# From Sparse to Soft Mixture of Experts

This folder contains the implementation of Soft MoE, presented in the paper:

- [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951),
  by Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby.

We provide the config files used to run some of the experiments reported in the
paper.

Notice that most experiments either train on JFT-4B, a proprietary dataset,
or use models pre-trained on it, thus we cannot release any of the checkpoints.
We have released the config file used to train on JFT-4B from scratch, for
reference.

We have also included a config file to pretrain on LAION-400M, which is a
publicly available dataset. This can be used replicate the experiments that we
conducted on this dataset and are reported in the paper. Note, however, that we
are not planning on releasing any checkpoint trained in this dataset.
