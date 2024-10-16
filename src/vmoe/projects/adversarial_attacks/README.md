# On the Adversarial Robustness of Mixture of Experts

This directory contains the code used in the paper
[On the Adversarial Robustness of Mixture of Experts](https://arxiv.org/abs/2210.10253),
by Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi,
and Srinadh Bhojanapalli.

The experiments in the paper attack a JFT-300M model and the same model
fine-tuned on ILSVRC2012 (a.k.a. ImageNet-1k). Because JFT-300M is a proprietary
dataset, these models are not publicly available. To illustrate how to use our
code, we have provided a
[config file](https://github.com/google-research/vmoe/blob/main/vmoe/projects/adversarial_attacks/configs/ilsvrc2012.py)
to attack a ILSVRC2012 model pre-trained on the public ImageNet-21k dataset.
