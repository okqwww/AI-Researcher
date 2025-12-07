Project: VQVAE with Rotation-Rescale Gradient Propagation (RR-VQVAE)

This project implements an innovative VQ-VAE variant that preserves gradient angles across the non-differentiable quantization by using a Householder-based rotation and rescaling transformation. It also includes EMA codebook updates and codebook usage monitoring to mitigate collapse.

Structure
- data/: dataset cache
- data_processing/: CIFAR-10 pipeline
- model/: encoder/decoder, quantizer with RR
- training/: training loop
- testing/: reconstruction + FID evaluation with CIFAR stats
- run_training_testing.py: full pipeline (2 epochs)

Run
python -m project.run_training_testing

Notes
- Rotation/rescaling treated as constants w.r.t gradients (stop-gradient) as per the innovation.
- Householder transform ensures efficient alignment; backward applies orthogonal mapping to gradients.
- EMA codebook update follows VQ-VAE v2 style.
