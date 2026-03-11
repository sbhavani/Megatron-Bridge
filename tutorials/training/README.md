# Training Tutorials

This directory contains hands-on Megatron Bridge training notebooks.

## Available Tutorials

- [Ultra-Scale Parallelism Runbook](ultra_scale_parallelism.ipynb): a practical multi-GPU walkthrough of data parallelism, tensor parallelism, pipeline parallelism, context parallelism, activation recomputation, communication overlap, checkpointing, and resume workflows using Megatron Bridge recipes.
- [Reduced Precision Training](reduced_precision_training.ipynb): a focused notebook on mixed-precision training behavior.

## Notes

- The ultra-scale runbook uses Qwen3 recipes as concrete examples, but the notebook is intended as a general guide to Megatron parallelism and training workflow design.
- The notebook uses mock data and short runs so you can validate topology and runtime behavior before plugging in a real dataset.
