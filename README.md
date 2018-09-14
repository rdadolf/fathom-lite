# Fathom lite

*[Fathom](https://github.com/rdadolf/fathom) for the Impatient.*

[![Build Status](https://travis-ci.org/rdadolf/fathom-lite.svg?branch=master)](https://travis-ci.org/rdadolf/fathom-lite)

Many Fathom users expressed difficulty in aquiring and using the large datasets required.
Fathom-lite is a solution for users who are interested in performance analysis, not model accuracy.
Instead of training and testing on full datasets, Fathom-lite provides small subsets of data which can be used to measure the performance characteristics of deep neural networks.
Most deep learning models require significant amounts of data to learn complex functions, but the performance behavior of individual training or inference batches is often very similar.

## Workloads

Fathom-lite maintains all eight workloads from the main verison.

## Datasets

Each sample dataset contains 1000 inputs of the same shape and size as the originals.

Dataset   | Dimensions
----------|------------------
MNIST     | 1000 x 784
ImageNet  | 1000 x 224 x 224 x 3
bAbI      | 1000 x 10 x 6 + 1000 x 6
WMT15     | 1000 x (bucket)
ALE       | 1000 x 84 x 84 x 3

## FAQ

*Is it working?*

You can check with the built-in tests. Make sure you have a few GB of RAM available for the larger models:

```
pytest
```

*What do I do with this?*

Analyzing model behavior can be tricky. Might we recommend [Dnnamo](https://github.com/rdadolf/dnnamo)?

