# Fathom lite

*[Fathom](https://github.com/rdadolf/fathom) for the Impatient.*

Many Fathom users expressed difficulty in aquiring and using the data required.
Fathom-lite is a solution for users who are interested in performance analysis, not model accuracy.
Instead of training and testing on full datasets, Fathom-lite provides small subsets of data which can be used to measure the performance characteristics of deep neural networks.

## Workloads

Fathom-lite maintains all eight workloads from the main verison.

## Datasets

Each sample dataset contains 1000 inputs of the same shape and size as the originals.

Dataset   | Dimensions
----------|------------------
MNIST     | 1000 x 784
ImageNet  | 1000 x 224 x 224 x 3
WMT15     | 1000 x (bucket)
bAbI      | 1000 x 50 ? + 1000 x ?
ALE       | 1000 x 84 x 84 x 3


