// ----------------------------------------------------------------------------
// Portions of this file have been modified by Ash in 2025.
// Original source: OpenMMLab/mmcv (Apache License 2.0).
// Modifications were made to support the working environment of TV3S.
// ----------------------------------------------------------------------------
#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <THC/THCAtomics.cuh>

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
