#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/all.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Kizuna custom ops

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Apply Layer Normalization to the input tensor.
  ops.def(
      "layer_norm(Tensor! out, Tensor input, Tensor weight, Tensor bias, float epsilon) -> ()");
  ops.impl("layer_norm", torch::kCUDA, &layer_norm);

  // In-place fused Add and Layer Normalization.
  ops.def(
      "fused_add_layer_norm(Tensor! input, Tensor! residual, Tensor weight, Tensor bias, float epsilon) -> ()");
  ops.impl("fused_add_layer_norm", torch::kCUDA, &fused_add_layer_norm);

  // Apply AdaLayer Normalization to the input tensor.
  ops.def(
      "ada_layer_norm(Tensor! out, Tensor input, Tensor gamma, Tensor beta, float epsilon) -> ()");
  ops.impl("ada_layer_norm", torch::kCUDA, &ada_layer_norm);

  // Apply AdaInstance Normalization to the input tensor.
  ops.def(
      "ada_instance_norm(Tensor! out, Tensor input, Tensor gamma, Tensor beta, float epsilon) -> ()");
  ops.impl("ada_instance_norm", torch::kCUDA, &ada_instance_norm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)