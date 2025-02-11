#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void layer_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
               torch::Tensor& bias, double epsilon);

void fused_add_layer_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, torch::Tensor& bias, double epsilon);

void ada_layer_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& gamma,
                   torch::Tensor& beta, double epsilon);