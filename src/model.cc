/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <iostream>
#include <iomanip>  // std::setprecision
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

real Model::State::getLoss() const {
  return lossValue_ / nexamples_;
}

void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, state);

  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue);

  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}

real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}


WeightsModel::WeightsModel(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    int32_t seed)
    : Model(wi, wo, args, seed),
      weights(2 * args->ws),
      weights_probs(2 * args->ws),
      weights_grad_(2 * args->ws),
      wsz_(2 * args->ws) {
  weights.ones();
}

void WeightsModel::exitWithError() {
  std::cerr << "This mode is not supported" << std::endl;
  exit(EXIT_FAILURE);
}

void WeightsModel::update(const std::vector<int32_t>& input,
                          int32_t target, real lr,
                          int32_t winOffset) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  computeHidden(input, hidden_);
  if (args_->loss == loss_name::ns) {
    exitWithError();
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    exitWithError();
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    real sampleLoss = softmax(target, lr, winOffset);
    loss_ += sampleLoss;
    weights.addVector(weights_grad_, -lr * sampleLoss);
  }
  nexamples_ += 1;

  // if (args_->model == model_name::sup) {
  //   grad_.mul(1.0 / input.size());
  // }
  // for (auto it = input.cbegin(); it != input.cend(); ++it) {
  //   wi_->addRow(grad_, *it, 1.0);
  // }
}

real WeightsModel::softmax(int32_t target, real lr, int32_t offset) {
  weights_grad_.zero();
  weights_probs.zero();
  // Apply weights
  real max = weights[0], z = 0.0;
  for (int32_t i = 0; i < wsz_; i++)
    max = std::max(weights[i], max);
  for (int32_t i = 0; i < wsz_; i++) {
    weights_probs[i] = exp(weights[i] - max);
    z += weights_probs[i];
  }
  for (int32_t i = 0; i < wsz_; i++) {
    weights_probs[i] /= z;
    // weights_probs[i] = std::max(weights_probs[i], (real)0.001);
    std::cerr << std::setprecision(2) << std::setw(5);
    std::cerr << weights_probs[i] << " ";
  }
  // Now weights_grad_ holds softmax()
  for (int32_t i = 0; i < wsz_; i++)
    weights_grad_[i] = ((i == offset) ? (1.0/z - weights_probs[offset]) : weights_probs[i]);
  // Now weights_grad_ hold the diff wrt weights. Before update it needs to be
  // multiplied by log(loss)

  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha * weights_probs[offset]);
  }
  std::cerr << std::setprecision(5) << std::setw(7) << "\t" << -log(output_[target]) * weights_probs[offset] << std::endl;
  return -log(output_[target]) * weights_probs[offset];
}

}
