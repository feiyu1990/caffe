#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/piecewise_ranking_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <math.h>       /* fabs */



namespace caffe {

template <typename Dtype>
void PRankingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[3]->channels(), 1);
  CHECK_EQ(bottom[3]->height(), 1);
  CHECK_EQ(bottom[3]->width(), 1);
  if (bottom.size() > 4) {
    CHECK_EQ(bottom[4]->channels(), 1);
    CHECK_EQ(bottom[4]->height(), 1);
    CHECK_EQ(bottom[4]->width(), 1);
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_ground_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_pred_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void PRankingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
    count,
    bottom[0]->cpu_data(),  // P_a
    bottom[1]->cpu_data(),  // P_b
    diff_.mutable_cpu_data());  // P_ai - P_bi
  caffe_sub(
    count,
    bottom[2]->cpu_data(),  // D_a
    bottom[3]->cpu_data(),  // D_b
    diff_ground_.mutable_cpu_data());  // D_ai - D_bi

  // margin1: similar margin; margin2: different margin
  Dtype margin1 = this->layer_param_.piecewise_ranking_loss_param().margin1();
  Dtype margin2 = this->layer_param_.piecewise_ranking_loss_param().margin2();
  bool norm2 = this->layer_param_.piecewise_ranking_loss_param().norml2();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (bottom.size() > 4 && bottom[4]->cpu_data()[i] < 0.5) {continue;}
    if (diff_ground_.cpu_data()[i] > 0) {  //if first is larger (G_a - G_b)
      dist_pred_.mutable_cpu_data()[i] = diff_.cpu_data()[i];
    }
    else { //else: (G_b - G_a)
      dist_pred_.mutable_cpu_data()[i] = - diff_.cpu_data()[i];
    }
    Dtype label_difference(fabs(diff_ground_.cpu_data()[i]));
    Dtype temp_loss(0.0);
    Dtype dist_this(dist_pred_.cpu_data()[i]);
    //conditions
    if (label_difference > margin2) {
      temp_loss = std::max(margin2 - dist_this, Dtype(0.0));
      if (norm2) {
        loss += temp_loss * temp_loss;
      }
      else {
        loss += temp_loss;
      }
    }
    else if (label_difference < margin1){
      temp_loss = std::max(Dtype(fabs(dist_this)) - margin1, Dtype(0.0));
      if (norm2) {
        loss += temp_loss * temp_loss;
      }
      else {
        loss += temp_loss;
      }
    }
    else {
      temp_loss = std::max(dist_this - margin2, Dtype(0.0)) + std::max(margin1 - dist_this, Dtype(0.0));
      if (norm2) {
        loss += temp_loss * temp_loss;
      }
      else {
        loss += temp_loss;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  if (norm2) {
    loss = loss / Dtype(2);
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PRankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin1 = this->layer_param_.piecewise_ranking_loss_param().margin1();
  Dtype margin2 = this->layer_param_.piecewise_ranking_loss_param().margin2();
  bool norm2 = this->layer_param_.piecewise_ranking_loss_param().norml2();
  const int count = bottom[0]->count();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      Dtype* bout = bottom[i]->mutable_cpu_diff();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        const Dtype sign_this = (diff_ground_.cpu_data()[j] > 0) ? 1 : -1;
        const Dtype alpha_this = alpha * sign_this;
         
        Dtype mdist(0.0);
        Dtype label_difference(fabs(diff_ground_.cpu_data()[j]));
        if (bottom.size() > 4 && bottom[4]->cpu_data()[j] < 0.5) {
          caffe_set(channels, Dtype(0), bout + (j*channels));
          continue;
        }
        if (label_difference > margin2) {
          mdist = margin2 - dist_pred_.cpu_data()[j];
          if (norm2) {
            if (mdist > Dtype(0.0)) { bout[j] = -alpha_this * mdist; }
            else { caffe_set(channels, Dtype(0), bout + (j*channels));}
          }
          else {
            if (mdist > Dtype(0.0)) { bout[j] = -alpha_this * Dtype(1.0); }
            else { caffe_set(channels, Dtype(0), bout + (j*channels));}
          }
        }

        else if (label_difference < margin1) {
          mdist = fabs(dist_pred_.cpu_data()[j]) - margin1;
          if (mdist > Dtype(0.0)) {
            if (norm2) {
              if (dist_pred_.cpu_data()[j] > 0) { bout[j] = alpha_this * mdist;}
              else {bout[j] = -alpha_this * mdist;}
            }
            else {
              if (dist_pred_.cpu_data()[j] > 0) { bout[j] = alpha_this * Dtype(1.0);}
              else {bout[j] = -alpha_this * Dtype(1.0);}
            }
          }
          else { caffe_set(channels, Dtype(0), bout + (j*channels));}
        }
        else {
          if (norm2) {
            mdist = dist_pred_.cpu_data()[j] - margin2;
            if (mdist > Dtype(0.0)) { bout[j] = alpha_this * mdist;}
            else {
              mdist = margin1 - dist_pred_.cpu_data()[j];
              if (mdist > Dtype(0.0)) { bout[j] = -alpha_this * mdist; }
              else { caffe_set(channels, Dtype(0), bout + (j*channels));}
            }
          }
          else {
            mdist = dist_pred_.cpu_data()[j] - margin2;
            if (mdist > Dtype(0.0)) { bout[j] = alpha_this * Dtype(1.0);}
            else {
              mdist = margin1 - dist_pred_.cpu_data()[j];
              if (mdist > Dtype(0.0)) { bout[j] = -alpha_this * Dtype(1.0); }
              else { caffe_set(channels, Dtype(0), bout + (j*channels));}
            }
          }
        }
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(count, loss_weight, bout);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PRankingLossLayer);
#endif

INSTANTIATE_CLASS(PRankingLossLayer);
REGISTER_LAYER_CLASS(PRankingLoss);

}  // namespace caffe
