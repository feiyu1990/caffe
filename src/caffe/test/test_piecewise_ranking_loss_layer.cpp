#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/piecewise_ranking_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <math.h>       /* fabs */

namespace caffe {

template <typename TypeParam>
class PRankingLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PRankingLossLayerTest()
      : blob_bottom_data_i_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_bottom_y_i_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_bottom_y_j_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    filler.Fill(this->blob_bottom_y_i_);
    blob_bottom_vec_.push_back(blob_bottom_y_i_);
    filler.Fill(this->blob_bottom_y_j_);
    blob_bottom_vec_.push_back(blob_bottom_y_j_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~PRankingLossLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_y_i_;
    delete blob_bottom_y_j_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_y_i_;
  Blob<Dtype>* const blob_bottom_y_j_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PRankingLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(PRankingLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PRankingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin1 = layer_param.piecewise_ranking_loss_param().margin1();
  const Dtype margin2 = layer_param.piecewise_ranking_loss_param().margin2();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  bool norm2 = layer_param.piecewise_ranking_loss_param().norml2();
  std::cout<< norm2 << endl;
  Dtype loss(0);
  Dtype temp_loss(0);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      Dtype diff_ground = this->blob_bottom_y_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_y_j_->cpu_data()[i*channels+j];
      Dtype label_difference(fabs(diff_ground));
      Dtype dist_this(0.0);
      if (diff_ground > 0) {  //if first is larger (G_a - G_b)
        dist_this = diff;
      }
      else { //else: (G_b - G_a)
        dist_this = - diff;
      }
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
  }

  loss = loss / static_cast<Dtype>(num);
  if (norm2) {
    loss = loss / Dtype(2);
  }
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(PRankingLossLayerTest, TestGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PRankingLossLayer<Dtype> layer(layer_param);
  layer_param.mutable_piecewise_ranking_loss_param()->set_norml2(true);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}
//
TYPED_TEST(PRankingLossLayerTest, TestGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_piecewise_ranking_loss_param()->set_norml2(false);
  PRankingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-3, 1702);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
