#ifndef CAFFE_P_RANKING_LOSS_LAYER_HPP_
#define CAFFE_P_RANKING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Piecewise ranking loss @f$
 *    see paper "Event-specific Image Importance" for loss details.
 */
template <typename Dtype>
class PRankingLossLayer : public LossLayer<Dtype> {
 public:
  explicit PRankingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 4; }
  virtual inline const char* type() const { return "PRankingLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index < 2;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_; // prediction difference between pair
  Blob<Dtype> diff_ground_;  // groundtruth difference between pair
  Blob<Dtype> dist_pred_; // prediction difference between pair after sign correction
};

}  // namespace caffe

#endif  // CAFFE_P_RANKING_LOSS_LAYER_HPP_
