#ifndef __TF_EAST_UTIL_HPP__
#define __TF_EAST_UTIL_HPP__

#include <opencv2/core/mat.hpp>
#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

namespace tf_east {
tf::Status ReadTensorFromMat(const cv::Mat& mat, tf::Tensor& outTensor);

std::vector<float> ResizeImage(cv::Mat& image, cv::Mat& resized_img,
                               int max_side_len = 2400);

std::vector<std::vector<float>> RestoreRectangle(
    std::vector<std::pair<int, int>>& yx_indexs,
    std::vector<std::vector<float>>& coords, std::vector<float>& angles,
    std::vector<float>& scores);

void GetScore(std::vector<std::vector<float>>& boxes,
              tf::TTypes<float, 4>::Tensor& f_score,
              std::vector<float>& final_scores);

int GetDist(int x1, int y1, int x2, int y2);
}  // namespace tf_east
#endif  //__TF_EAST_UTIL_HPP__
