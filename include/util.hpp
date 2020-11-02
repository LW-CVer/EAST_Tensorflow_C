#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <opencv2/core/mat.hpp>
#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

tf::Status ReadTensorFromMat(const cv::Mat& mat, tf::Tensor& outTensor);

std::vector<float> ResizeImage(cv::Mat& image, cv::Mat& resized_img,
                               int max_side_len = 2400);

std::vector<std::vector<float>> RestoreRectangle(
    std::vector<std::pair<int, int>>& yx_indexs,
    std::vector<std::vector<float>>& coords, std::vector<float>& angles,
    std::vector<float>& scores);
#endif  //__UTIL_HPP__
