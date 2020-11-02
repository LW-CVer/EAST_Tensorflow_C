#ifndef __EAST_BASE_HPP__
#define __EAST_BASE_HPP__
#include <memory>
#include <string>
#include <vector>
#include "opencv2/core/mat.hpp"
struct EastResult
{
    int label;
    int box_coordinates[8];
    float score;
};

class EastBase
{
   public:
    virtual ~EastBase() = default;
    virtual int init(const std::string& ini_path) = 0;
    virtual int load_model(const std::string& model_path) = 0;
    virtual std::vector<EastResult> detect(cv::Mat& image,
                                           double score_threshold) = 0;
};

std::shared_ptr<EastBase> CreateEast();
#endif
