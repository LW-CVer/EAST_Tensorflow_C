#ifndef __EAST_HPP__
#define __EAST_HPP__

#include "east_base.hpp"
#include "tensorflow/core/public/session.h"
namespace tf = tensorflow;

class East : public EastBase
{
   public:
    East();
    ~East();
    int init(const std::string& ini_path) override;
    int load_model(const std::string& model_path) override;
    std::vector<EastResult> detect(cv::Mat& image,
                                   double score_threshold) override;

   private:
    std::vector<tf::Tensor> m_outputs;
    std::string m_inputlayer;
    std::vector<std::string> m_outputlayer;
    tf::Session* m_session;
    std::vector<EastResult> m_detection_results;
    int m_gpu_index;
    float m_gpu_fraction;
    float m_box_threshold;
    float m_nms_threshold;
    //后处理，会将结果存储在m_detection_resultsl
    std::vector<std::vector<float>> process(int h, int w,
                                            double score_threshold,
                                            std::vector<float>& ratios);
};

#endif
