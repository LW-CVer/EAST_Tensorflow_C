#include "../include/east.hpp"
#include "../include/INIReader.hpp"
#include "../include/ini.hpp"
#include "../include/lanms.h"
#include "../include/util.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow/core/platform/env.h"

East::East()
    : m_inputlayer("input_images:0"),
      m_outputlayer{"feature_fusion/Conv_7/Sigmoid:0",
                    "feature_fusion/concat_3:0"},
      m_session(nullptr)
{
}
East::~East()
{
    m_session->Close();
    m_session = nullptr;
}

int East::init(const std::string& ini_path)
{
    INIReader reader(ini_path);
    m_gpu_index = reader.GetInteger("device", "gpu_index", 0);
    m_gpu_fraction = reader.GetReal("device", "gpu_fraction", 1);
    m_box_threshold = reader.GetReal("threshold", "box_threshold", 0.1);
    m_nms_threshold = reader.GetReal("threshold", "nms_threshold", 0.2);
    return 0;
}

int East::load_model(const std::string& model_path)
{
    std::cout << "Loading model..." << std::endl;
    auto options = tf::SessionOptions();
    auto gpu_options = options.config.mutable_gpu_options();
    gpu_options->set_visible_device_list(std::to_string(m_gpu_index));
    gpu_options->set_per_process_gpu_memory_fraction(m_gpu_fraction);
    options.config.set_allow_soft_placement(true);
    tf::Status status = tf::NewSession(options, &m_session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    tf::GraphDef graph_def;
    status = ReadBinaryProto(tf::Env::Default(), model_path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    status = m_session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    std::cout << "Model successfully loaded." << std::endl;
    return 0;
}
//后处理
std::vector<std::vector<float>> East::process(int h, int w,
                                              double score_threshold,
                                              std::vector<float>& ratios,
                                              std::vector<float>& final_scores)
{
    //数据类型为tf::TTypes<float, 4>::Tensor
    auto f_score = m_outputs[0].tensor<float, 4>();
    auto f_geometry = m_outputs[1].tensor<float, 4>();
    // std::cout<<geometry.shape().dim_size(0)<<"
    // "<<geometry.shape().dim_size(1)<<"
    // "<<geometry.shape().dim_size(2)<<std::endl;
    // 第一个是纵坐标，第二个是横坐标
    std::vector<std::pair<int, int>> yx_indexs;
    std::vector<std::vector<float>> coords;
    std::vector<float> angles;
    std::vector<float> scores;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (f_score(0, i, j, 0) > score_threshold) {
                yx_indexs.emplace_back(i, j);
                coords.push_back(
                    {f_geometry(0, i, j, 0), f_geometry(0, i, j, 1),
                     f_geometry(0, i, j, 2), f_geometry(0, i, j, 3)});
                scores.push_back(f_score(0, i, j, 0));
                angles.push_back(f_geometry(0, i, j, 4));
            }
        }
    }
    std::vector<std::vector<float>> temp_boxes =
        tf_east::RestoreRectangle(yx_indexs, coords, angles, scores);
    std::vector<std::vector<float>> boxes =
        lanms::merge_quadrangle_n9(temp_boxes, scores, m_nms_threshold);
    //获取每个box的分数
    tf_east::GetScore(boxes, f_score, final_scores);
    for (int i = 0; i < boxes.size(); i++) {
        for (int j = 0; j < 8; j++) {
            if (j % 2 == 0) {
                boxes[i][j] /= ratios[1];
            } else {
                boxes[i][j] /= ratios[0];
            }
        }
    }
    return boxes;
}

std::vector<EastResult> East::detect(cv::Mat& image, double score_threshold)
{
    m_detection_results.clear();
    cv::Mat resized_img;
    std::vector<float> ratios = tf_east::ResizeImage(image, resized_img);
    tf::TensorShape image_shape = tf::TensorShape();
    image_shape.AddDim(1);
    image_shape.AddDim(resized_img.rows);
    image_shape.AddDim(resized_img.cols);
    image_shape.AddDim(3);
    tf::Tensor image_tensor = tf::Tensor(tf::DT_FLOAT, image_shape);
    tf::Status ImageTensorStatus =
        tf_east::ReadTensorFromMat(resized_img, image_tensor);
    if (!ImageTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << ImageTensorStatus;
    }
    m_outputs.clear();
    std::vector<std::pair<std::string, tf::Tensor>> inputs;
    inputs.push_back(std::make_pair(m_inputlayer, image_tensor));
    tf::Status runStatus =
        m_session->Run(inputs, m_outputlayer, {}, &m_outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
    }
    std::vector<float> final_scores;
    //网络特征图相对输入图片缩放4倍
    std::vector<std::vector<float>> results =
        this->process(resized_img.rows / 4, resized_img.cols / 4,
                      score_threshold, ratios, final_scores);
    int index = 0;
    for (auto& result : results) {
        if (tf_east::GetDist(int(result[0]), int(result[1]), int(result[2]),
                             int(result[3])) < 5 ||
            tf_east::GetDist(int(result[0]), int(result[1]), int(result[6]),
                             int(result[7])) < 5) {
            index++;
            continue;
        }
        EastResult temp;
        // 0代表了检测的文本
        temp.label = 0;
        temp.score = final_scores[index];
        temp.box_coordinates[0] = int(result[0]);
        temp.box_coordinates[1] = int(result[1]);
        temp.box_coordinates[2] = int(result[2]);
        temp.box_coordinates[3] = int(result[3]);
        temp.box_coordinates[4] = int(result[4]);
        temp.box_coordinates[5] = int(result[5]);
        temp.box_coordinates[6] = int(result[6]);
        temp.box_coordinates[7] = int(result[7]);
        m_detection_results.push_back(temp);
        index++;
    }
    return m_detection_results;
}

std::shared_ptr<EastBase> CreateEast()
{
    return std::shared_ptr<EastBase>(new East());
}
