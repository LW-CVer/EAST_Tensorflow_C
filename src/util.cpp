#include "../include/util.hpp"

#include <assert.h>
#include <math.h>
#include <Eigen/Dense>
#include <cv.hpp>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "tensorflow/cc/ops/standard_ops.h"
namespace tf = tensorflow;

tf::Status ReadTensorFromMat(const cv::Mat& mat, tf::Tensor& outTensor)
{
    auto root = tf::Scope::NewRootScope();
    using namespace ::tf::ops;
    float* p = outTensor.flat<float>().data();
    cv::Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);
    return tf::Status::OK();
}

//缩放图片，网络要求输入图片尺度是32的倍数
std::vector<float> ResizeImage(cv::Mat& image, cv::Mat& resized_img,
                               int max_size_len)
{
    float ratio = 0;
    int resize_w = image.cols;
    int resize_h = image.rows;
    if ((image.cols > image.rows ? image.cols : image.rows) > max_size_len) {
        if (image.rows > image.cols) {
            ratio = float(max_size_len) / image.rows;
        } else {
            ratio = float(max_size_len) / image.cols;
        }
    } else {
        ratio = 1.0;
    }
    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);

    resize_h = resize_h % 32 == 0 ? resize_h : (resize_h / 32 - 1) * 32;
    resize_w = resize_w % 32 == 0 ? resize_w : (resize_w / 32 - 1) * 32;

    resize_h = resize_h > 32 ? resize_h : 32;

    resize_w = resize_w > 32 ? resize_w : 32;

    cv::cvtColor(image, resized_img, cv::COLOR_BGR2RGB);
    cv::resize(resized_img, resized_img, cv::Size(resize_w, resize_h));

    float ratio_h = resize_h / float(image.rows);
    float ratio_w = resize_w / float(image.cols);
    std::vector<float> ratios{ratio_h, ratio_w};
    return ratios;
}

//角度为正，顺时针旋转；角度为负，逆时针旋转
std::vector<std::vector<float>> RestoreRectangle(
    std::vector<std::pair<int, int>>& yx_indexs,
    std::vector<std::vector<float>>& coords, std::vector<float>& angles,
    std::vector<float>& scores)
{
    assert(yx_indexs.size() == coords.size() && coords.size() == angles.size());
    //横纵坐标交换并映射回原始尺度
    std::vector<std::pair<int, int>> xy_indexs;
    for (auto& index : yx_indexs) {
        xy_indexs.emplace_back(index.second * 4, index.first * 4);
    }

    //获取角度为正或为负的点,以及对应的坐标和索引
    //角度为正数
    std::vector<float> pos_angles;
    std::vector<std::pair<int, int>> pos_xy_indexs;
    std::vector<std::vector<float>> pos_coords;
    //角度为负数
    std::vector<float> neg_angles;
    std::vector<std::pair<int, int>> neg_xy_indexs;
    std::vector<std::vector<float>> neg_coords;

    //按正负对scores进行排序
    std::vector<float> temp_scores = scores;
    scores.clear();

    for (int i = 0; i < angles.size(); i++) {
        if (angles[i] >= 0) {
            pos_angles.push_back(angles[i]);
            pos_xy_indexs.push_back(xy_indexs[i]);
            pos_coords.push_back(coords[i]);
            scores.push_back(temp_scores[i]);
        }
    }
    for (int i = 0; i < angles.size(); i++) {
        if (angles[i] < 0) {
            neg_angles.push_back(angles[i]);
            neg_xy_indexs.push_back(xy_indexs[i]);
            neg_coords.push_back(coords[i]);
            scores.push_back(temp_scores[i]);
        }
    }
    //存储返回结果
    std::vector<std::vector<float>> final_results;

    if (pos_xy_indexs.size() > 0) {
        Eigen::MatrixXf pos_p;
        pos_p = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();
        pos_p.resize(10, pos_coords.size());
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < pos_coords.size(); j++) {
                switch (i) {
                    case 0:
                        pos_p(i, j) = 0;

                        break;
                    case 1:
                        pos_p(i, j) = -pos_coords[j][0] - pos_coords[j][2];

                        break;
                    case 2:
                        pos_p(i, j) = pos_coords[j][1] + pos_coords[j][3];

                        break;
                    case 3:
                        pos_p(i, j) = -pos_coords[j][0] - pos_coords[j][2];

                        break;
                    case 4:
                        pos_p(i, j) = pos_coords[j][1] + pos_coords[j][3];

                        break;
                    case 5:
                        pos_p(i, j) = 0;

                        break;
                    case 6:
                        pos_p(i, j) = 0;

                        break;
                    case 7:
                        pos_p(i, j) = 0;

                        break;
                    case 8:
                        pos_p(i, j) = pos_coords[j][3];

                        break;
                    case 9:
                        pos_p(i, j) = -pos_coords[j][2];

                        break;
                }
            }
        }
        //转置，N*10
        pos_p.transposeInPlace();
        Eigen::MatrixXf pos_rotate_x(pos_angles.size(), 2);
        Eigen::MatrixXf pos_rotate_y(pos_angles.size(), 2);
        for (int i = 0; i < pos_angles.size(); i++) {
            pos_rotate_x(i, 0) = cos(pos_angles[i]);
            pos_rotate_x(i, 1) = sin(pos_angles[i]);
            pos_rotate_y(i, 0) = -sin(pos_angles[i]);
            pos_rotate_y(i, 1) = cos(pos_angles[i]);
        }
        Eigen::MatrixXf pos_rotate(pos_angles.size(), 10);
        for (int i = 0; i < pos_angles.size(); i++) {
            for (int j = 0; j < 10; j++) {
                if (j % 2 == 0) {
                    pos_rotate(i, j) = pos_p(i, j) * pos_rotate_x(i, 0) +
                                       pos_p(i, j + 1) * pos_rotate_x(i, 1);
                    pos_rotate(i, j + 1) = pos_p(i, j) * pos_rotate_y(i, 0) +
                                           pos_p(i, j + 1) * pos_rotate_y(i, 1);
                }
            }
        }
        Eigen::MatrixXf pos_result(pos_angles.size(), 8);
        for (int i = 0; i < pos_angles.size(); i++) {
            for (int j = 0; j < 8; j++) {
                if (j % 2 == 0) {
                    pos_result(i, j) = pos_xy_indexs[i].first -
                                       pos_rotate(i, 8) + pos_rotate(i, j);
                } else {
                    pos_result(i, j) = pos_xy_indexs[i].second -
                                       pos_rotate(i, 9) + pos_rotate(i, j);
                }
            }
        }
        for (int i = 0; i < pos_result.rows(); i++) {
            std::vector<float> temp;
            for (int j = 0; j < pos_result.cols(); j++) {
                temp.push_back(pos_result(i, j));
            }
            final_results.push_back(temp);
        }
    }

    if (neg_xy_indexs.size() > 0) {
        Eigen::MatrixXf neg_p;
        neg_p = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();
        neg_p.resize(10, neg_coords.size());
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < neg_coords.size(); j++) {
                switch (i) {
                    case 0:

                        neg_p(i, j) = -neg_coords[j][1] - neg_coords[j][3];
                        break;
                    case 1:

                        neg_p(i, j) = -neg_coords[j][0] - neg_coords[j][2];
                        break;
                    case 2:

                        neg_p(i, j) = 0;
                        break;
                    case 3:

                        neg_p(i, j) = -neg_coords[j][0] - neg_coords[j][2];
                        break;
                    case 4:

                        neg_p(i, j) = 0;
                        break;
                    case 5:

                        neg_p(i, j) = 0;
                        break;
                    case 6:

                        neg_p(i, j) = -neg_coords[j][1] - neg_coords[j][3];
                        break;
                    case 7:

                        neg_p(i, j) = 0;
                        break;
                    case 8:

                        neg_p(i, j) = -neg_coords[j][1];
                        break;
                    case 9:

                        neg_p(i, j) = -neg_coords[j][2];
                        break;
                }
            }
        }

        neg_p.transposeInPlace();
        /*std::cout<<"************"<<std::endl;
        for(int i =0;i<neg_angles.size();i++){
            for(int j =0;j<10;j++){
                std::cout<<neg_p(i,j)<<" ";
            }
            std::cout<<""<<std::endl;
        }*/
        Eigen::MatrixXf neg_rotate_x(neg_angles.size(), 2);
        Eigen::MatrixXf neg_rotate_y(neg_angles.size(), 2);
        for (int i = 0; i < neg_angles.size(); i++) {
            neg_rotate_x(i, 0) = cos(-neg_angles[i]);
            neg_rotate_x(i, 1) = -sin(-neg_angles[i]);
            neg_rotate_y(i, 0) = sin(-neg_angles[i]);
            neg_rotate_y(i, 1) = cos(-neg_angles[i]);
        }

        Eigen::MatrixXf neg_rotate(neg_angles.size(), 10);
        for (int i = 0; i < neg_angles.size(); i++) {
            for (int j = 0; j < 10; j++) {
                if (j % 2 == 0) {
                    neg_rotate(i, j) = neg_p(i, j) * neg_rotate_x(i, 0) +
                                       neg_p(i, j + 1) * neg_rotate_x(i, 1);
                    neg_rotate(i, j + 1) = neg_p(i, j) * neg_rotate_y(i, 0) +
                                           neg_p(i, j + 1) * neg_rotate_y(i, 1);
                }
            }
        }
        /*
        for(int i =0;i<neg_angles.size();i++){
            for(int j =1;j<10;j++){
                std::cout<<neg_rotate(i,j)<<" ";
            }
            std::cout<<""<<std::endl;
        }*/

        Eigen::MatrixXf neg_result(neg_angles.size(), 8);
        for (int i = 0; i < neg_angles.size(); i++) {
            for (int j = 0; j < 8; j++) {
                if (j % 2 == 0) {
                    neg_result(i, j) = neg_xy_indexs[i].first -
                                       neg_rotate(i, 8) + neg_rotate(i, j);
                } else {
                    neg_result(i, j) = neg_xy_indexs[i].second -
                                       neg_rotate(i, 9) + neg_rotate(i, j);
                }
            }
        }
        for (int i = 0; i < neg_result.rows(); i++) {
            std::vector<float> temp;
            for (int j = 0; j < neg_result.cols(); j++) {
                temp.push_back(neg_result(i, j));
            }
            final_results.push_back(temp);
        }
    }
    return final_results;
}
