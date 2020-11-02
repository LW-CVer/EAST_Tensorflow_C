#include <iostream>
#include "../include/east_base.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
int main()
{
    std::string image_path = "../../model/test.jpg";
    std::string model_path = "../../model/east.pb";
    std::string ini_path = "../../config/east.ini";
    std::shared_ptr<EastBase> east = CreateEast();
    east->init(ini_path);
    east->load_model(model_path);
    cv::Mat test_img;
    test_img = cv::imread(image_path);
    std::vector<EastResult> results = east->detect(test_img, 0.8);
    for (auto result : results) {
        std::cout << result.label << std::endl;
        std::cout << result.score << std::endl;
        for (int i = 0; i < 8; i++) {
            std::cout << result.box_coordinates[i] << " ";
        }
        std::cout << "" << std::endl;
    }
    return 0;
}
