//
// Created by WZTENG on 2020/08/28 028.
//

#ifndef YOLOV5_ENET_H
#define YOLOV5_ENET_H


#include "ncnn/net.h"
#include "ncnn/benchmark.h"
#include "ncnnmodelbase.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

class ENet : public ncnnModelBase {
public:
    ENet(QObject *parent = 0);
    virtual ~ENet();

    virtual bool    predict(cv::Mat & frame);
    cv::Mat detect_enet(cv::Mat image);

private:
    vector<vector<int>> cityspace_colormap{
        {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
        {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
        {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}
        };
//    int cityspace_colormap[][3] = {
//            {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
//            {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
//            {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}
//    };

    int target_size_w = 512;
    int target_size_h = 512;
};


#endif //YOLOV5_ENET_H
