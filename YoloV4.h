#ifndef YOLOV4_H
#define YOLOV4_H

#include <QDebug>

#include "ncnn/net.h"
#include "ncnn/cpu.h"
#include "ncnn/benchmark.h"
#include "YoloV5.h"

#include "opencv2/opencv.hpp"//添加Opencv相关头文件
#include "ncnnmodelbase.h"
#include "math.h"

using namespace std;
using namespace cv;

class YoloV4 : public ncnnModelBase {
public:
    YoloV4(QObject *parent = 0);
    virtual ~YoloV4();

    virtual bool    predict(cv::Mat & frame);
    std::vector<BoxInfo> detect(cv::Mat & image, float threshold, float nms_threshold);
    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush"};
private:
    static std::vector<BoxInfo>
    decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold);

//    static void nms(std::vector<BoxInfo>& result,float nms_threshold);
    int input_size = 640/* / 2*/;
    int num_class = 80;
public:
    bool hasGPU;
    bool toUseGPU;
};


#endif //YOLOV4_H
