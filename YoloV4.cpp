#include "YoloV4.h"

//yolo-fastest-opt
YoloV4::YoloV4(QObject *parent) : ncnnModelBase("yolo-fastest-opt", parent)
{
//    net.opt.use_fp16_arithmetic = true;  // fp16运算加速
}

YoloV4::~YoloV4() {

}

bool YoloV4::predict(cv::Mat & frame)
{
    double ncnnstart = ncnn::get_current_time();
    std::vector<BoxInfo> boxes = detect(frame, 0.3, 0.7);
//    qDebug()<<"boxes: "<<boxes.size();
//    putText(frame, to_string(boxes.size()), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
    for(BoxInfo &boxInfo: boxes)
    {
        qDebug()<<"boxInfo: "<<boxInfo.x1<<", "<<boxInfo.y1<<", "<<boxInfo.x2<<", "<<boxInfo.y2;
        rectangle(frame, Point(boxInfo.x1, boxInfo.y1), Point(boxInfo.x2, boxInfo.y2), Scalar(0, 255, 0), 2);
//        putText(frame, labels[boxInfo.label], Point(boxInfo.x1, boxInfo.y1), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 5);
    }
    double ncnnfinish = ncnn::get_current_time();
    double model_time = (double)(ncnnfinish - ncnnstart) / 1000;
    putText(frame, to_string(model_time), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);

    return true;
}

std::vector<BoxInfo> YoloV4::detect(cv::Mat & image, float threshold, float nms_threshold) {

    cv::Mat temp;
    resize(image, temp, Size(input_size, input_size));
    ncnn::Mat in_net = ncnn::Mat::from_pixels(temp.data, ncnn::Mat::PIXEL_BGR2RGB, input_size, input_size);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);

    auto ex = net.create_extractor();
    ex.set_light_mode(true);
//    ex.set_num_threads(4);

#if NCNN_VULKAN
    qDebug()<<"gpu count: "<<ncnn::get_gpu_count();
    if(ncnn::get_gpu_count() > 0)
    {
        net.set_vulkan_device(0);
        ex.set_vulkan_compute(true);
    }
    else
        ex.set_vulkan_compute(false);
#endif
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    ncnn::Mat blob;
    ex.extract("output", blob);//output  detection_out
    auto boxes = decode_infer(blob, {(int) image.cols, (int) image.rows}, input_size, num_class, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());
//    nms(result,nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
YoloV4::decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold) {
    qDebug()<<"data shape: "<<data.w<<", "<<data.h;
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
        box.x1 = std::max(values[2] * (float) frame_size.width, (float)0);
        box.y1 = std::max(values[3] * (float) frame_size.height, (float)0);
        box.x2 = std::min(values[4] * (float) frame_size.width, (float) frame_size.width);
        box.y2 = std::min(values[5] * (float) frame_size.height, (float) frame_size.height);
        result.push_back(box);
    }
    return result;
}

