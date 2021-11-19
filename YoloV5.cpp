//
// Created by 邓昊晴 on 14/6/2020.
//

#include "YoloV5.h"

YoloV5::YoloV5(QObject *parent) : ncnnModelBase("yolov5m-opt-fp16", parent) {

}

YoloV5::~YoloV5() {

}

bool YoloV5::predict(cv::Mat & frame)
{
    double ncnnstart = ncnn::get_current_time();
    std::vector<BoxInfo> boxes = detect(frame, 0.25, 0.45);
    for(BoxInfo &boxInfo: boxes)
    {
//        qDebug()<<"boxInfo: "<<boxInfo.x1<<", "<<boxInfo.y1<<", "<<boxInfo.x2<<", "<<boxInfo.y2;
        rectangle(frame, Point(boxInfo.x1, boxInfo.y1), Point(boxInfo.x2, boxInfo.y2), Scalar(0, 255, 0), 2);
//        putText(frame, labels[boxInfo.label], Point(boxInfo.x1, boxInfo.y1), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 5);
    }
    double ncnnfinish = ncnn::get_current_time();
    double model_time = (double)(ncnnfinish - ncnnstart) / 1000;
    putText(frame, to_string(model_time), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
    return true;
}

std::vector<BoxInfo> YoloV5::detect(cv::Mat image, float threshold, float nms_threshold)
{
    // letterbox pad to multiple of 32
    int w = image.cols;
    int h = image.rows;
    int width = w;
    int height = h;
    float scale = 1.f;
    // 长边缩放到input_size
    if (w > h) {
        scale = (float) input_size / w;
        w = input_size;
        h = h * scale;
    } else {
        scale = (float) input_size / h;
        h = input_size;
        w = w * scale;
    }

    resize(image, image, Size(w, h));
    ncnn::Mat in_net = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
//    if (toUseGPU) {  // 消除提示
//        ex.set_vulkan_compute(toUseGPU);
//    }
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    for (const auto &layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        auto boxes = decode_infer(blob, layer.stride, {width, height}, input_size,
                                  num_class, layer.anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());
    }

    qDebug()<<"proposals: "<<result.size();

    nms(result, nms_threshold);
    for(BoxInfo& info: result)
    {
        qDebug()<<"label: "<<labels[info.label];
    }
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
YoloV5::decode_infer(ncnn::Mat &data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
                     const std::vector<yolocv::YoloSize> &anchors, float threshold) {
    std::vector<BoxInfo> result;
    int grid_size = int(sqrt(data.h));
    float *mat_data[data.c];
    for (int i = 0; i < data.c; i++) {
        mat_data[i] = data.channel(i);
    }
    float cx, cy, w, h;
    for (int shift_y = 0; shift_y < grid_size; shift_y++) {
        for (int shift_x = 0; shift_x < grid_size; shift_x++) {
            int loc = shift_x + shift_y * grid_size;
            for (int i = 0; i < 3; i++) {
                float *record = mat_data[i];
                float *cls_ptr = record + 5;
                for (int cls = 0; cls < num_classes; cls++) {
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    if (score > threshold) {
                        cx = (sigmoid(record[0]) * 2.f - 0.5f + (float) shift_x) * (float) stride;
                        cy = (sigmoid(record[1]) * 2.f - 0.5f + (float) shift_y) * (float) stride;
                        w = pow(sigmoid(record[2]) * 2.f, 2) * anchors[i].width;
                        h = pow(sigmoid(record[3]) * 2.f, 2) * anchors[i].height;
                        //printf("[grid size=%d, stride = %d]x y w h %f %f %f %f\n",grid_size,stride,record[0],record[1],record[2],record[3]);
                        BoxInfo box;
                        box.x1 = std::max(0, std::min(frame_size.width, int((cx - w / 2.f) * (float) frame_size.width / (float) net_size)));
                        box.y1 = std::max(0, std::min(frame_size.height, int((cy - h / 2.f) * (float) frame_size.height / (float) net_size)));
                        box.x2 = std::max(0, std::min(frame_size.width, int((cx + w / 2.f) * (float) frame_size.width / (float) net_size)));
                        box.y2 = std::max(0, std::min(frame_size.height, int((cy + h / 2.f) * (float) frame_size.height / (float) net_size)));
                        box.score = score;
                        box.label = cls;
                        result.push_back(box);
                    }
                }
            }
            for (auto &ptr:mat_data) {
                ptr += (num_classes + 5);
            }
        }
    }
    return result;
}

void YoloV5::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
