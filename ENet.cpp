//
// Created by WZTENG on 2020/08/28 028.
//

#include "ENet.h"
//#include "SimplePose.h"

ENet::ENet(QObject *parent) : ncnnModelBase("ENet_sim-opt", parent)
{

}

ENet::~ENet()
{

}

bool ENet::predict(cv::Mat & frame)
{
    double ncnnstart = ncnn::get_current_time();
    cv::Mat prediction = detect_enet(frame);
    for(int i=0; i<prediction.rows; i++)
    {
        for(int j=0; j<prediction.cols; j++)
        {
            int pt = prediction.at<uchar>(i,j);
            if(pt < cityspace_colormap.size())
            {
                frame.at<Vec3b>(i,j)[0] = cityspace_colormap[pt][0];
                frame.at<Vec3b>(i,j)[1] = cityspace_colormap[pt][1];
                frame.at<Vec3b>(i,j)[2] = cityspace_colormap[pt][2];
            }
        }
    }

    double ncnnfinish = ncnn::get_current_time();
    double model_time = (double)(ncnnfinish - ncnnstart) / 1000;
    putText(frame, to_string(model_time), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);

    return true;
}


cv::Mat ENet::detect_enet(cv::Mat image)
{
    int ori_width = image.cols;
    int ori_height = image.rows;

    resize(image, image, Size(target_size_w, target_size_h));
    ncnn::Mat in_net = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, target_size_w, target_size_h);

//    float mean[3] = {0.0f, 0.0f, 0.0f};
//    float norm[3] = {1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f};
    float mean[3] = {123.68f, 116.28f, 103.53f};
    float norm[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in_net.substract_mean_normalize(mean, norm);

    ncnn::Mat maskout;

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
    ex.input("input", in_net);
    ex.extract("output", maskout);

    int mask_c = maskout.c;
    int mask_w = maskout.w;
    int mask_h = maskout.h;

    cv::Mat prediction = cv::Mat::zeros(mask_h, mask_w, CV_8UC1);
    ncnn::Mat chn[mask_c];
    for (int i = 0; i < mask_c; i++) {
        chn[i] = maskout.channel(i);
    }
    for (int i = 0; i < mask_h; i++) {
        const float *pChn[mask_c];
        for (int c = 0; c < mask_c; c++) {
            pChn[c] = chn[c].row(i);
        }

        auto *pCowMask = prediction.ptr<uchar>(i);

        for (int j = 0; j < mask_w; j++) {
            int maxindex = 0;
            float maxvalue = -1000;
            for (int n = 0; n < mask_c; n++) {
                if (pChn[n][j] > maxvalue) {
                    maxindex = n;
                    maxvalue = pChn[n][j];
                }
            }
            pCowMask[j] = maxindex;
        }

    }

    cv::Mat pred_resize;
    cv::resize(prediction, pred_resize, cv::Size(ori_width, ori_height), 0, 0, cv::INTER_NEAREST);

    ncnn::Mat maskMat;
    maskMat = ncnn::Mat::from_pixels_resize(pred_resize.data, ncnn::Mat::PIXEL_GRAY,
                                            pred_resize.cols, pred_resize.rows,
                                            ori_width, ori_height);


    return pred_resize;
}

