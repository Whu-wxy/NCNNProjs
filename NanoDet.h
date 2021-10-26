// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ncnn/cpu.h"
#include "ncnn/net.h"
#include "ncnn/benchmark.h"
#include "ncnnmodelbase.h"
#include <exception>

using namespace std;
using namespace cv;

enum NANODET_TYPE
{
    NANO_M,
    NANO_M416,
    NANO_G,
    NANO_E320,
    NANO_E416,
    NANO_E512,
    NANO_REPVGG416
};

struct NANOObject
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class NanoDet : public ncnnModelBase
{
public:
    NanoDet(QObject *parent = 0);
    virtual ~NanoDet();

    virtual bool    predict(cv::Mat & frame);

private:
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
//    ncnn::UnlockedPoolAllocator blob_pool_allocator;
//    ncnn::PoolAllocator workspace_pool_allocator;

    int load(NANODET_TYPE type);
    int detect(const cv::Mat& rgb, std::vector<NANOObject>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);
    int draw(cv::Mat& rgb, const std::vector<NANOObject>& objects);

};

#endif // NANODET_H
