#ifndef DETECTORPSENET_H
#define DETECTORPSENET_H

#include <QObject>
#include <QTime>
#include <QDebug>
#include <QDir>
#include <QtGlobal>
#include <queue>

#include "opencv2/opencv.hpp"//添加Opencv相关头文件
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"
#include "ncnn/layer.h"
//#include "ncnn/pipeline.h"
//#include "ncnn/allocator.h"
//#include "ncnn/command.h"
#include "ncnn/benchmark.h"
#include "ncnnmodelbase.h"

using namespace std;
using namespace cv;

class DetectorPSENet : public ncnnModelBase
{
    Q_OBJECT
public:
    explicit DetectorPSENet(QObject *parent = 0);
    virtual ~DetectorPSENet();

signals:

public slots:

private:

public:
    virtual bool    predict(cv::Mat & frame);

    void    pse_decode(ncnn::Mat& features,
                                  std::map<int, std::vector<cv::Point>>& contours_map,
                                  const float thresh,
                                  const float min_area, int min_map_id=0
                                  );
};

#endif // DETECTORPSENET_H
