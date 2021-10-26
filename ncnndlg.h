#ifndef NCNNDLG_H
#define NCNNDLG_H

#include <QDialog>
#include <QtWidgets>
#include <QPushButton>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFile>
#include <QButtonGroup>
#include <QDebug>
#include <iostream>
#include <QFileDialog>

#ifdef Q_OS_ANDROID
#include <QtAndroidExtras>
#endif // Q_OS_ANDROID

#include <QMessageBox>
#include <QApplication>
#include <QFileInfo>

#include "opencv2/opencv.hpp"//添加Opencv相关头文件
#include "opencv2/core/core.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgproc.hpp"

#include "imgutils.h"
#include "androidsetup.h"
#include "time.h"
#include "ncnnmodelbase.h"
#include "detectorpsenet.h"
#include "YoloV5CustomLayer.h"
#include "YoloV4.h"
#include "DBFace.h"
#include "ENet.h"
#include "NanoDet.h"

#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"
#include "ncnn/layer.h"

using namespace cv;
using namespace std;

enum Model
{
    MD_YOLO,
    MD_NANODET,
    MD_PSENet,
    MD_ENet,
    MD_DBFace
};

class NCNNDlg : public QDialog
{
    Q_OBJECT

public:
    NCNNDlg(QWidget *parent = 0);
    ~NCNNDlg();


private:
    QLabel*     imgLabel;
    QButtonGroup* btnGroup;
    QPushButton* captureBtn;
    QPushButton* albumBtn;
    QPushButton* processBtn;
    QPushButton* saveBtn;
    QPushButton* switchBtn;

    ncnnModelBase*  m_ncnnModel;
    Model   m_curModel;

    Mat     outputImg;

protected slots:
    void btnClicked(int);
};

#endif // NCNNDLG_H
