#include "ncnndlg.h"
#include <QDesktopWidget>

#ifdef Q_OS_ANDROID
QString selectedFileName = "/storage/emulated/0/DetectorData/test.jpg";
#else
QString selectedFileName = "D:/QtWork/NCNNProjs/src/test.jpg";
#endif

#ifdef Q_OS_ANDROID

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_amin_NCNNProjs_NCNNProjs_fileSelected(JNIEnv */*env*/,
                                             jobject /*obj*/,
                                             jstring results)
{
    selectedFileName = QAndroidJniObject(results).toString();
}

#ifdef __cplusplus
}
#endif

#endif // Q_OS_ANDROID


NCNNDlg::NCNNDlg(QWidget *parent)
    : QDialog(parent)
{
#ifdef Q_OS_ANDROID
    QDesktopWidget * deskTop = qApp->desktop();
    QRect rect = deskTop->availableGeometry();
    this->setFixedSize(rect.width(), rect.height());
#else
    showMaximized();
#endif // Q_OS_ANDROID

//    m_ncnnModel = new DetectorPSENet(this);
    m_ncnnModel = new YoloV4(this);

    QVBoxLayout* mainl = new QVBoxLayout(this);
    QGridLayout* mainLay = new QGridLayout();

    imgLabel = new QLabel(this);
    QImage bkground(":/img/src/test.jpg");
    bkground = bkground.scaled(this->size(), Qt::KeepAspectRatio);
    imgLabel->setPixmap(QPixmap::fromImage(bkground));

    captureBtn = new QPushButton("拍照");
    captureBtn->setChecked(true);
    albumBtn = new QPushButton("相册");
    processBtn = new QPushButton("处理");
    saveBtn = new QPushButton("保存");
    btnGroup = new QButtonGroup(this);
    btnGroup->addButton(captureBtn,0);
    btnGroup->addButton(albumBtn,1);
    btnGroup->addButton(processBtn,2);
    btnGroup->addButton(saveBtn,3);
    btnGroup->setExclusive(true);

    mainLay->addWidget(captureBtn    ,1,0,1,1);
    mainLay->addWidget(albumBtn      ,1,1,1,1);
    mainLay->addWidget(processBtn    ,1,2,1,1);
    mainLay->addWidget(saveBtn       ,1,3,1,1);

    mainl->addStretch();
    mainl->addWidget(imgLabel);
    mainl->addStretch();
    mainl->addLayout(mainLay);

    connect(btnGroup,SIGNAL(buttonClicked(int)),this,SLOT(btnClicked(int)));
}

NCNNDlg::~NCNNDlg()
{

}

#ifdef Q_OS_ANDROID
void NCNNDlg::btnClicked(int btnID)
{
    if(btnID == 0)
    {
        selectedFileName = "#";
        QAndroidJniObject::callStaticMethod<void>("com/amin/NCNNProjs/NCNNProjs",
                                                  "captureAnImage",
                                                  "()V");
        while(selectedFileName == "#")
            qApp->processEvents();

        if(selectedFileName != "#")
        {
            if(QFile(selectedFileName).exists())
            {
                outputImg.release();

                QImage bkground(selectedFileName);
                bkground = bkground.scaled(this->size(), Qt::KeepAspectRatio);
                imgLabel->setPixmap(QPixmap::fromImage(bkground));
            }
        }
    }
    else if(btnID == 1)
    {
        selectedFileName = "#";
        QAndroidJniObject::callStaticMethod<void>("com/amin/NCNNProjs/NCNNProjs",
                                                  "openAnImage",
                                                  "()V");

        while(selectedFileName == "#")
            qApp->processEvents();

        if(QFile(selectedFileName).exists())
        {
            outputImg.release();

            QImage bkground(selectedFileName);
            bkground = bkground.scaled(this->size(), Qt::KeepAspectRatio);
            imgLabel->setPixmap(QPixmap::fromImage(bkground));
        }
    }
    else if(btnID == 2)
    {
        if(selectedFileName.length()==0 || selectedFileName == "#")
        {
            QMessageBox::information(this, "提示", "请打开一张图片");
            return;
        }

        qDebug()<<"open img from: "<<selectedFileName;
        //处理图片
        Mat frame = imread(selectedFileName.toStdString());
        if(!m_ncnnModel->hasLoadNet())
            return;

        if(m_ncnnModel->predict(frame))//处理图片
        {
            outputImg = frame.clone();

            QImage img = MatToQImage(frame);
            img = img.scaled(this->size(), Qt::KeepAspectRatio);
            imgLabel->setPixmap(QPixmap::fromImage(img));
        }
    }
    else if(btnID == 3)
    {
        if(outputImg.empty()) return;

        QFileInfo info(selectedFileName);
        if(!info.exists()) return;

        AndroidSetup setup;
        QString dataDir = setup.getAppDataDir();
        QString savePath = dataDir + QDir::separator() + QDateTime::currentDateTime().toString("ncnn_yyMdhms") + ".jpg";

        imwrite(savePath.toStdString(), outputImg);  //opencv4androud可以编译通过

        QMessageBox::information(this, "提示", "保存成功");
    }
}

#else
void NCNNDlg::btnClicked(int btnID)
{
    if(btnID == 0)
    {

    }
    else if(btnID == 1)
    {
        selectedFileName = QFileDialog::getOpenFileName(this, "选择图片", "/", "Images (*.png *.bmp *.jpg)");

        if(QFile(selectedFileName).exists())
        {
            outputImg.release();

            QImage bkground(selectedFileName);
            bkground = bkground.scaled(this->size(), Qt::KeepAspectRatio);
            imgLabel->setPixmap(QPixmap::fromImage(bkground));
        }
        else
            selectedFileName = "#";
    }
    else if(btnID == 2)
    {
        if(selectedFileName.length()==0 || selectedFileName == "#")
        {
            QMessageBox::information(this, "提示", "请打开一张图片");
            return;
        }

        qDebug()<<"open img from: "<<selectedFileName;
        //处理图片
        Mat frame = imread(selectedFileName.toStdString());
        if(frame.empty() || frame.cols == 0)
            return;
        Mat frame2Detect = frame.clone();
        if(!m_ncnnModel->hasLoadNet())
            return;

        if(m_ncnnModel->predict(frame))//处理图片
        {
            outputImg = frame.clone();

            QImage img = MatToQImage(frame);
            img = img.scaled(this->size(), Qt::KeepAspectRatio);
            imgLabel->setPixmap(QPixmap::fromImage(img));
        }
    }
    else if(btnID == 3)
    {
        if(outputImg.empty()) return;

        QFileInfo info(selectedFileName);
        if(!info.exists()) return;

        QString dataDir = info.path();
        QString savePath = dataDir + QDir::separator() + QDateTime::currentDateTime().toString("ncnn_yyMdhms") + ".jpg";

        imwrite(savePath.toStdString(), outputImg);

        QMessageBox::information(this, "提示", "保存成功");
    }
}

#endif // Q_OS_ANDROID
