#include "detector.h"
#include <QFile>
#include <androidsetup.h>

Detector::Detector(QObject *parent) : QObject(parent)
{
    bLoad = false;

    moveFiles();
#ifdef Q_OS_ANDROID
    AndroidSetup setup;
    QString dataDir = setup.getAppDataDir();
#else
    QString dataDir = "D:/QtWork/NCNNDemo/src/";
#endif // Q_OS_ANDROID

    QString modelDir(dataDir+"/psenet_lite_mbv2.param");

    if(QFile::exists(modelDir))
    {
        QByteArray ba = modelDir.toLatin1();
        char *modeldir = ba.data();

        int res = net.load_param(modeldir);  //鍔犺浇妯″瀷
        qDebug()<<"param consumed: "<<res;

        modelDir = dataDir+"/psenet_lite_mbv2.bin";

        ba = modelDir.toLatin1();
        modeldir = ba.data();
        int consumed = net.load_model(modeldir);    //"assets:/dst/psenet_lite_mbv2.bin"
        qDebug()<<"bin consumed: "<<consumed;

        bLoad = true;
    }
    else
    {
        qDebug()<<"weights file not exist";
        bLoad = false;
    }
}

Detector::~Detector()
{
    net.clear(); //鍗歌浇妯″瀷

}

//浠巃ssets绉诲姩鍒版櫘閫氳矾寰勪笅
bool Detector::moveFiles()
{
#ifdef Q_OS_ANDROID
    AndroidSetup setup;
    QString dataDir = setup.getAppDataDir();
    qDebug()<<"data Dir:"<<dataDir;

    if(QFile::exists("assets:/dst/psenet_lite_mbv2.bin"))
        qDebug()<<"psenet_lite_mbv2.bin exist";
    else
        qDebug()<<"psenet_lite_mbv2.bin not exist";

    if(QFile::exists("assets:/dst/psenet_lite_mbv2.param"))
        qDebug()<<"psenet_lite_mbv2.param exist";
    else
        qDebug()<<"psenet_lite_mbv2.param not exist";

    bool bMove = true;
    QString dstName = dataDir + "/psenet_lite_mbv2.bin";
    if(!QFile::copy("assets:/dst/psenet_lite_mbv2.bin", dstName))
    {
        qDebug()<<"copy psenet_lite_mbv2.bin fail";
        bMove = false;
    }

    dstName = dataDir + "/psenet_lite_mbv2.param";    //psenet_lite_mbv2    psenet_lite_mbv2
    if(!QFile::copy("assets:/dst/psenet_lite_mbv2.param", dstName))
    {
        qDebug()<<"copy psenet_lite_mbv2.param fail";
        bMove = false;
    }

    dstName = dataDir + "/test.jpg";
    if(!QFile::copy("assets:/dst/test.jpg", dstName))
    {
        qDebug()<<"copy test.jpg fail";
        bMove = false;
    }

    qDebug()<<"move status:"<<bMove;
    return bMove;
#endif // Q_OS_ANDROID
}

bool Detector::detect(Mat & frame, map<int, vector<Point>>& contours_map, double& model_time)
{
    clock_t start=0, finish=0;

    qDebug()<<"origin in H: "<<frame.rows;
    qDebug()<<"origin in W: "<<frame.cols;

    frame = resize_img(frame, 800);   //pse_sim<800,  psenet_lite_mbv2鍥惧儚鍙互寰堝ぇ(1500閮藉彲浠

    int w = frame.cols;
    int h = frame.rows;
    qDebug()<<"in H: "<<h;
    qDebug()<<"in W: "<<w;

    //from_pixels_resize
    ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, w, h); //512
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    qDebug()<<"ncnn in H: "<<in.h;
    qDebug()<<"ncnn in W: "<<in.w;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(true);//鍚敤鏃讹紝涓棿Blob灏嗚鍥炴敹
    //extractor.set_num_threads(4); //澶氱嚎绋嬪姞閫熺殑寮€鍏筹紝璁剧疆绾跨▼鏁拌兘鍔犲揩璁＄畻

    // Android涓敤clock()璁℃椂涓嶅噯纭紝瑕佺敤ncnn::get_current_time()
    start = clock();
    double ncnnstart = ncnn::get_current_time();

    int bSuccess = extractor.input("input", in);
    if(bSuccess != 0)
        return false;
    qDebug()<<"extractor in";


    ncnn::Mat out;
    bSuccess = extractor.extract("out", out);
    finish = clock();
    double ncnnfinish = ncnn::get_current_time();

    if(bSuccess != 0)
        return false;

    qDebug()<<"extractor out";


    qDebug()<<"out H: "<<out.h;
    qDebug()<<"out W: "<<out.w;
    qDebug()<<"out C: "<<out.c;

    qDebug()<<"model time: "<<(double)(finish - start) / CLOCKS_PER_SEC;
    model_time = (double)(finish - start) / CLOCKS_PER_SEC;
    qDebug()<<"ncnn model time: "<<(double)(ncnnfinish - ncnnstart)/1000;
    qDebug()<<"ncnn";

    float *outdata = (float *) out.data;
    if(outdata != nullptr)
    {
        frame = Mat(out.h, out.w, CV_8UC1, Scalar(0));

        for(int i=0; i<frame.rows; i++)
        {
            for(int j=0; j<frame.cols; j++)
            {
                if (outdata[i * out.w + j + out.w*out.h*0] >= 0.3)
                    frame.at<uchar>(i, j) = 255;
                else
                    frame.at<uchar>(i, j) = 0;
            }
        }
    }

    start = clock();
    pse_decode(out, contours_map, 0.5, 5, 2);  //min_map_id=0-5,濡傛灉灏忔枃鏈娴嬩笉鍒帮紝鍙互灏濊瘯鎶婅繖涓暟澧炲ぇ
    finish = clock();
    qDebug()<<"pse time: "<<(double)(finish - start) / CLOCKS_PER_SEC;

    return true;
}


// Code from https://link.zhihu.com/?target=https%3A//github.com/ouyanghuiyu/chineseocr_lite/blob/master/ncnn_project/ocr/src/ocr.cpp
void Detector::pse_decode(ncnn::Mat& features,
                          map<int, vector<Point>>& contours_map,
                          const float thresh,
                          const float min_area, int min_map_id)
{

    // get kernels
    float *srcdata = (float *) features.data;
    std::vector<cv::Mat> kernels;

    float _thresh = thresh;
    cv::Mat scores = cv::Mat::zeros(features.h, features.w, CV_32FC1);
    for (int c = features.c - 1; c >= min_map_id; --c){
        cv::Mat kernel(features.h, features.w, CV_8UC1);
        for (int i = 0; i < features.h; i++) {
            for (int j = 0; j < features.w; j++) {

                if (c==features.c - 1) scores.at<float>(i, j) = srcdata[i * features.w + j + features.w*features.h*c ] ;

                if (srcdata[i * features.w + j + features.w*features.h*c ] >= _thresh) {
                    // std::cout << srcdata[i * src.w + j] << std::endl;
                    kernel.at<uint8_t>(i, j) = 1;
                } else {
                    kernel.at<uint8_t>(i, j) = 0;
                }

            }
        }
        kernels.push_back(kernel);
    }

    // make label
    cv::Mat label;
    std::map<int, int> areas;
    std::map<int, float> scores_sum;
    cv::Mat mask(features.h, features.w, CV_32S, cv::Scalar(0));
    cv::connectedComponents(kernels[kernels.size()-1], label, 4);


    for (int y = 0; y < label.rows; ++y) {
        for (int x = 0; x < label.cols; ++x) {
            int value = label.at<int32_t>(y, x);
            float score = scores.at<float>(y,x);
            if (value == 0) continue;
            areas[value] += 1;

            scores_sum[value] += score;
        }
    }

    std::queue<cv::Point> queue, next_queue;

    for (int y = 0; y < label.rows; ++y) {

        for (int x = 0; x < label.cols; ++x) {
            int value = label.at<int>(y, x);

            if (value == 0) continue;
            if (areas[value] < min_area) {
                areas.erase(value);
                continue;
            }

            if (scores_sum[value]*1.0 /areas[value] < 0.35  )
            {
                areas.erase(value);
                scores_sum.erase(value);
                continue;
            }
            cv::Point point(x, y);
            queue.push(point);
            mask.at<int32_t>(y, x) = value;
        }
    }

    /// growing text line
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int idx = kernels.size()-2; idx >= 0; --idx) {
        while (!queue.empty()) {
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int value = mask.at<int32_t>(y, x);

            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int _x = x + dx[d];
                int _y = y + dy[d];

                if (_y < 0 || _y >= mask.rows) continue;
                if (_x < 0 || _x >= mask.cols) continue;
                if (kernels[idx].at<uint8_t>(_y, _x) == 0) continue;
                if (mask.at<int32_t>(_y, _x) > 0) continue;

                cv::Point point_dxy(_x, _y);
                queue.push(point_dxy);

                mask.at<int32_t>(_y, _x) = value;
                is_edge = false;
            }

            if (is_edge) next_queue.push(point);
        }
        std::swap(queue, next_queue);
    }

    // make contoursMap
    for (int y=0; y < mask.rows; ++y)
        for (int x=0; x < mask.cols; ++x) {
            int idx = mask.at<int32_t>(y, x);
            if (idx == 0) continue;
            contours_map[idx].emplace_back(cv::Point(x, y));
        }
}


void Detector::pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(5);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

Mat Detector::resize_img(cv::Mat src, int long_size)
{
    int w = src.cols;
    int h = src.rows;
    // std::cout<<"鍘熷浘灏哄 (" << w << ", "<<h<<")"<<std::endl;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)long_size / w;
        w = long_size;
        h = h * scale;
    }
    else
    {
        scale = (float)long_size / h;
        h = long_size;
        w = w * scale;
    }
    if (h % 32 != 0)
    {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0)
    {
        w = (w / 32 + 1) * 32;
    }
    // std::cout<<"缂╂斁灏哄 (" << w << ", "<<h<<")"<<std::endl;
    cv::Mat result;
    cv::resize(src, result, cv::Size(w, h));
    return result;
}
