#-------------------------------------------------
#
# Project created by QtCreator 2020-05-18T12:13:54
#
#-------------------------------------------------

QT       += core gui
CONFIG += c++11

android{
QT       += androidextras
}

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NCNNProjs
TEMPLATE = app

#QMAKE_CFLAGS += -fno-rtti -fno-exceptions
#QMAKE_CXXFLAGS += -fno-rtti -fno-exceptions
#QMAKE_CXXFLAGS -= NCNN_DISABLE_RTTI
#QMAKE_CXXFLAGS += -D NCNN_DISABLE_RTTI=OFF
#QMAKE_CFLAGS += -D NCNN_DISABLE_RTTI=OFF

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
    DBFace.cpp \
    ENet.cpp \
    NanoDet.cpp \
    YoloV4.cpp \
    YoloV5.cpp \
    YoloV5CustomLayer.cpp \
    detectorpsenet.cpp \
        main.cpp \
        ncnndlg.cpp \
    imgutils.cpp \
    androidsetup.cpp \
    ncnnmodelbase.cpp \

HEADERS += \
    DBFace.h \
    ENet.h \
    NanoDet.h \
    YoloV4.h \
    YoloV5.h \
    YoloV5CustomLayer.h \
    detectorpsenet.h \
        ncnndlg.h \
    imgutils.h \
    androidsetup.h \
    ncnnmodelbase.h

CONFIG += mobility
MOBILITY =

android {
#openmp
#QMAKE_CFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp # -lgomp

NCNN_DIR = D:/ncnn-lib/ncnn-20210720-android/armeabi-v7a
# D:/ncnn-lib/ncnn-20210720-android/armeabi-v7a
# D:/ncnn-lib/android_rtti
# D:/ncnn-lib/ncnn-20210720-android-vulkan-shared/armeabi-v7a
LIBS += -L$$NCNN_DIR/lib -lncnn
INCLUDEPATH += $$NCNN_DIR/include
DEPENDPATH += $$NCNN_DIR/include
PRE_TARGETDEPS += $$NCNN_DIR/lib/libncnn.a  # .so .a


ANDROID_OPENCV = D:/opencv-4.5.3-android-sdk/OpenCV-android-sdk/sdk/native
# D:/ncnn-lib/opencv-mobile-4.5.3-android/sdk/native
# D:/opencv-4.5.3-android-sdk/OpenCV-android-sdk/sdk/native
# D:/OpenCV-android-sdk/sdk/native (gcc)
INCLUDEPATH += \
$$ANDROID_OPENCV/jni/include/opencv    \
$$ANDROID_OPENCV/jni/include/opencv2    \
$$ANDROID_OPENCV/jni/include

LIBS += $$ANDROID_OPENCV/libs/armeabi-v7a/libopencv_java4.so \
$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_highgui.a \
$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_imgproc.a \
$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_imgcodecs.a \
$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_core.a     \
$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/libIlmImf.a  \
$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/liblibjpeg-turbo.a \
#$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/liblibpng.a \
#$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/liblibtiff.a \
#$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/liblibjasper.a \
#$$ANDROID_OPENCV/3rdparty/libs/armeabi-v7a/libtbb.a \

#LIBS += $$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_highgui.a \
#$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_imgproc.a \
#$$ANDROID_OPENCV/staticlibs/armeabi-v7a/libopencv_core.a

#  psenet_lite_mbv2  yolov4-tiny-opt yolo-fastest-opt MobileNetV2-YOLOv3-Nano-coco
data.files += src/yolo-fastest-opt.bin
data.files += src/yolo-fastest-opt.param
data.files += src/test.jpg
data.path = /assets/dst/
INSTALLS += data


LIBS += -landroid
}


win32{
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp -lgomp

INCLUDEPATH += D:\OpenCVMinGW3.4.1\include
LIBS += D:\OpenCVMinGW3.4.1\bin\libopencv_*.dll

LIBS += -L$$PWD/../../ncnn-lib/winlib/ -lncnn

INCLUDEPATH += $$PWD/../../ncnn-lib/include
DEPENDPATH += $$PWD/../../ncnn-lib/include

PRE_TARGETDEPS += $$PWD/../../ncnn-lib/winlib/libncnn.a
}



contains(ANDROID_TARGET_ARCH,armeabi-v7a) {
    ANDROID_EXTRA_LIBS = \
        $$ANDROID_OPENCV/libs/armeabi-v7a/libopencv_java4.so \
#        $$NCNN_DIR/lib/libncnn.so
}

DISTFILES += \
    android/AndroidManifest.xml \
    android/gradle/wrapper/gradle-wrapper.jar \
    android/gradlew \
    android/res/values/libs.xml \
    android/build.gradle \
    android/gradle/wrapper/gradle-wrapper.properties \
    android/gradlew.bat \
    android/src/com/amin/NCNNProjs/NCNNProjs.java

ANDROID_PACKAGE_SOURCE_DIR = $$PWD/android

RESOURCES += \
    src.qrc

