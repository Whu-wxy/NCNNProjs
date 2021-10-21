#include "androidsetup.h"

#ifdef Q_OS_ANDROID

#include <QAndroidJniEnvironment>
#include <QAndroidJniObject>
#include <QtAndroid>
#include <QDebug>

AndroidSetup::AndroidSetup()
{

}


QString AndroidSetup::getExternalStorageDir()
{
    QAndroidJniObject externalStorageDir = QAndroidJniObject::callStaticObjectMethod(
                "android/os/Environment",
                "getExternalStorageDirectory",
                "()Ljava/io/File;");
    return (externalStorageDir.toString());
}

QString AndroidSetup::getAppDataDir()
{
    QString szDir = getExternalStorageDir();
    szDir += QDir::separator() + QString("DetectorData");
    qDebug()<<szDir;
    QDir dir(szDir);
    if(!dir.exists())
    {
        if(dir.mkpath(szDir))
            qDebug()<<"dir made";
        else
            qDebug()<<"dir not made";
    }
    else
        qDebug()<<"dir exist";
    return szDir;
}

#endif // Q_OS_ANDROID
