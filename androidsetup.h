#ifndef ANDROIDSETUP_H
#define ANDROIDSETUP_H

#include <QObject>
#include <QDir>

#ifdef Q_OS_ANDROID

class AndroidSetup
{
public:
    explicit AndroidSetup();

    QString getExternalStorageDir();
    QString getAppDataDir();

signals:

public slots:
};

#endif // Q_OS_ANDROID

#endif // ANDROIDSETUP_H
