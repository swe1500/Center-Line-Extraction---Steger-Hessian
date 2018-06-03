TARGET = ssd
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
TEMPLATE = app
SOURCES += \
    main.cpp
INCLUDEPATH += /usr/local/include \
/usr/local/include/opencv \
/usr/local/include/opencv2 \

LIBS += -L/usr/local/lib \
/usr/local/lib/*.so \

