#! /bin/bash

# Open CV pre-requisites
# ant 1.9.6

export OPENCV_HOME="$(pwd)/opencv-3.1.0"
cd $OPENCV_HOME
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=OFF ..
make 

mkdir ../../lib
cp lib/libopencv_java310.so ../../lib/
mv bin/opencv-310.jar ../../lib/
cd ../../lib/
mkdir -p native/linux/x86_64
cp libopencv_java310.so native/linux/x86_64/
jar -cMf opencv-native-310.jar native

# needs leiningen
lein localrepo install opencv-310.jar opencv/opencv 3.1.0
lein localrepo install opencv-native-310.jar opencv/opencv-native 3.1.0
