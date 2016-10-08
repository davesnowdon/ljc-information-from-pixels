#! /bin/bash

# Open CV pre-requisites
# cmake
# ant 1.9.6

BUILDOPTS="-DBUILD_SHARED_LIBS=OFF"

# disable video on MacOS since QTKit has been removed from recent versions of MacOS
BUILDOS=$(uname -s)
if [ "$BUILDOS" == "Darwin" ]; then
    BUILDOPTS="${BUILDOPTS} -DBUILD_opencv_videoio=OFF"
    echo "Disabling video"
fi


export OPENCV_HOME="$(pwd)/opencv-3.1.0"
cd $OPENCV_HOME
mkdir build
cd build
cmake ${BUILDOPTS} ..
make 

mkdir ../../lib
cp bin/opencv-310.jar ../../lib/
cp lib/libopencv_java310.so ../../lib/
cd ../../lib/
if [ "$BUILDOS" == "Darwin" ]; then
    mkdir -p native/macosx/x86_64
    cp libopencv_java310.so native/macosx/x86_64/
else
    mkdir -p native/linux/x86_64
    cp libopencv_java310.so native/linux/x86_64/
fi
jar -cMf opencv-native-310.jar native

# needs leiningen
lein localrepo install opencv-310.jar opencv/opencv 3.1.0
lein localrepo install opencv-native-310.jar opencv/opencv-native 3.1.0
