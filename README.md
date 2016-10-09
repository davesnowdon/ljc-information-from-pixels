Building
========
This code depends on OpenCV built with java support and installed into the local maven repo. See build-opencv.sh for a script to build OpenCV with java support on Linux.

Use the gradle wrapper to build the demo code.

    ./gradlew build

Demos
=====

On Linux, if java does not pick up the native library from the opencv-native-310.jar you may need to set LD_LIBRARY_PATH to point to the directory containing the native library.

For brevity, the command line below assume that the java classpath already contains the required dependencies

    # Find faces
    java com.davesnowdon.ifp.Main --command find-faces --image <INPUT IMAGE> --output <OUTPUT IMAGE>


    # Find blob specified using HSV range
    java com.davesnowdon.ifp.Main --command find-blob --image src/test/resources/contains-blue-blob.jpg --low 84,80,80 --high 104,255,255 --output out.jpg


    # Blur image
    java com.davesnowdon.ifp.Main --command blur --image <INPUT IMAGE> --kernel-size $1 --output <OUTPUT IMAGE>