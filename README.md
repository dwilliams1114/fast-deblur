# Overview
This repo contains various implementations of the "Fast-Method" deblur algorithm, as well as a GUI for easy usage.
"Fast-Method" is described in the paper "A fast, simple, and parallelizable deconvolution algorithm for real-time applications" by Daniel Williams in SPIE.
Libraries required to run the Java application are included.
More details on the algorithm are found at https://danielwilliams.tech/fast-deblur.

# Components
This repo contains 4 components:
- GUI for performing image deblurring with various algorithms (main method in Interface.java)
- Pre-compiled Java executable of the GUI (in DeblurGUI.java)
- One dimensional usage of the Fast-Method  (main in Deblur1D.java)
- C implementations for one and two dimensions (in C folder, Deblur1D.c and Deblur2D.c)

# Setup in Eclipse IDE
To run all the main code, you'll need Java-OpenCV, Java-OpenCL, Java-OpenGL, and Java-FFMPEG bindings.  All are included in the Libraries folder.
The code was written for Windows x86_64.  If you're using something else, you'll have to find the libraries for your system.
To set up in Eclipse IDE:
1. Install Java JDK 18.0.2 or greater (but 1.8.0 works if you don't care about OpenGL)
1. Import the project (fast-deblur is an Eclipse project)
1. Right click on the project -> Properties -> Java Build Path -> Add Library -> User Library -> User Libraries -> New...
1. Create a library named "OpenCV" and click "Add JARs".  Add Libraries/JavaCV/opencv-3415.jar.
1. Expand the drop down for the newly created library.
1. Under OpenCV, expand opencv-3415.jar.
1. Click on "Native library location" and browse to "fast-deblur/JavaCV".
1. Create a library named "Java2D" and click "Add JARs".  Add all JARs in Libraries/Java2D/.
1. Expand the Java2D -> JOCL-0.2.0RC.jar
1. Click on "Native library location" and browse to "fast-deblur/Java2D".
1. Create a library named "OpenCV_FFMPEG" and click "Add JARs".  Add all JARs in Libraries/OpenCV_FFMPEG/.
1. Open the Run Configurations for the main class, Interface.java
1. Switch to the "Arguments" tab
1. Under "VM Arguments", add: `--add-exports java.base/java.lang=ALL-UNNAMED` `--add-exports java.desktop/sun.awt=ALL-UNNAMED` `--add-exports java.desktop/sun.java2d=ALL-UNNAMED`.
1. Click "Run".  It will always run with these arguments now.
1. All the errors disappear and everything works perfectly on the first try. :joy:

# Images
The GUI:
![Screenshot of the GUI](/Images/GUI_1.png)

A real photograph of text, deblurred using Fast-Method, then sharpened using the Sharpen tool:
![Blurry text on left, deblurred text on right](/Images/FastMethod_Example.png)

# Description of files
- Interface.java is the main class for the GUI.  It makes calls to most of the other files.
- ImageEffects.java creates the little effects pop-up boxes in the GUI.
- Algorithms.java contains most of the CPU-based implementations of the algorithms.
- GPUAlgorithms.java is the driver for running the algorithms on the GPU.
- FastMethod.cl is the OpenCL kernel implementation of the Fast-Method deconvolution algorithm.
- RichardsonLucy.cl is the OpenCL kernel implementation of the Richardson-Lucy deconvolution algorithm.
- DeblurOpenGL.java is the driver for performing the Fast-Method in OpenGL.
- fshader.glsl and vshader.glsl are the OpenGL implementation of the Fast-Method.
- VideoDecoder.java is the implementation for video decoding and processing using the Fast-Method.
- Deblur1D.java is the main class for running the Fast-Method in 1D.  It includes audio processing.
- WienerFilter.java is a Java-OpenCV implementation of Wiener deconvolution using the FFT.
- GPUProgram.java is a generic Java-to-OpenCL interfacing library.  (Basically glue code.)
- ShaderProgram.java is a generic Java-to-OpenGL interfacing library.  (Also just glue code.)
- The C folder contains minimalistic C implementations of the Fast-Method algorithm.
