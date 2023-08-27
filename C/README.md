# Overview
This folder contains minimalistic C examples of the Fast-Method deblur algorithm in one and two dimensions.
This code is *not* highly optimized.  For optimized implementations, use the Java code as a reference.

# To Compile
The only dependencies are stb_image.h and stb_image_write.h for Deblur2D.c. They are included here.
`gcc Deblur1D.c -o Deblur1D.exe -O && Deblur1D.exe`
`gcc Deblur2D.c -o Deblur2D.exe -O && Deblur2D.exe`

# List of Files
- Deblur1D.c: The Fast-Method deconvolution algorithm for one-dimension.
- Deblur2D.c: The Fast-Method deconvolution algorithm for two-dimensional color images.
- stb_image.h and stb_image_write.h: library for reading and writing PNG images.

