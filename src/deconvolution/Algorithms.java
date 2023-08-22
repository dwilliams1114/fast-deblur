package deconvolution;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import gpuTesting.GPUProgram;

// This class contains CPU-based image processing algorithms

public class Algorithms {
	
	// This is the image in array form
	static float[][][] imageArray;
	
	// This is used only for GPU programs which need to reuse data many times
	static byte[] cachedByteBuffer;
	
	// Number of threads to use in calculations or rendering
	static int numThreads = 1;
	
	// Used to keep track of multithreaded algorithms
	static int threadsCompleted = 0;
	
	// Whether to use the GPU when available
	static boolean useOpenCL = true;
	
	// Whether to use the OpenGL implementation with the GLCanvas (overrides useGPU)
	static boolean useOpenGL = false;
	
	// Sqrt(2) constant
	static final float sqrt2 = (float)Math.sqrt(2);
	
	// Change the contrast of the image around a certain brightness level
	static float[][][] adjust(final float[][][] image,
			final float contrast, final float brightness, final float saturation, final float exposure) {
		
		Interface.setProcessName("Adjusting contrast");
		
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		final float newContrast = (float)(Math.pow(1.03, contrast));
		
		final float[][][] newImage = new float[3][width][height];
		
		final float modifiedExposure = (float)Math.pow(2, (exposure-1) * 0.3);
		
		// Loop over the pixels in parallel
		threadsCompleted = 0;
		for (int k = 0; k < numThreads; k++) {
			final int threadOffset = k;
			new Thread(new Runnable() {
				public void run() {
					for (int x = threadOffset; x < width; x += numThreads) {
						for (int y = 0; y < height; y++) {
							for (int j = 0; j < 3; j++) {
								newImage[j][x][y] = (image[j][x][y] - 127) * newContrast + 127 + brightness;
								if (exposure != 1) {
									newImage[j][x][y] = exposeColor(newImage[j][x][y], exposure, modifiedExposure);
								}
							}
							
							if (saturation != 1) {
								float average = (newImage[0][x][y] + newImage[1][x][y] + newImage[2][x][y])/3f;
								newImage[0][x][y] = average + (newImage[0][x][y] - average) * saturation;
								newImage[1][x][y] = average + (newImage[1][x][y] - average) * saturation;
								newImage[2][x][y] = average + (newImage[2][x][y] - average) * saturation;
							}
						}
						if (threadOffset == 0) {
							Interface.updateProgress((double)x/width);
						}
						
						// Exit early if the effect has been canceled
						if (ImageEffects.isCanceled) {
							break;
						}
					}
					
					synchronized(Algorithms.class) {
						threadsCompleted++;
					}
				}
			}).start();
		}
		
		// Wait for all of the threads to complete
		while (threadsCompleted < numThreads) {
			try {
				Thread.sleep(1);
			} catch (Exception e) {}
		}
		
		Interface.updateProgress(1);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		return newImage;
	}
	
	// This function maps old exposure to new exposure by a certain amount
	static float exposeColor(float val, float amount, float modifiedAmount) {
		if (val <= 260) {
			return (float)(1 - Math.pow(1 - val/260, modifiedAmount)) * 255;
		} else {
			if (amount <= 0) {
				amount = 20;
			}
			return (val - 260) / (float)Math.pow(2, amount) + 255;
		}
	}
	
	// Driver method for the fastMethod function
	static float[][][] fastMethodSwitch(final float[][][] image,
			final float amountOffset, final float radius,
			final int iterations, boolean commit) {
		
		if (useOpenGL) { // Render with OpenGL

			int width = image[0].length;
			int height = image[0][0].length;
			
			// Enable deblurring only if we are using an image effects dialog
			DeblurOpenGL.setDeblurEnabled(true);
			
			// Set the image parameters
			DeblurOpenGL.setImageParameters(width, height, radius);
			
			if (cachedByteBuffer == null) {
				// Deconvolve and display using OpenGL
				cachedByteBuffer = extractByteArray(Interface.previewImage);
				
				// Copy from the float image to the preview image
				for (int x = 0; x < width; x++) {
					for (int y = 0; y < height; y++) {
						int i = (y * width + x) * 3;
						cachedByteBuffer[i + 0] = (byte)clamp(image[2][x][y]);
						cachedByteBuffer[i + 1] = (byte)clamp(image[1][x][y]);
						cachedByteBuffer[i + 2] = (byte)clamp(image[0][x][y]);
					}
				}
			}
			
			// No deblurring here:
			// The actual deblurring takes place in the Interface.redrawPreviewImage() function on render.
			
			// Cannot save if using OpenGL for rendering
			if (commit) {
				Interface.displayOpenGLSaveWarning();
			}
			
			// The image is drawn directly to the screen, so we don't need to return it
			return null;
			
		} else if (useOpenCL) { // Render with OpenCL (GPU)
			
			int width = image[0].length;
			int height = image[0][0].length;
			
			// Only convert the image if it hasn't already been done
			if (!GPUProgram.isInitialized() || cachedByteBuffer == null) {
				GPUProgram.initializeGPU();
				
				// Convert the preview image to 3BYTE_BGR format
				Interface.previewImage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
				byte[] buffer = extractByteArray(Interface.previewImage);
				
				// Copy from the float image to the preview image
				for (int x = 0; x < width; x++) {
					for (int y = 0; y < height; y++) {
						int i = (y * width + x) * 3;
						buffer[i + 0] = (byte)clamp(image[2][x][y]);
						buffer[i + 1] = (byte)clamp(image[1][x][y]);
						buffer[i + 2] = (byte)clamp(image[0][x][y]);
					}
				}
				
				// TODO why aren't we writing directly to "cachedByteBuffer" above?
				// Make a copy of the image
				cachedByteBuffer = new byte[buffer.length];
				System.arraycopy(buffer, 0, cachedByteBuffer, 0, buffer.length);
			}
			
			return GPUAlgorithms.fastMethodGPU(cachedByteBuffer, iterations,
					width, height, amountOffset, radius, commit);
			
		} else { // Render pixels using CPU
			
			Interface.setProcessName("Deblurring");
			final long startTime = System.currentTimeMillis();
			
			// This contains the most accurate image on each iteration
			float[][][] newApproximation = image;
			for (int i = 0; i < iterations; i++) {
				newApproximation = fastMethod(image, newApproximation,
						amountOffset, radius);
			}
			
			Interface.setProcessName("Deblurring (" + (System.currentTimeMillis() - startTime) + "ms)");
			Interface.updateProgress(1);
			
			return newApproximation;
		}
	}
	
	// Deblur using a new more precise iterative technique
	// Programmed by Daniel Williams on May 4, 2019
	// Array is accessed as image[rgb][x index][y index]
	// REMARKS:
	// This is the most accurate and fast algorithm.  It is best.
	// Subtracts two circles to compute the gradient, then suppresses ringing with an outer circle.
	// The mathematical proof backs up this technique very well.
	static float[][][] fastMethod(final float[][][] originalImage,
			final float[][][] newApproximation, final float amountOffset, final float radius) {
		
		final int width = originalImage[0].length;
		final int height = originalImage[0][0].length;
		
		// Generate the blur kernel
		final int[] coords1 = generateCircle(radius);
		final int[] coords2 = generateCircle(radius + 1);
		final int[] coordsOuter = generateCircleQuarterDensity(radius * 2); // Skip 75% of pixels for speed.
		final int coords1Count = coords1.length/2;
		final int coords2Count = coords2.length/2;
		final int coordsOuterCount = coordsOuter.length/2;
		
		// Used to ensure that the weight of the inner ring is the same as the weight of the outer ring
		final float innerToOuterRatio =  (float)coords1Count / coords2Count;
		
		final float innerMult = amountOffset / 2.0f * 0.67f; // Why 0.67 here?
		
		// Create the image to write to
		final float[][][] outImage = new float[3][width][height];
		
		// Start this in a number of parallel threads
		final Thread[] threads = new Thread[numThreads];
		for (int k = 0; k < numThreads; k++) {
			final int threadOffset = k;
			threads[k] = new Thread(new Runnable() {
				public void run() {

					float gradientR = 0;
					float gradientG = 0;
					float gradientB = 0;
					
					float outerR = 0;
					float outerG = 0;
					float outerB = 0;
					
					int x2 = 0;
					int y2 = 0;
					
					// Convolve over the image with the data from the two coordinate lists
					for (int x = threadOffset; x < width; x += numThreads) {
						
						for (int y = 0; y < height; y++) {
							
							gradientR = 0;
							gradientG = 0;
							gradientB = 0;
							
							// Integrate over the inner negative ring (radius r+1)
							for (int i = 0; i < coords2Count; i++) {
								x2 = coords2[i * 2 + 0] + x;
								y2 = coords2[i * 2 + 1] + y;
								
								// Clamp coordinates to the image bounds
								x2 = clamp(x2, 0, width-1);
								y2 = clamp(y2, 0, height-1);
								
								gradientR -= originalImage[0][x2][y2];
								gradientG -= originalImage[1][x2][y2];
								gradientB -= originalImage[2][x2][y2];
							}
							
							// Scale the negative ring to the same weight as the inner positive ring
							gradientR *= innerToOuterRatio;
							gradientG *= innerToOuterRatio;
							gradientB *= innerToOuterRatio;
							
							// Integrate over the inner positive ring (radius r)
							for (int i = 0; i < coords1Count; i++) {
								x2 = coords1[i * 2 + 0] + x;
								y2 = coords1[i * 2 + 1] + y;
								
								// Clamp coordinates to the image bounds
								x2 = clamp(x2, 0, width-1);
								y2 = clamp(y2, 0, height-1);
								
								gradientR += originalImage[0][x2][y2];
								gradientG += originalImage[1][x2][y2];
								gradientB += originalImage[2][x2][y2];
							}
							
							outerR = 0;
							outerG = 0;
							outerB = 0;
							
							// Integrate over the outer positive ring (radius 2r)
							for (int i = 0; i < coordsOuterCount; i++) {
								x2 = coordsOuter[i * 2 + 0] + x;
								y2 = coordsOuter[i * 2 + 1] + y;
								
								// Clamp coordinates to the image bounds
								x2 = clamp(x2, 0, width-1);
								y2 = clamp(y2, 0, height-1);
								
								outerR += newApproximation[0][x2][y2];
								outerG += newApproximation[1][x2][y2];
								outerB += newApproximation[2][x2][y2];
							}
							
							// Calculate the new color of this pixel.
							// Inner ring contributes (+2*pi*r * f(x,y)) * 255
							// Outer ring contributes (-2*pi*r * f(x,y)) * 255
							// Outermost ring contributes (+1.0 * f(x,y)) * 255
							outImage[0][x][y] = innerMult * gradientR + outerR / coordsOuterCount;
							outImage[1][x][y] = innerMult * gradientG + outerG / coordsOuterCount;
							outImage[2][x][y] = innerMult * gradientB + outerB / coordsOuterCount;
						}
						
						// Update the progress
						if ((threadOffset == 0) && (x % 32 == 0)) {
							Interface.updateProgress((double)x/width);
						}

						// Exit early if the effect has been canceled
						if (ImageEffects.isCanceled) {
							break;
						}
					}
				}
			});
		}
		
		// Start all the threads
		for (int k = 0; k < numThreads; k++) {
			threads[k].start();
		}
		
		// Wait for all of the threads to complete
		try {
			for (int k = 0; k < numThreads; k++) {
				threads[k].join();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Interface.updateProgress(1);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		return outImage;
	}
	
	// This function switches between the GPU and CPU
	static float[][][] richardsonLucySwitch(final float[][][] image,
					float radius, int iterations, boolean commit) {
		if (useOpenCL) {
			GPUProgram.initializeGPU();
			return GPUAlgorithms.deblurRichardsonLucyGPU(image, radius, iterations, commit);
		} else {
			return richardsonLucy(image, radius, iterations);
		}
	}
	
	// Deblur using Richardson-Lucy algorithm
	static float[][][] richardsonLucy(final float[][][] image, float radius, final int iterations) {
		
		Interface.setProcessName("Deblurring");
		
		final long startTime = System.currentTimeMillis();
		
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		// Generate a disk kernel
		// This must have an odd width and height
		final float[][] kernel = new float[(int)(radius + 0.2) * 2 + 1][(int)(radius + 0.2) * 2 + 1];
		final int offset = kernel.length / 2;
		for (int x = 0; x < kernel.length; x++) {
			for (int y = 0; y < kernel[0].length; y++) {
				float dist = (float)Math.hypot(x - offset, y - offset);
				float intensity = (radius - dist + 0.5f);
				if (intensity <= 0) {
					kernel[x][y] = 0;
				} else {
					kernel[x][y] = 1;
				}
			}
		}
		
		// Save this for a preview
		Interface.lastKernel = kernel;
		
		// Create the image that stores the previous image approximation
		//final float[][][] firstApproximation = Algorithms.fastMethod(Algorithms.copyImage(image), Algorithms.copyImage(image), 1, radius);
		final float[][][] newImage = new float[3][width][height];
		for (int x = 0; x < width; x += 1) {
			for (int y = 0; y < height; y++) {
				newImage[0][x][y] = image[0][x][y];
				newImage[1][x][y] = image[1][x][y];
				newImage[2][x][y] = image[2][x][y];
			}
		}
		
		// Create the image to perform calculations on in the middle of the calculation
		final float[][][] middleBlur = new float[3][width][height];

		// Run the whole algorithm many times
		for (int i = 0; i < iterations; i++) {
			
			// middleBlur = image / blur(oldImage)
			// Run this in parallel
			threadsCompleted = 0;
			for (int k = 0; k < numThreads; k++) {
				final int threadOffset = k;
				new Thread(new Runnable() {
					public void run() {
						
						// Convolve over the image
						for (int x = threadOffset; x < width; x += numThreads) {
							for (int y = 0; y < height; y++) {
								// Sum the pixels around this pixel
								float r = 0;
								float g = 0;
								float b = 0;
								int sampleCount = 0;
								for (int x2 = -offset; x2 <= offset; x2++) {
									if (x2 + x < 0 || x2 + x >= width) {
										continue;
									}
									for (int y2 =  -offset; y2 <= offset; y2++) {
										if (y2 + y < 0 || y2 + y >= height ||
												kernel[x2 + offset][y2 + offset] == 0) {
											continue;
										}
										
										r += newImage[0][x + x2][y + y2];
										g += newImage[1][x + x2][y + y2];
										b += newImage[2][x + x2][y + y2];
										sampleCount++;
									}
								}
								
								r /= sampleCount;
								g /= sampleCount;
								b /= sampleCount;
								
								if (r < 0.01f) {
									r = 0.01f;
								}
								if (g < 0.01f) {
									g = 0.01f;
								}
								if (b < 0.01f) {
									b = 0.01f;
								}
								
								middleBlur[0][x][y] = image[0][x][y] / r;
								middleBlur[1][x][y] = image[1][x][y] / g;
								middleBlur[2][x][y] = image[2][x][y] / b;
							}
							
							// Exit early if the effect has been canceled
							if (ImageEffects.isCanceled) {
								break;
							}
						}
						
						synchronized(Algorithms.class) {
							threadsCompleted++;
						}
					}
				}).start();
			}
			
			// Wait for all of the threads to complete
			while (threadsCompleted < numThreads) {
				try {
					Thread.sleep(1);
				} catch (Exception e) {}
			}
			
			// newImage *= blur(middleBlur)
			// Run this in parallel
			threadsCompleted = 0;
			for (int k = 0; k < numThreads; k++) {
				final int threadOffset = k;
				new Thread(new Runnable() {
					public void run() {
						
						// Convolve over the image
						for (int x = threadOffset; x < width; x += numThreads) {
							for (int y = 0; y < height; y++) {
								// Sum the pixels around this pixel
								float r = 0;
								float g = 0;
								float b = 0;
								int sampleCount = 0;
								for (int x2 = -offset; x2 <= offset; x2++) {
									if (x2 + x < 0 || x2 + x >= width) {
										continue;
									}
									for (int y2 =  -offset; y2 <= offset; y2++) {
										if (y2 + y < 0 || y2 + y >= height ||
												kernel[x2 + offset][y2 + offset] == 0) {
											continue;
										}
										
										r += middleBlur[0][x + x2][y + y2];
										g += middleBlur[1][x + x2][y + y2];
										b += middleBlur[2][x + x2][y + y2];
										sampleCount++;
									}
								}
								
								newImage[0][x][y] *= (r / sampleCount);
								newImage[1][x][y] *= (g / sampleCount);
								newImage[2][x][y] *= (b / sampleCount);
							}
							
							// Exit early if the effect has been canceled
							if (ImageEffects.isCanceled) {
								break;
							}
						}
						
						synchronized(Algorithms.class) {
							threadsCompleted++;
						}
					}
				}).start();
			}
			
			// Wait for all of the threads to complete
			while (threadsCompleted < numThreads) {
				try {
					Thread.sleep(1);
				} catch (Exception e) {}
			}
			
			Interface.updateProgress((double)i / iterations);
			
			// Exit early if the effect has been canceled
			if (ImageEffects.isCanceled) {
				return null;
			}
		}
		
		Interface.setProcessName("Deblurring (" + (System.currentTimeMillis() - startTime) + "ms)");
		Interface.updateProgress(1);
		
		return newImage;
	}
	
	// Deblur the image using the proven method in one dimension.
	// Array length must be at least radius*2 + 1.
	static float[] deblur1D(final float[] inputSamples, final int radius) {
		final float[] outputSamples = new float[inputSamples.length];
		
		final float innerRingScale = (2f * radius + 1) / 2f;
		final float outerRingScale = 0.5f;
		
		for (int i = 0; i < radius*2 + 1; i++) {
			outputSamples[i] = innerRingScale * (
					+ inputSamples[Math.max(i - radius, 0)]
					- inputSamples[Math.max(i - radius - 1, 0)]
					+ inputSamples[i + radius]
					- inputSamples[i + radius + 1])
					+(inputSamples[Math.max(i - 2 * radius - 1, 0)]
					+ inputSamples[i + 2 * radius + 1]) * outerRingScale;
		}
		
		for (int i = radius*2 + 1; i < inputSamples.length - radius*2 - 1; i++) {
			outputSamples[i] = innerRingScale * (
					+ inputSamples[i - radius]
					- inputSamples[i - radius - 1]
					+ inputSamples[i + radius]
					- inputSamples[i + radius + 1])
					+(inputSamples[i - 2 * radius - 1]
					+ inputSamples[i + 2 * radius + 1]) * outerRingScale;
		}
		
		for (int i = inputSamples.length - radius*2 - 1; i < inputSamples.length; i++) {
			outputSamples[i] = innerRingScale * (
					+ inputSamples[i - radius]
					- inputSamples[i - radius - 1]
					+ inputSamples[Math.min(i + radius, inputSamples.length-1)]
					- inputSamples[Math.min(i + radius + 1, inputSamples.length-1)])
					+(inputSamples[i - 2 * radius - 1]
					+ inputSamples[Math.min(i + 2 * radius + 1, inputSamples.length-1)]) * outerRingScale;
		}
		
		return outputSamples;
	}
	
	// Deblur the image using the proven method in one dimension, iterated N times.
	// Array length must be at least radius*2 + 1.
	static float[] deblur1Dv2(final float[] inputSamples, final int radius, final int ITERATIONS) {
		
		final float innerRingScale = (2f * radius + 1) / 2f;
		final float outerRingScale = 0.5f;
		
		float[] outputSamples = null;
		float[] previousIteration = inputSamples;
		//float[] previousIteration = new float[inputSamples.length];
		
		for (int j = 0; j < ITERATIONS; j++) {
			outputSamples = new float[inputSamples.length];
			
			for (int i = 0; i < radius*4 + 2; i++) {
				outputSamples[i] = innerRingScale * (
						+ inputSamples[Math.max(i - radius, 0)]
						- inputSamples[Math.max(i - radius - 1, 0)]
						+ inputSamples[i + radius]
						- inputSamples[i + radius + 1])
						+(previousIteration[Math.max(i - 2 * radius - 1, 0)]
						+ previousIteration[i + 2 * radius + 1]) * outerRingScale;
			}
			for (int i = radius*4 + 2; i < inputSamples.length - radius*4 - 2; i++) {
				outputSamples[i] = innerRingScale * (
						+ inputSamples[i - radius]
						- inputSamples[i - radius - 1]
						+ inputSamples[i + radius]
						- inputSamples[i + radius + 1])
						+(previousIteration[i - 2 * radius - 1]
						+ previousIteration[i + 2 * radius + 1]) * outerRingScale;
			}
			for (int i = inputSamples.length - radius*4 - 2; i < inputSamples.length; i++) {
				outputSamples[i] = innerRingScale * (
						+ inputSamples[i - radius]
						- inputSamples[i - radius - 1]
						+ inputSamples[Math.min(i + radius, inputSamples.length-1)]
						- inputSamples[Math.min(i + radius + 1, inputSamples.length-1)])
						+(previousIteration[i - 2 * radius - 1]
						+ previousIteration[Math.min(i + 2 * radius + 1, inputSamples.length-1)]) * outerRingScale;
			}
			previousIteration = outputSamples;
		}
		
		return outputSamples;
	}
	
	// Blur an array of samples in 1D using a disk blur.
	static float[] blur1D(final float[] inputSamples, final int radius) {
		final float[] outputSamples = new float[inputSamples.length];
		
		for (int i = 0; i < inputSamples.length; i++) {
			int sampleCount = 0;
			float total = 0;
			for (int j = Math.max(i - radius, 0); j <= Math.min(i + radius, inputSamples.length-1); j++) {
				total += inputSamples[j];
				sampleCount++;
			}
			outputSamples[i] = total / sampleCount;
		}
		
		return outputSamples;
	}
	
	// Deblur the image using the inverse blur kernel (also known as 'sharpen' effect).
	// Radius is a minimum of 0.5.
	// The blur algorithm only removes brokeh blurs
	static float[][][] sharpen(final float[][][] image, float weight, final float radius) {
		Interface.setProcessName("Sharpening");
		
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		final float newWeight = weight / radius / radius;
		
		// This must have an odd width and height
		final float[][] kernel = new float[(int)(radius + 0.4) * 2 + 1][(int)(radius + 0.4) * 2 + 1];
		final int offset = kernel.length / 2;
		for (int x = 0; x < kernel.length; x++) {
			for (int y = 0; y < kernel[0].length; y++) {
				float dist = (float)Math.hypot(x - offset, y - offset);
				float intensity = (radius - dist + 0.5f) * 1.6f;
				if (intensity < 0) {
					intensity = 0;
				} else if (intensity > 1) {
					intensity = 1;
				}
				kernel[x][y] = intensity;
			}
		}
		kernel[offset][offset] = 0;
		
		// Save the kernel for display
		Interface.lastKernel = kernel;
		
		final float[][][] newImage = new float[3][width][height];
		// Run this in parallel
		threadsCompleted = 0;
		for (int k = 0; k < numThreads; k++) {
			final int threadOffset = k;
			new Thread(new Runnable() {
				public void run() {
					
					// Convolve over the image
					for (int x = threadOffset; x < width; x += numThreads) {
						for (int y = 0; y < height; y++) {
							// (original pixel) = ((this pixel) - Sum(weight[i] * pixel[i])) / (this pixel weight)
							
							float weightedR = 0;
							float weightedG = 0;
							float weightedB = 0;
							float sum = 1;
							for (int x2 = -offset; x2 <= offset; x2++) {
								if (x2 + x < 0 || x2 + x >= width) {
									continue;
								}
								for (int y2 =  -offset; y2 <= offset; y2++) {
									if (y2 + y < 0 || y2 + y >= height ||
											kernel[x2 + offset][y2 + offset] == 0) {
										continue;
									}
									
									weightedR += kernel[x2 + offset][y2 + offset] * image[0][x + x2][y + y2];
									weightedG += kernel[x2 + offset][y2 + offset] * image[1][x + x2][y + y2];
									weightedB += kernel[x2 + offset][y2 + offset] * image[2][x + x2][y + y2];
									sum += kernel[x2 + offset][y2 + offset];
								}
							}
							
							weightedR = (image[0][x][y] - weightedR / sum) * sum;
							weightedG = (image[1][x][y] - weightedG / sum) * sum;
							weightedB = (image[2][x][y] - weightedB / sum) * sum;
							
							newImage[0][x][y] = weightedR * newWeight + image[0][x][y] * (1 - newWeight);
							newImage[1][x][y] = weightedG * newWeight + image[1][x][y] * (1 - newWeight);
							newImage[2][x][y] = weightedB * newWeight + image[2][x][y] * (1 - newWeight);
						}
						
						if (threadOffset == 0) {
							Interface.updateProgress((double)x/width);
						}
						
						// Exit early if the effect has been canceled
						if (ImageEffects.isCanceled) {
							break;
						}
					}
					
					synchronized(Algorithms.class) {
						threadsCompleted++;
					}
				}
			}).start();
		}

		// Wait for all of the threads to complete
		while (threadsCompleted < numThreads) {
			try {
				Thread.sleep(1);
			} catch (Exception e) {}
		}
		
		Interface.updateProgress(1);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		return newImage;
	}
	
	// Function for circle-generation using Midpoint Circle algorithm (floating-point version).
	// This works for r up to about 500 for granularity 0.01.
	static int[] generateCircle(float r) {
		
		// Compute an overestimate for the number of pixels that will be in this circle.
		// (This is usually spot-on)
		int pixelCountOverestimate;
		if (r <= 0.4f) {
			pixelCountOverestimate = 8;
		} else {
			pixelCountOverestimate = 8 * (int)Math.round(1.4142157f * r - 9.983285E-5f);
		}
        
		int[] coords = new int[pixelCountOverestimate];
		float x = r;
		int y = 0;
	    int i = 0;
		int rRounded = Math.round(r);
		
		// Place the top, bottom, left, right points
		coords[i++] =  rRounded;
	    coords[i++] =  0;
		coords[i++] = -rRounded;
		coords[i++] =  0;
		
		coords[i++] =  0;
		coords[i++] =  rRounded;
		coords[i++] =  0;
		coords[i++] =  -rRounded;
		
	    while (i < coords.length) {
	        x = (float)Math.sqrt(x*x - 2*y - 1);
	        y++;
	        
	        int xRounded = Math.round(x);
	        if (xRounded == 0) {
	        	break;
	        }
	        if (xRounded < y) {
	        	break;
	        }

	        coords[i++] =  xRounded;
		    coords[i++] =  y;
			coords[i++] = -xRounded;
			coords[i++] =  y;
			coords[i++] =  xRounded;
			coords[i++] = -y;
			coords[i++] = -xRounded;
			coords[i++] = -y;
			
			// if x == y, then further points would be duplicates
			if (xRounded == y) {
				break;
			}
			
			coords[i++] =  y;
			coords[i++] =  xRounded;
			coords[i++] = -y;
			coords[i++] =  xRounded;
			coords[i++] =  y;
			coords[i++] = -xRounded;
			coords[i++] = -y;
			coords[i++] = -xRounded;
	    }
	    
	    // If the actual length didn't match the expected length, then
	    // copy the data to new array of the proper length.
	    if (i != pixelCountOverestimate) {
		    int[] coordsNew = new int[i];
		    System.arraycopy(coords, 0, coordsNew, 0, i);
		    return coordsNew;
	    } else {
	    	return coords;
	    }
	}
	
	// Function for circle-generation using Midpoint Circle algorithm (floating-point version)
	// Records every fourth point.
	// This works for r up to about 500 for granularity 0.01
	static int[] generateCircleQuarterDensity(float r) {
		
		// Compute an overestimate for the number of pixels that will be in this circle.
		// (This is usually spot-on)
		int pixelCountOverestimate;
		if (r < 5) {
			pixelCountOverestimate = 8;
		} else {
			pixelCountOverestimate = 8 + 16 * (int)Math.floor(sqrt2 / 8 * r + 0.06271961f);
			if (2 * sqrt2 * r + 7.004713f < pixelCountOverestimate) {
				pixelCountOverestimate -= 8;
			}
		}
		
		int[] coords = new int[pixelCountOverestimate];
		float x = r;
		int y = 0;
	    int i = 0;
		int rRounded = Math.round(r);
		
		// Place the top, bottom, left, right points
		coords[i++] =  rRounded;
	    coords[i++] =  0;
		coords[i++] = -rRounded;
		coords[i++] =  0;
		
		coords[i++] =  0;
		coords[i++] =  rRounded;
		coords[i++] =  0;
		coords[i++] =  -rRounded;
		
	    while (i < coords.length) {
	        x = (float)Math.sqrt(x*x - 2*y - 1);
	        y++;
	        
	        int xRounded = Math.round(x);
	        if (xRounded == 0) {
	        	break;
	        }
	        if (xRounded < y) {
	        	break;
	        }
	        
	        if (y % 4 == 0) {
		        coords[i++] =  xRounded;
			    coords[i++] =  y;
				coords[i++] = -xRounded;
				coords[i++] =  y;
				coords[i++] =  xRounded;
				coords[i++] = -y;
				coords[i++] = -xRounded;
				coords[i++] = -y;
				
				// if x == y, then further points would be duplicates
				if (xRounded == y) {
					break;
				}
				
				coords[i++] =  y;
				coords[i++] =  xRounded;
				coords[i++] = -y;
				coords[i++] =  xRounded;
				coords[i++] =  y;
				coords[i++] = -xRounded;
				coords[i++] = -y;
				coords[i++] = -xRounded;
	        }
	    }
	    
	    // If the actual length didn't match the expected length, then
	    // copy the data to new array of the proper length.
	    if (i != pixelCountOverestimate) {
		    int[] coordsNew = new int[i];
		    System.arraycopy(coords, 0, coordsNew, 0, i);
		    return coordsNew;
	    } else {
	    	return coords;
	    }
	}
	
	// Generate two arrays of coordinates representing the inner and outer edge of a circle.
	// Inside ring has values of 1.0, outside ring has values of approx. -0.99
	// Sum is 0, sum of absolute values is 1.
	// Returns: Object[]{coordinates, weights}
	// Arrays are indexed by: arr[i*2]=x, arr[i*2+1]=y.
	static Object[] generateCircleKernel(final float radius, final boolean saveKernel) {
		
		// Generate the deblurring kernel
		// This must have an odd width and height
		final int width = (int)(radius + 1.25) * 2 + 1;
		final boolean[][] kernel = new boolean[width][width];
		int offset = width / 2;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < width; y++) {
				float dist = (float)Math.hypot(x - offset, y - offset);
				float intensity = (radius - dist + 0.5f);
				if (intensity > 0) {
					kernel[x][y] = true;
				}
			}
		}
		
		if (saveKernel) {
			// Save this for a preview
			Interface.lastKernel = boolArrayToFloat2D(kernel);
		}
		
		// Over-estimate the number of pixels in the inner and outer rings
		final int numPixelsOverestimate = (int)(2 * Math.PI * (radius+1) + 1);
		final int[] innerCoords = new int[numPixelsOverestimate * 2];
		final int[] outerCoords = new int[numPixelsOverestimate * 2];
		int numInnerRingPixels = 0;
		int numOuterRingPixels = 0;

		// Iterate over each pixel in the kernel and determine whether it is part of the inner ring
		for (int x = 0; x < kernel.length; x++) {
			for (int y = 0; y < kernel[0].length; y++) {
				
				// If this pixel is ON, and an adjacent pixel is OFF, then this is part of the inner ring
				if (kernel[x][y] && (
						(x > 0 && !kernel[x-1][y]) ||
						(x < width-1 && !kernel[x+1][y]) ||
						(y > 0 && !kernel[x][y-1]) ||
						(y < width-1 && !kernel[x][y+1]))) {
					
					innerCoords[numInnerRingPixels * 2 + 0] = x;
					innerCoords[numInnerRingPixels * 2 + 1] = y;
					numInnerRingPixels++;
				}
			}
		}
		
		// Iterate over each pixel in the kernel and determine whether it is part of the outer ring
		for (int x = 0; x < kernel.length; x++) {
			for (int y = 0; y < kernel[0].length; y++) {
				
				// If this pixel is OFF, and an adjacent pixel is ON, then this is part of the outer ring
				if (!kernel[x][y] && (
						(x > 0 && kernel[x-1][y]) ||
						(x < width-1 && kernel[x+1][y]) ||
						(y > 0 && kernel[x][y-1]) ||
						(y < width-1 && kernel[x][y+1]))) {
					
					outerCoords[numOuterRingPixels * 2 + 0] = x;
					outerCoords[numOuterRingPixels * 2 + 1] = y;
					numOuterRingPixels++;
				}
			}
		}
		
		final int totalNumPixels = numOuterRingPixels + numInnerRingPixels;
		
		// Even indices are x coordinates, odd indices are y coordinates
		final int[] coords = new int[totalNumPixels * 2];
		
		final float[] weights = new float[totalNumPixels];
		
		// Weights are computed according to the following constraints:
		// sum(all weights) = 0
		// sum(inner ring) = -sum(outer ring)
		// All inner weights are 1.0
		// All outer weights are about -0.99
		
		// Mathematical notes:
		// numInnerRingPixels = 2*pi*r * 0.9001
		// numOuterRingPixels = 2*pi*(r+e) * 0.9001
		
		// Fill in the new arrays with the values from the kernel
		for (int i = 0; i < numInnerRingPixels; i++) {
			weights[i] = 1.0f;
			coords[i * 2 + 0] = innerCoords[i * 2 + 0] - width/2;
			coords[i * 2 + 1] = innerCoords[i * 2 + 1] - width/2;
		}
		for (int i = 0; i < numOuterRingPixels; i++) {
			int i2 = i + numInnerRingPixels; // Shift index past the first set of numbers
			weights[i2] = -(float)numInnerRingPixels / numOuterRingPixels; // Effectively multiplying the outer ring by -r/(r+e)
			coords[i2 * 2 + 0] = outerCoords[i * 2 + 0] - width/2;
			coords[i2 * 2 + 1] = outerCoords[i * 2 + 1] - width/2;
		}
		return new Object[] {weights, coords};
	}
	
	// Generate an array of coordinates representing the outer edge of a circle only.
	// Returns: int[] coords array
	static int[] generateOuterCircleKernel(final float radius, final boolean saveKernel) {
		// Generate the outer ring correction kernel
		final Object[] arrays = generateCircleKernel(radius, saveKernel);
		final float[] weights = (float[])arrays[0];
		final int[] coords = (int[])arrays[1];
		
		// Remove the inner (positive) ring
		int positiveCount = 0;
		for (int i = 0; i < weights.length; i++) {
			if (weights[i] >= 0) {
				weights[i] = 0;
			} else {
				positiveCount++;
			}
		}
		
		final int[] newCoords = new int[positiveCount * 2];
		int index = 0;
		for (int i = 0; i < weights.length; i++) {
			if (weights[i] != 0) {
				newCoords[index * 2 + 0] = coords[i * 2 + 0];
				newCoords[index * 2 + 1] = coords[i * 2 + 1];
				index++;
			}
		}
		
		return newCoords;
	}
	
	// Perform a disk blur on the given image
	static float[][][] diskBlur(final float[][][] originalImage, final double radius) {
		final int width = originalImage[0].length;
		final int height = originalImage[0][0].length;
		final float[][][] newImage = new float[3][width][height];

		final long startTime = System.currentTimeMillis();
		final int radiusMax = (int)Math.ceil(radius + 0.5);
		
		Interface.setProcessName("Blurring");
		
		// Blur in parallel threads
		final Thread[] threads = new Thread[numThreads];
		for (int k = 0; k < numThreads; k++) {
			final int threadOffset = k;
			threads[k] = new Thread(new Runnable() {
				public void run() {
					
					// Convolve over the image with a disk kernel
					for (int x = threadOffset; x < width; x += numThreads) {
						for (int y = 0; y < height; y++) {
							
							float sumR = 0;
							float sumG = 0;
							float sumB = 0;
							int count = 0;
							
							// Sum over all the pixels inside the disk
							for (int i = -radiusMax; i <= radiusMax; i++) {
								for (int j = -radiusMax; j <= radiusMax; j++) {
									int xIndex = x + i;
									int yIndex = y + j;
									if (xIndex >= 0 && xIndex < width && yIndex >= 0 && yIndex < height) {
										if (Math.sqrt(i * i + j * j) <= radius + 0.5) {
											sumR += originalImage[0][xIndex][yIndex];
											sumG += originalImage[1][xIndex][yIndex];
											sumB += originalImage[2][xIndex][yIndex];
											count++;
										}
									}
								}
							}
							
							newImage[0][x][y] = sumR / count;
							newImage[1][x][y] = sumG / count;
							newImage[2][x][y] = sumB / count;
						} // end-for y
						
						// Update the progress
						if ((threadOffset == 0) && (x % 8 == 0)) {
							Interface.updateProgress((double)x/width);
						}
						
						// Exit early if the effect has been canceled
						if (ImageEffects.isCanceled) {
							break;
						}
					} // end-for x
				}
			}); // end-thread
		}
		
		// Start all the threads
		for (int k = 0; k < numThreads; k++) {
			threads[k].start();
		}
		
		// Wait for all of the threads to complete
		try {
			for (int k = 0; k < numThreads; k++) {
				threads[k].join();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Interface.setProcessName("Blurring (" + (System.currentTimeMillis() - startTime) + "ms)");
		Interface.updateProgress(1);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		return newImage;
	}
	
	// Take an image and convert it to an rgb float array.
	// This method does not work with some highly compressed PNG images
	static float[][][] imageToArray(BufferedImage image) {
		// Extract colors
		final byte[] pixels = extractByteArray(image);
		final int width = image.getWidth();
		final int height = image.getHeight();
		
		final float[][][] newImage = new float[3][width][height];
		
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int i = (y * width + x) * 3;
				newImage[0][x][y] = pixels[i + 2] & 0xff; // red
				newImage[1][x][y] = pixels[i + 1] & 0xff; // green
				newImage[2][x][y] = pixels[i + 0] & 0xff; // blue
			}
		}
		
		return newImage;
	}
	
	// Return the normalized root mean square error between the two given images. (Normalized by brightness)
	// Images must be the same size.
	static double normalizedRootMeanSquareError(final float[][][] image1, final float[][][] image2) {
		
		double totalSquareError = 0;
		final int width = image1[0].length;
		final int height = image2[0][0].length;
		
		// Iterate over each color channel
		for (int channel = 0; channel < 3; channel++) {
			float avg1 = 0;
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					avg1 += image1[channel][x][y];
				}
			}
			avg1 /= width * height;
			
			float avg2 = 0;
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					avg2 += image2[channel][x][y];
				}
			}
			avg2 /= width * height;
			
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					double error = (image1[channel][x][y] - avg1) - (image2[channel][x][y] - avg2);
					totalSquareError += error * error;
				}
			}
		}
		
		// Divide by total number of samples
		totalSquareError /= (width * height * 3);
		
		return Math.sqrt(totalSquareError);
	}
	
	// Return a copy of the given image array
	static float[][][] copyImage(final float[][][] image) {
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		final float[][][] newImage = new float[3][width][height];
		
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				newImage[0][x][y] = image[0][x][y];
				newImage[1][x][y] = image[1][x][y];
				newImage[2][x][y] = image[2][x][y];
			}
		}
		
		return newImage;
	}
	
	// Take an array and convert it to a BufferedImage
	static BufferedImage arrayToImage(final float[][][] image) {
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		final BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		final byte[] bufferData = extractByteArray(newImage);
		
		// Loop over the pixels in parallel
		threadsCompleted = 0;
		for (int k = 0; k < numThreads; k++) {
			final int threadOffset = k;
			new Thread(new Runnable() {
				public void run() {
					for (int x = threadOffset; x < width; x += numThreads) {
						for (int y = 0; y < height; y++) {
							final int r = clamp(image[0][x][y]);
							final int g = clamp(image[1][x][y]);
							final int b = clamp(image[2][x][y]);
							final int i = (y * width + x) * 3;
							bufferData[i + 0] = (byte)b;
							bufferData[i + 1] = (byte)g;
							bufferData[i + 2] = (byte)r;
						}
					}
					
					synchronized(Algorithms.class) {
						threadsCompleted++;
					}
				}
			}).start();
		}
		
		// Wait for all of the threads to complete
		while (threadsCompleted < numThreads) {
			try {
				Thread.sleep(1);
			} catch (Exception e) {}
		}
		
		return newImage;
	}
	
	// Convert a 2D boolean array to a 2D float array where false=0 and true=1
	static float[][] boolArrayToFloat2D(final boolean[][] boolArr) {
		final float[][] floatArr = new float[boolArr.length][boolArr[0].length];
		for (int x = 0; x < boolArr.length; x++) {
			for (int y = 0; y < boolArr.length; y++) {
				if (boolArr[x][y]) {
					floatArr[x][y] = 1;
				}
			}
		}
		return floatArr;
	}
	
	// Scale every element of a float array in-place.
	// Same pointer is optionally returned.
	static float[][][] scaleImage(final float[][][] image, final float scale) {
		for (int i = 0; i < image.length; i++) {
			for (int x = 0; x < image[0].length; x++) {
				for (int y = 0; y < image[0][0].length; y++) {
					image[i][x][y] *= scale;
				}
			}
		}
		return image;
	}
	
	// Extract byte array from BufferedImage
	static byte[] extractByteArray(BufferedImage image) {
		DataBufferByte dataBufferByte = (DataBufferByte) image.getRaster().getDataBuffer();
		return dataBufferByte.getData();
	}
	
	// Clamp the value to within two constraints
	static int clamp(int a, int min, int max) {
		if (a <= min) {
			return min;
		} else if (a >= max) {
			return max;
		}
		return a;
	}
	
	// Clamp the color to within its limits
	static int clamp(float a) {
		if (a > 255) {
			return 255;
		} else if (a < 0) {
			return 0;
		} else {
			return (int)a;
		}
	}
	
	// Clamp the color to within its limits
	static int clamp(int a) {
		if (a > 255) {
			return 255;
		} else if (a < 0) {
			return 0;
		}
		return a;
	}
	
	// Easy print function
	static void print(final Object o) {
		System.out.println(o);
	}
}
