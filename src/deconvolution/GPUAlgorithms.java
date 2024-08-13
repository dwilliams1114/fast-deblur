package deconvolution;

import gpuAbstraction.GPUMem;
import gpuAbstraction.GPUProgram;

// This class contains all of the algorithms that run on the GPU

public class GPUAlgorithms {
	
	// This is for deblurring on the GPU only
	static GPUProgram fastMethodProgram;
	static GPUProgram rlProgram;
	static GPUProgram sharpenProgram;
	
	// Basic deblurring function with GPU acceleration.  This requires an outside OpenCL file "fastMethod.cl"
	// Programmed by Daniel Williams on April 29, 2019 - June 4, 2022
	static float[][][] fastMethodGPU(final byte[] originalImage, int iterations, int width, int height,
			final float amountOffset, float radius, boolean commit) {
		
		UserInterface.setProcessName("Deblurring");
		
		final long startTime = System.currentTimeMillis();
		
		// Generate the blur kernel
		final int[] coords1 = Algorithms.generateCircle(radius);
		final int[] coords2 = Algorithms.generateCircle(radius + 1);
		final int[] coordsOuter;
		if (radius <= 3) {
			coordsOuter = Algorithms.generateCircle(radius * 2 + 0.5f);
		} else {
			// Skip 75% of pixels for speed on larger blur radiuses.
			coordsOuter = Algorithms.generateCircleQuarterDensity(radius * 2 + 0.5f);
		}
		final int coords1Count = coords1.length/2;
		final int coords2Count = coords2.length/2;
		final int coordsOuterCount = coordsOuter.length/2;
		
		// Used to ensure that the weight of the inner ring is the same as the weight of the outer ring
		final float innerToOuterRatio =  (float)coords1Count / coords2Count;
		
		final float innerMult = amountOffset / 2.0f * 0.67f; // Why 0.67 here?
		
		// Create and execute the program on the GPU
		if (fastMethodProgram == null) {
			String baseDir = "";
			if (!UserInterface.isPackagedAsJar) {
				baseDir = "src/deconvolution/";
			}
			fastMethodProgram = new GPUProgram("fastMethod", baseDir + "FastMethod.cl");
			fastMethodProgram.setGlobalWorkGroupSizes(width, height);
		}
		
		byte[] newApproximation = originalImage;
		
		// Create the image to write to (as a single dimensional int array)
		final byte[] linearOutImage = Algorithms.extractByteArray(UserInterface.previewImage);
		
		GPUMem mem1 = fastMethodProgram.setArgument(0, linearOutImage, GPUProgram.WRITE);
		GPUMem mem2 = fastMethodProgram.setArgument(1, newApproximation, GPUProgram.READ);
		fastMethodProgram.setArgument(2, originalImage, GPUProgram.READ); // This only needs to be set once, or if there is a new input image
		fastMethodProgram.setArgument(3, coords1, GPUProgram.READ);
		fastMethodProgram.setArgument(4, coords2, GPUProgram.READ);
		fastMethodProgram.setArgument(5, coordsOuter, GPUProgram.READ);
		fastMethodProgram.setArgument(6, coords1Count, GPUProgram.READ);
		fastMethodProgram.setArgument(7, coords2Count, GPUProgram.READ);
		fastMethodProgram.setArgument(8, coordsOuterCount, GPUProgram.READ);
		fastMethodProgram.setArgument(9, innerToOuterRatio, GPUProgram.READ);
		fastMethodProgram.setArgument(10, innerMult, GPUProgram.READ);
		
		for (int i = 0; i < iterations; i++) {
			UserInterface.updateProgress((double)i/iterations);
			
			fastMethodProgram.executeKernelNoCopyback();
			
			// The input to the next is the result of the previous
			if (i != iterations - 1) {
				GPUProgram.copyGPUMem(mem1, mem2);
			}
			
			if (ImageEffects.isCanceled) {
				return null;
			}
		}
		
		fastMethodProgram.copyFromGPU();
		
		float[][][] commitImage = null;
		if (commit) {
			// Convert the 1d image array into a 2d array
			commitImage = new float[3][width][height];
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int i = (y * width + x) * 3;
					commitImage[2][x][y] = linearOutImage[i + 0] & 0xFF;
					commitImage[1][x][y] = linearOutImage[i + 1] & 0xFF;
					commitImage[0][x][y] = linearOutImage[i + 2] & 0xFF;
				}
			}
		}
		
		UserInterface.setProcessName("Deblurring (" + (System.currentTimeMillis() - startTime) + "ms)");
		UserInterface.updateProgress(1);
		
		return commitImage; // If this is null, then Algorithms.bImage contains the data (from the GPU)
	}
	
	// Richardson-Lucy deconvolution with GPU acceleration.  This requires an outside OpenCL file.
	// Programmed by Daniel Williams on November 8, 2019
	static float[][][] deblurRichardsonLucyGPU(final float[][][] image,
				float radius, int iterations, boolean commit) {
		
		final long startTime = System.currentTimeMillis();
		
		UserInterface.setProcessName("Deblurring");
		
		final int width = image[0].length;
		final int height = image[0][0].length;
		
		// Generate a disk kernel
		// This must have an odd width and height
		final int kernelWidth = (int)(radius + 0.2) * 2 + 1;
		final float[] kernel = new float[kernelWidth * kernelWidth];
		final int offset = kernelWidth / 2;
		for (int x = 0; x < kernelWidth; x++) {
			for (int y = 0; y < kernelWidth; y++) {
				if (Math.hypot(x - offset, y - offset) < radius + 0.375) {
					kernel[y * kernelWidth + x] = 1;
				}
			}
		}
		
		UserInterface.lastKernel = null;
		
		// Create the image that stores the previous image approximation
		final float[] newImage = new float[width * height * 3];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int i = (y * width + x) * 3;
				newImage[i + 0] = image[0][x][y];
				newImage[i + 1] = image[1][x][y];
				newImage[i + 2] = image[2][x][y];
			}
		}
		
		// Create the image to perform calculations on in the middle of the calculation
		final float[] middleImage = new float[width * height * 3];
		
		// Create and execute the program on the GPU
		if (rlProgram == null) {
			String baseDir = "";
			if (!UserInterface.isPackagedAsJar) {
				baseDir = "src/deconvolution/";
			}
			rlProgram = new GPUProgram("rlIteration", baseDir + "RichardsonLucy.cl");
			rlProgram.setGlobalWorkGroupSizes(width, height);
			
			// This one is written over anyway, so we only need to initialize it once
			rlProgram.setArgument(1, middleImage, GPUProgram.READ_WRITE);
			
			// Only set this constant references on the first initialization
			// newImage is initialized to the original image.  It is now held in the GPU.
			rlProgram.setArgument(2, newImage, GPUProgram.READ);
		}
		
		// Reset the starting approximation every time
		rlProgram.setArgument(0, newImage, GPUProgram.READ_WRITE);
		
		// These values may have changed, so set them
		rlProgram.setArgument(3, kernel, GPUProgram.READ);
		rlProgram.setArgument(4, kernelWidth, GPUProgram.READ);
		rlProgram.setArgument(5, offset, GPUProgram.READ);
		
		// Run the whole algorithm many times
		for (int i = 0; i < iterations; i++) {
			
			// middleBlur = image / blur(oldImage)
			rlProgram.setArgument(6, 0, GPUProgram.READ); // First RL algorithm mode
			rlProgram.executeKernelNoCopyback();
			
			// newImage = newImage * blur(middleBlur)
			rlProgram.setArgument(6, 1, GPUProgram.READ); // Second RL algorithm mode
			rlProgram.executeKernelNoCopyback();
			
			UserInterface.updateProgress((double)i / iterations);
			
			// Exit early if the effect has been canceled
			if (ImageEffects.isCanceled) {
				UserInterface.cancelProgress();
				return null;
			}
			
			/* Preview the image after each iteration
			// Copy the image from the GPU output to the preview BufferedImage
			final int[] bufferData = Algorithms.extractByteArray(Algorithms.bImage);
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int i2 = (y * width + x) * 3;
					final int r = clamp(newImage[i2 + 0]);
					final int g = clamp(newImage[i2 + 1]);
					final int b = clamp(newImage[i2 + 2]);
					bufferData[y * width + x] = (r << 16) | (g << 8) | b;
				}
			}
			Interface.redrawPreviewImage();
			*/
		}
		
		rlProgram.copyFromGPU();
		
		float[][][] commitImage = null;
		if (commit) {
			// Convert the 1d image array into a 2d array
			commitImage = new float[3][width][height];
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int i = (y * width + x) * 3;
					commitImage[0][x][y] = newImage[i + 0];
					commitImage[1][x][y] = newImage[i + 1];
					commitImage[2][x][y] = newImage[i + 2];
				}
			}
		} else {
			// Copy the image from the GPU output to the preview BufferedImage
			final byte[] bufferData = Algorithms.extractByteArray(UserInterface.previewImage);
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int i = (y * width + x) * 3;
					bufferData[i + 2] = (byte)clamp(newImage[i + 0]);
					bufferData[i + 1] = (byte)clamp(newImage[i + 1]);
					bufferData[i + 0] = (byte)clamp(newImage[i + 2]);
				}
			}
		}
		
		UserInterface.setProcessName("Deblurring (" + (System.currentTimeMillis() - startTime) + "ms)");
		UserInterface.updateProgress(1);
		
		// Deallocate the memory if the preview has ended
		if (!ImageEffects.isDialogShowing) {
			deallocateMemory();
		}
		
		return commitImage; // If this is null, then Algorithms.bImage contains the data (from the GPU)
	}
	
	// Perform the laplacian sharpen on the GPU
	static float[][][] sharpenGPU(final byte[] originalImage, int width, int height,
			final float weight, float radius, boolean commit) {
		
		UserInterface.setProcessName("Sharpening");
		
		final long startTime = System.currentTimeMillis();
		
		// Create and execute the program on the GPU
		if (sharpenProgram == null) {
			String baseDir = "";
			if (!UserInterface.isPackagedAsJar) {
				baseDir = "src/deconvolution/";
			}
			sharpenProgram = new GPUProgram("sharpen", baseDir + "Sharpen.cl");
			sharpenProgram.setGlobalWorkGroupSizes(width, height);
			//sharpenProgram.setLocalWorkGroupSizes(64, 16);
		}
		
		// Create the image to write to (as a single dimensional int array)
		final byte[] linearOutImage = Algorithms.extractByteArray(UserInterface.previewImage);

		sharpenProgram.setArgument(0, linearOutImage, GPUProgram.WRITE);
		sharpenProgram.setArgument(1, originalImage, GPUProgram.READ);
		sharpenProgram.setArgument(2, radius, GPUProgram.READ);
		sharpenProgram.setArgument(3, weight, GPUProgram.READ);
		sharpenProgram.executeKernel();
		
		float[][][] commitImage = null;
		if (commit) {
			// Convert the 1d image array into a 2d array
			commitImage = new float[3][width][height];
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					int i = (y * width + x) * 3;
					commitImage[2][x][y] = linearOutImage[i + 0] & 0xFF;
					commitImage[1][x][y] = linearOutImage[i + 1] & 0xFF;
					commitImage[0][x][y] = linearOutImage[i + 2] & 0xFF;
				}
			}
		}
		
		final long endTime = System.currentTimeMillis();
		UserInterface.setProcessName("Sharpening (" + (endTime - startTime) + "ms)");
		UserInterface.updateProgress(1);
		
		return commitImage; // If this is null, then Algorithms.bImage contains the data (from the GPU)
	}
	// Reset the memory for all of the GPU programs
	static void deallocateMemory() {
		if (fastMethodProgram != null) {
			fastMethodProgram.dispose();
			fastMethodProgram = null;
		}
		if (rlProgram != null) {
			rlProgram.dispose();
			rlProgram = null;
		}
		if (sharpenProgram != null) {
			sharpenProgram.dispose();
			sharpenProgram = null;
		}
	}
	
	// Clamp the color to within its limits
	static int clamp(float a) {
		if (a > 255) {
			a = 255;
		} else if (a < 0) {
			a = 0;
		}
		return (int)a;
	}
	
	// Clamp the color to within its limits
	static int clamp(int a) {
		if (a > 255) {
			a = 255;
		} else if (a < 0) {
			a = 0;
		}
		return a;
	}
	
	// Easy print function
	static void print(final Object o) {
		System.out.println(o);
	}
}
