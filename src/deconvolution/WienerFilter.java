package deconvolution;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/*
Need to implement from:
https://docs.opencv.org/4.x/de/d3c/tutorial_out_of_focus_deblur_filter.html
Look at
https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d
for optimizations.
Also look at:
https://opencv-java-tutorials.readthedocs.io/en/latest/05-fourier-transform.html
*/

public class WienerFilter {
	
	private static boolean loadedOpenCVLibrary = false;
	
	public static void main(String[] args) {
		
		// Load the native OpenCV library
		if (!loadedOpenCVLibrary) {
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			loadedOpenCVLibrary = true;
		}
		
		Mat imgIn1 = loadImageFromFile("src/deconvolution/Results/JWST 1080 20.png");
		Mat imgIn2 = loadImageFromFile("src/deconvolution/Results/JWST 1080 20.png");
		//Mat imgIn1 = loadImageFromFile("src/deconvolution/Image Radius 12.png");
		//Mat imgIn2 = loadImageFromFile("src/deconvolution/Image Radius 12.png");
		final float[][][] blurred = Algorithms.scaleImage(matToFloatArr(imgIn1), 256);
		final float[][][] original = Algorithms.scaleImage(matToFloatArr(imgIn2), 256);
		
		int snr = 3000; // Signal-to-noise ratio
		int blurRadius = 20; // Radius of disk blur
		
		// Approximate amount by which to extend the border to eliminate unwanted edge-effects.
		final int borderExpansion = (int)(Math.sqrt(snr) * blurRadius * 1.1 + 5);

		final int[] addedMargins = computePaddingForDFT(imgIn1, borderExpansion);
		final int newWidth =  imgIn1.width() + addedMargins[2] + addedMargins[3];
		final int newHeight = imgIn1.height() + addedMargins[0] + addedMargins[1];
		
		// Compute the point-spread-function and Wiener filter ahead of time.
		Mat psf = calcPSF(new Size(newWidth, newHeight), blurRadius);
		Mat wienerFilter = calcWnrFilter(psf, 1.0 / snr);

		//* Code warm up:
		padForDFT(imgIn2, addedMargins);
		Mat imgOut2 = wienerDeconvolve(imgIn2, wienerFilter);
		imgOut2 = cropImage(imgOut2, addedMargins);
		//*/
		
		long startTime = System.currentTimeMillis();
		
		// Extend the borders of the image
		padForDFT(imgIn1, addedMargins);
		
		// Perform deconvolution
		Mat imgOut1 = wienerDeconvolve(imgIn1, wienerFilter);
		
		// Crop the images back to the original size
		imgOut1 = cropImage(imgOut1, addedMargins);
		
		long endTime = System.currentTimeMillis();
		print(endTime - startTime + " ms");
		
		/* Print out a slice of the matrix
		float[][][] arr = matToFloatArr(imgIn1);
		for (int i = 0; i < arr[0].length; i++) {
			print(arr[0][i][arr[0][0].length/2]);
		}
		//*/

		//* Compute the NRMSE
		final float[][][] deblurred = Algorithms.scaleImage(matToFloatArr(imgOut1), 256);
		double nrmseBlur = Algorithms.normalizedRootMeanSquareError(original, blurred);
		double nrmseDeblur = Algorithms.normalizedRootMeanSquareError(original, deblurred);
		print("Blurred: " + nrmseBlur);
		print("Deblur:  " + nrmseDeblur);
		//*/
		
		// Display the images
		//Core.normalize(imgOut, imgOut, 0, 255,  Core.NORM_MINMAX);
		Core.multiply(imgOut1, Scalar.all(256), imgOut1);
		displayMatrix(imgOut1);
		Core.multiply(imgOut2, Scalar.all(256), imgOut2);
		displayMatrix(imgOut2);
	}
	
	// Compute and return the ideal padding size for this image.
	// Take into account the DFT size and the padding needed to reduce edge effects in the Wiener deconvolution.
	// Return the added [top, bottom, left, right] margin.
	private static int[] computePaddingForDFT(Mat inputImg, final int minBorder) {

		// Ensure that some minimum amount of padding is added to eliminate edge effects
		final int minRows = inputImg.rows() + minBorder;
		final int minCols = inputImg.cols() + minBorder;
		
		// Compute optimal size for DFT
		final int m = Core.getOptimalDFTSize(minRows) - inputImg.rows();
		final int n = Core.getOptimalDFTSize(minCols) - inputImg.cols();
		final int top = m / 2;
		final int bottom = m - top;
		final int left = n / 2;
		final int right = n - left;
		
		return new int[] {top, bottom, left, right};
	}
	
	// Expand input image with the given margins.
	private static void padForDFT(Mat inputImg, final int[] margins) {
		
		// Extend the borders of the image
		Core.copyMakeBorder(inputImg, inputImg, margins[0], margins[1], margins[2], margins[3], Core.BORDER_REPLICATE);
	}
	
	// Crop the given image by the given amount
	private static Mat cropImage(Mat inputImg, final int[] margins) {
		return inputImg.submat(margins[0], inputImg.rows() - margins[1], margins[2], inputImg.cols() - margins[3]);
	}
	
	// Read a Mat from the given file path.
	// Matrix is returned in float32 format, normalized to [0,1).
	private static Mat loadImageFromFile(final String fileName) {
		Mat imgIn = Imgcodecs.imread(fileName);
		
		if (imgIn.empty()) {
			System.err.println("Could not read image!");
			System.exit(1);
		}
		
		// Make sure the image has even width and height
		if (imgIn.cols() % 2 != 0 || imgIn.rows() % 2 != 0) {
			System.err.println("Image must have even rows and columns");
			System.exit(2);
		}
		
		// Convert to float32 grayscale
		imgIn.convertTo(imgIn, CvType.CV_32F, 1.0/256, 0.0); // outMat, type, scale, shift
		//Imgproc.cvtColor(imgIn, imgIn, Imgproc.COLOR_BGR2GRAY);
		
		return imgIn;
	}
	
	// Scale the given buffered image
	private static BufferedImage rescaleImage(final BufferedImage inImage, final double scale) {
		final BufferedImage newImage = new BufferedImage(
				(int)(inImage.getWidth() * scale),
				(int)(inImage.getHeight() * scale),
				BufferedImage.TYPE_3BYTE_BGR);
		
		final Graphics g = newImage.createGraphics();
		g.drawImage(inImage, 0, 0, newImage.getWidth(), newImage.getHeight(), null);
		g.dispose();
		
		return newImage;
	}
	
	// Convert a matrix to a float array. [rgb][row][column]
	private static float[][][] matToFloatArr(final Mat m) {
		
		float[][][] arr = new float[m.channels()][m.cols()][m.rows()];
		
		for (int row = 0; row < m.rows(); row++) {
			for (int col = 0; col < m.cols(); col++) {
				final double[] values = m.get(row, col);
				for (int channel = 0; channel < m.channels(); channel++) {
					arr[channel][col][row] = (float)values[channel];
				}
			}
		}
		
		return arr;
	}
	
	// Make the total of the matrix equal to 'total', and return a copy.
	static Mat normalizeMat(Mat m, double total) {
		Scalar sum = Core.sumElems(m);
		sum.val[0] /= total;
		Mat newMat = new Mat();
		Core.divide(m, sum, newMat);
		return newMat;
	}
	
	// Show the given matrix in a JFrame
	private static void displayMatrix(Mat m) {
		BufferedImage image = matToBufferedImage(m);
		displayImage(image);
	}
	
	// Show the given buffered image in a JFrame
	private static void displayImage(BufferedImage img) {
		JFrame frame = new JFrame("Image");
		frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		// Scale down if too large
		if (img.getHeight() > 1020) {
			img = rescaleImage(img, 1010.0 / img.getHeight());
		}
		
		JLabel label = new JLabel(new ImageIcon(img));
		frame.add(label);
		
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
	// Convert the given matrix to a buffered image.
	// The matrix may be grayscale or color, in any data type.
	private static BufferedImage matToBufferedImage(Mat m) {
		
		// Determine RGB or grayscale
		int bufferedImageType = 0;
		if (m.channels() == 1) {
			bufferedImageType = BufferedImage.TYPE_BYTE_GRAY;
		} else if (m.channels() == 3) {
			bufferedImageType = BufferedImage.TYPE_3BYTE_BGR;
		} else {
			System.err.println("Unsupported mat for matToBufferedImage: channels = " + m.channels());
			System.exit(1);
		}
		
		// Convert the image to byte format for the buffered image
		Mat tmp = new Mat();
		m.convertTo(tmp, CvType.CV_8UC3, 1.0, 0.0); // Destination, Type, Scale, Gain
		m = tmp;
		
		// Create an empty image in matching format
		BufferedImage newImage = new BufferedImage(m.width(), m.height(), bufferedImageType);

		// Get the BufferedImage's backing array and copy the pixels directly into it
		final byte[] data = Algorithms.extractByteArray(newImage);
		m.get(0, 0, data);
		
		return newImage;
	}
	
	// Convert a float[rgb][x][y] array into an OpenCV Mat
	private static Mat floatArrayToMat(float[][][] image) {
		final int width = image[0].length;
		final int height = image[0][0].length;
		final int channels = 3;
		
		Mat mat = new Mat(height, width, CvType.CV_32FC(channels));

		float[] data = new float[channels];

		for (int x = 0; x < width; x++) {
		    for (int y = 0; y < height; y++) {
		        for (int c = 0; c < channels; c++) {
		            data[c] = image[c][x][y];
		        }
		        mat.put(y, x, data);
		    }
		}
		
		return mat;
	}
	
	// Performs Wiener deconvolution on the given float[][][] array.
	// This function is meant to be initiated by the GUI Interface.
	// This is called in ImageEffects.java.
	public static float[][][] wienerDeconvolvePublic(final float[][][] image, int blurRadius, int snr) {

		Interface.setProcessName("Deblurring");
		
		// Load the native OpenCV library
		if (!loadedOpenCVLibrary) {
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			loadedOpenCVLibrary = true;
		}
		
		final long startTime = System.currentTimeMillis();
		
		final Mat imageMat = floatArrayToMat(image);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		// Approximate amount by which to extend the border to eliminate unwanted edge-effects.
		final int borderExpansion = (int)(Math.sqrt(snr) * blurRadius * 1.1 + 5);
		
		final int[] addedMargins = computePaddingForDFT(imageMat, borderExpansion);
		final int newWidth =  imageMat.width() + addedMargins[2] + addedMargins[3];
		final int newHeight = imageMat.height() + addedMargins[0] + addedMargins[1];
		
		// Compute the point-spread-function and Wiener filter ahead of time.
		Mat psf = calcPSF(new Size(newWidth, newHeight), blurRadius);
		Mat wienerFilter = calcWnrFilter(psf, 1.0 / snr);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		// Extend the borders of the image
		padForDFT(imageMat, addedMargins);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		// Perform deconvolution
		Mat imgOut1 = wienerDeconvolve(imageMat, wienerFilter);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		// Crop the images back to the original size
		imgOut1 = cropImage(imgOut1, addedMargins);
		
		Interface.setProcessName("Deblurring (" + (System.currentTimeMillis() - startTime) + "ms)");
		Interface.updateProgress(1);
		
		// Exit early if the effect has been canceled
		if (ImageEffects.isCanceled) {
			return null;
		}
		
		return matToFloatArr(imgOut1);
	}
	
	// Compute the Wiener Filter and return the result.
	// 'inputImg' is the image to be deblurred.
	// 'wienerFilter' is the Wiener filter
	private static Mat wienerDeconvolve(final Mat inputImg, final Mat wienerFilter) {
		
		// Extract red, green, blue components
		final ArrayList<Mat> bgrPlanes = new ArrayList<Mat>(3);
		Core.split(inputImg, bgrPlanes);
		
		
		final Mat zeroMat = Mat.zeros(inputImg.size(), CvType.CV_32F);
		final ArrayList<Mat> planesB = new ArrayList<Mat>(2);
		final ArrayList<Mat> planesG = new ArrayList<Mat>(2);
		final ArrayList<Mat> planesR = new ArrayList<Mat>(2);
		
		final Thread[] threads = new Thread[3];
		
		threads[0] = new Thread(new Runnable() {
			public void run() {
				// Add an all-zero complex component to the input image
				final ArrayList<Mat> planes = new ArrayList<Mat>(2);
				planes.add(bgrPlanes.get(0));
				planes.add(zeroMat);
				final Mat matB = new Mat();
				Core.merge(planes, matB);
				
				// Compute DFT of input image
				Core.dft(matB, matB, Core.DFT_SCALE);
				
				// Multiply DFT(input) x DFT(filter)
				Core.mulSpectrums(matB, wienerFilter, matB, 0);
				
				// Compute inverse DFT to get final image
				Core.idft(matB, matB);
				
				// Extract only the real part
				Core.split(matB, planesB);
				
				// Merge back into RGB image
				bgrPlanes.set(0, planesB.get(0));
			}
		});
		
		threads[1] = new Thread(new Runnable() {
			public void run() {
				final ArrayList<Mat> planes = new ArrayList<Mat>(2);
				planes.add(bgrPlanes.get(1));
				planes.add(zeroMat);
				final Mat matG = new Mat();
				Core.merge(planes, matG);
				
				// Compute DFT of input image
				Core.dft(matG, matG, Core.DFT_SCALE);
				
				// Multiply DFT(input) x DFT(filter)
				Core.mulSpectrums(matG, wienerFilter, matG, 0);
				
				// Compute inverse DFT to get final image
				Core.idft(matG, matG);
				
				// Extract only the real part
				Core.split(matG, planesG);
				
				// Merge back into RGB image
				bgrPlanes.set(1, planesG.get(0));
			}
		});
		
		threads[2] = new Thread(new Runnable() {
			public void run() {
				final ArrayList<Mat> planes = new ArrayList<Mat>(2);
				planes.add(bgrPlanes.get(2));
				planes.add(zeroMat);
				final Mat matR = new Mat();
				Core.merge(planes, matR);
				
				// Compute DFT of input image
				Core.dft(matR, matR, Core.DFT_SCALE);
				
				// Multiply DFT(input) x DFT(filter)
				Core.mulSpectrums(matR, wienerFilter, matR, 0);
				
				// Compute inverse DFT to get final image
				Core.idft(matR, matR);
				
				// Extract only the real part
				Core.split(matR, planesR);
				
				// Merge back into RGB image
				bgrPlanes.set(2, planesR.get(0));
			}
		});
		
		// Run each of the threads in parallel
		for (int i = 0; i < threads.length; i++) {
			threads[i].start();
		}
		
		// Wait for each of the threads to finish
		try {
			for (int i = 0; i < threads.length; i++) {
				threads[i].join();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// Merge back into BGR image
		Core.merge(bgrPlanes, inputImg);
		
		return inputImg;
	}
	
	// Create a disk kernel
	private static Mat calcPSF(Size filterSize, int blurRadius) {
		Mat psr = new Mat(filterSize, CvType.CV_32F, new Scalar(0));
		Point point = new Point(filterSize.width / 2, filterSize.height / 2);
		
		// Draw a solid disk
		Imgproc.circle(psr, point, blurRadius + 1, new Scalar(255), -1, Imgproc.LINE_8);
		
		// Normalize to total of 1
		Scalar sum = Core.sumElems(psr);
		Core.divide(psr, sum, psr);
		return psr;
	}
	
	private static Mat fftshift(final Mat inputImg) {
		Mat outputImg = inputImg.clone();
		int cx = outputImg.cols() / 2;
		int cy = outputImg.rows() / 2;
		Mat q0 = new Mat(outputImg, new Rect(0, 0, cx, cy));
		Mat q1 = new Mat(outputImg, new Rect(cx, 0, cx, cy));
		Mat q2 = new Mat(outputImg, new Rect(0, cy, cx, cy));
		Mat q3 = new Mat(outputImg, new Rect(cx, cy, cx, cy));
		Mat tmp = new Mat();
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
		
		return outputImg;
	}
	
	// Compute the Wiener filter, assuming S(f) = 1.  That is, the power spectral density is constant.
	// 'inputPSR' is the point-spread-function of the same size as the image to be deblurred.
	// 'nsr' is the noise-to-signal ratio
	private static Mat calcWnrFilter(final Mat inputPSF, double nsr) {
		Mat h_PSF_shifted = fftshift(inputPSF);
		
		ArrayList<Mat> planes = new ArrayList<Mat>(2);
		planes.add(h_PSF_shifted);
		planes.add(Mat.zeros(inputPSF.size(), CvType.CV_32F));
		
		Mat complexI = new Mat();
		Core.merge(planes, complexI);
		Core.dft(complexI, complexI);
		Core.split(complexI, planes);
		
		// Compute absolute value of planes.get(0) squared
		Mat zeroMat = Mat.zeros(inputPSF.size(), CvType.CV_32F);
		Mat absMat = new Mat();
		Core.absdiff(planes.get(0), zeroMat, absMat);
		Mat denom = new Mat();
		Core.pow(absMat, 2, denom);
		
		Mat denomAdded = new Mat();
		Core.add(denom, new Scalar(nsr), denomAdded);
		
		Mat output_G = new Mat();
		Core.divide(planes.get(0), denomAdded, output_G);
		
		// Add an all-zero complex component to output_G
		final ArrayList<Mat> complexOutputPlanes = new ArrayList<Mat>(2);
		complexOutputPlanes.add(output_G);
		complexOutputPlanes.add(Mat.zeros(inputPSF.size(), CvType.CV_32F));
		Core.merge(complexOutputPlanes, output_G);
		
		return output_G;
	}
	
	private static void print(Object o) {
		System.out.println(o);
	}
}
