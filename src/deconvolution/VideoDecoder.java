package deconvolution;

import java.awt.image.BufferedImage;

import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

import gpuAbstraction.GPUProgram;

// This class decodes and processes videos frame by frame

public class VideoDecoder {
	
	// True when video is playing
	public static boolean isVideoRunning = false;
	
	// Blur radius for the video (constant from start to end)
	static private float blurRadius = 37;
	
	// Intensity of the deblurring (best is 1)
	static private float deblurAmount = 1.0f;
	
	// Number of iterations to run the deblurring algorithm (not used for OpenGL version)
	static private final int deblurIterations = 1;
	
	// Previous video frame used for multithreaded video pipeline processing
	static private BufferedImage previousVideoFrame;
	
	// Whether to save the processed video
	static final boolean saveVideo = false;
	static final String videoFileOutName = "D:/Video/Out1.mp4";
	
	// File to read in
	static String videoFileInName = "D:/Video/Blurry/Playground 37.mp4";
	
	// Main method for video decoding
	public static void main(String[] args) {
		
		UserInterface.setupFrame();
		
		/*
		for (int i = 0; i < 10; i++) {
			VideoDecoder.decodeVideoBytedeco();
		}
		//*/
		
		VideoDecoder.decodeVideoBytedeco(); // Use bytedeco OpenCV bindings
		//VideoDecoder.decodeVideoJavaOpenCV(); // Use Java OpenCV bindings
	}
	
	// Run video deconvolution using Bytedeco in a loop.
	// Read the given video file.
	// This is called by Interface.java.
	public static void runLoopedVideoDeconvolution(String filePath, double deblurRadius) {
		isVideoRunning = true;
		new Thread(new Runnable() {
			public void run() {
				blurRadius = (float)deblurRadius;
				videoFileInName = filePath;
				while (true) {
					VideoDecoder.decodeVideoBytedeco();
				}
			}
		}).start();
	}
	
	// Decode 
	static void decodeVideoBytedeco() {
		isVideoRunning = true;
		
		if (UserInterface.frame == null) {
			print("Frame must be setup before processing video");
			System.exit(1);
		}
		
		final FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFileInName);
		
		try {
			grabber.start();
			int width = grabber.getImageWidth();
			int height = grabber.getImageHeight();
			int numFrames = grabber.getLengthInFrames();
			final double timeLength = grabber.getLengthInTime() / 1000.0; // Now in milliseconds
			
			// Utilities for saving the processed video:
			Java2DFrameConverter videoFrameConverter;
			FFmpegFrameRecorder recorder;
			if (saveVideo) {
				videoFrameConverter = new Java2DFrameConverter();
				recorder = new FFmpegFrameRecorder(videoFileOutName, width, height);
				recorder.setVideoCodec(avcodec.AV_CODEC_ID_HEVC);
				recorder.setAudioChannels(0);
				recorder.setVideoBitrate(13000000);
		        recorder.setFrameRate(grabber.getFrameRate());
		        recorder.start();
			}
			
			// Update on-screen statistics
			UserInterface.imageInfoLabel.setText(width + "x" + height + " px, " +
					numFrames + " frames, " + String.format("%.2f", grabber.getFrameRate()) + " fps");
			
			// If we are using OpenGL, then initialize the shader with the video frame parameters
			if (Algorithms.useOpenGL) {
				DeblurOpenGL.setImageParameters(width, height, blurRadius, deblurAmount);
			}
			
			final long startTime = System.currentTimeMillis();
			
			BufferedImage[] nextGrabbedImage = new BufferedImage[1]; // Java pointer hack
			
			// Get the first frame
			previousVideoFrame = new Java2DFrameConverter().convert(grabber.grabImage());
			
			// Iterate through each frame in the video
			int frameNum = 0;
			while (frameNum < numFrames-1) {
				
				// Perform video decoding in parallel with deconvolution
				Thread frameGrabThead = new Thread(new Runnable() {
					public void run() {
						
						Java2DFrameConverter converter = new Java2DFrameConverter();
						
						Frame frame;
						try {
							frame = grabber.grabImage();
							if (frame == null) {
								return;
							}
						} catch (Exception e) {
							e.printStackTrace();
							System.exit(1);
							return;
						}
						
						// Returns type TYPE_3BYTE_BGR
						nextGrabbedImage[0] = converter.convert(frame);
					}
				});
				
				// Start fetching the next frame while deblurring the previous
				frameGrabThead.start();
				
				// Get the buffer from the BufferedImage
				final byte[] buffer = Algorithms.extractByteArray(previousVideoFrame);
				
				final double criticalCodeTime;
				
				if (Algorithms.useOpenGL) { // Deblur using the OpenGL pipeline
					
					long openGLStart = System.nanoTime();
					
					// Deconvolve and display using OpenGL
					DeblurOpenGL.setImageToRender(buffer);
					DeblurOpenGL.render();
					
					criticalCodeTime = (System.nanoTime() - openGLStart) / 1000000.0;
					
				} else if (Algorithms.useOpenCL) { // Deblur using the GPU (OpenCL)
					
					GPUProgram.initializeGPU(); // Initialization only runs once
					UserInterface.previewImage = previousVideoFrame;
					
					long gpuStart = System.nanoTime();
					
					GPUAlgorithms.fastMethodGPU(buffer, deblurIterations, width, height, deblurAmount, blurRadius, false);
					
					criticalCodeTime = (System.nanoTime() - gpuStart) / 1000000.0;
					
					// Update the GUI visual
					UserInterface.redrawPreviewImage();
					
					// Save the video if desired
					if (saveVideo) {
						// Ignore the last frame
						if (frameNum < numFrames - 2) {
							recorder.record(videoFrameConverter.convert(previousVideoFrame));
						}
					}
				
				} else { // Deblur on the CPU (multithreaded)
					
					long cpuStart = System.nanoTime();
					
					// Convert the buffer to a 3d float array
					float[][][] floatImage = new float[3][width][height];
					for (int x = 0; x < width; x++) {
						for (int y = 0; y < height; y++) {
							int i = (y * width + x) * 3;
							floatImage[0][x][y] = buffer[i + 2] & 0xFF;
							floatImage[1][x][y] = buffer[i + 1] & 0xFF;
							floatImage[2][x][y] = buffer[i + 0] & 0xFF;
						}
					}
					
					floatImage = Algorithms.fastMethodSwitch(floatImage, 1.0f, blurRadius, deblurIterations, false);
					UserInterface.previewImage = Algorithms.arrayToImage(floatImage);

					criticalCodeTime = (System.nanoTime() - cpuStart) / 1000000.0;
					
					// Update the GUI visual
					UserInterface.redrawPreviewImage();
				}
				
				print("Frame " + frameNum + " " + criticalCodeTime + " ms");
				
				// Wait for the next frame to finish being decoded
				frameGrabThead.join();
				
				// Ready for the next frame
				previousVideoFrame = nextGrabbedImage[0];
				
				frameNum++;
			}
			
			// If we are also re-recording video
			if (saveVideo) {
				recorder.stop();
				recorder.close();
			}
			
			grabber.close();
			long duration = System.currentTimeMillis() - startTime;
			print("Processing rate: " + String.format("%.2f", 
					(timeLength / duration)) + " x realtime");
			print(width + "x" + height + " at " + String.format("%.2f",
					(1000.0 * numFrames / duration)) + " fps, " + 
					String.format("%.1f ms/frame", (double)duration / numFrames));
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				grabber.release();
				grabber.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
        
		isVideoRunning = false;
	}
	
	static void print(Object o) {
		System.out.println(o);
	}
}
