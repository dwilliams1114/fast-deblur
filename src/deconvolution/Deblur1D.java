package deconvolution;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.AudioFileFormat.Type;
import javax.sound.sampled.AudioFormat.Encoding;

// This class contains functions for testing my custom 1D deblurring technique.
// Created on September 28, 2022
// Last updated on September 29, 2022

public class Deblur1D {
	
	static float audioSampleRate;
	
	public static void main(String[] args) {
		
		/*
		double[] errorValues = new double[1000];
		for (int j = 1; j < errorValues.length; j++) {
			final int radius = 8;
			float[] arr = generateImpulse(10000);
			
			arr = Algorithms.blur1D(arr, radius);
			arr = Algorithms.deblur1Dv2(arr, radius, j);
			
			double rootMeanSquareError = 0;
			for (int i = 0; i < arr.length; i++) {
				double diff = arr[i] - (i == arr.length/2 ? 1 : 0);
				rootMeanSquareError += diff*diff;
			}
			rootMeanSquareError = Math.sqrt(rootMeanSquareError / arr.length);
			errorValues[j] = rootMeanSquareError;
			
			print(j + ", " + (errorValues[j] / Math.pow(errorValues[j-1], 1)));
			//print(j + ", " + rate);
			//print(j + ", " + (rootMeanSquareError*rootMeanSquareError*1000));
		}
		for (int i = 2; i < errorValues.length-1; i++) {
			double top = Math.log(Math.abs((errorValues[i+1] - errorValues[i]) / (errorValues[i] - errorValues[i-1])));
			double bottom = Math.log(Math.abs((errorValues[i] - errorValues[i-1]) / (errorValues[i-1] - errorValues[i-2])));
			double orderOfConvergence = top / bottom;
			print(orderOfConvergence);
		}
		//*/
		
		// Fitting by eye:
		// RMSE converges by the 1/sqrt(sqrt(x))
		// MSE converges by the 1/sqrt(x)
		// Order of convergence of RMSE is about q=1, with mu=0.999
		
		//*
		final int radius = 40;
		float[] arr = generateStep(800);
		
		arr = Algorithms.blur1D(arr, radius);
		arr = Algorithms.deblur1Dv2(arr, radius, 2); //1, 2, 3, 10, 51, 1001
		
		//*
		for (int i = 0; i < arr.length; i++) {
			//print(String.format("%.6f", arr[i]));
			print(i + "," + String.format("%.6f", arr[i]));
		}
		//*/
		
		/*
		final int radius = 1000;
		float[] samples = readSoundFile();
		samples = Algorithms.blur1D(samples, radius);
		saveSoundFile(samples, "_BLUR");
		quantizeTo16Bits(samples);
		long startTime = System.currentTimeMillis();
		//float[] samples1 = Algorithms.deblur1Dv2(samples, radius, 4);
		//float[] samples2 = Algorithms.deblur1Dv2(samples, radius, 3);
		//samples = weightedAverage(samples1, samples2, 0.6, 0.6);
		samples = Algorithms.deblur1Dv2(samples, radius, 21);
		//samples = Algorithms.deblur1D(samples, radius);
		long endTime = System.currentTimeMillis();
		print("Done in " + (endTime - startTime) + " ms");
		saveSoundFile(samples, "_DEBLUR");
		//*/
		
		/*  Compute NRMSE for various number of deblur iterations
		final int radius = 1000;
		float[] samplesOriginal = readSoundFile("sound/Flourish.wav");
		float[] samplesBlurred = Algorithms.blur1D(samplesOriginal, radius);
		print(computeNRMSE(samplesBlurred, samplesOriginal));
		//print("Deblur:   " + computeNRMSE(samplesOriginal, samplesDeblurred));
		for (int i = 1; i < 1000; i++) {
			float[] samplesDeblurred = Algorithms.deblur1Dv2(samplesBlurred, radius, i);
			print(computeNRMSE(samplesOriginal, samplesDeblurred));
		}
		//*/
	}
	
	// Compute the normalized root mean square error
	static double computeNRMSE(float[] samples1, float[] samples2) {
		
		double average1 = 0;
		for (int i = 0; i < samples1.length; i++) {
			average1 += samples1[i];
		}
		average1 /= samples1.length;
		
		double average2 = 0;
		for (int i = 0; i < samples2.length; i++) {
			average2 += samples2[i];
		}
		average2 /= samples2.length;
		
		// Compute MSE
		double squareError = 0;
		for (int i = 0; i < samples1.length; i++) {
			double error = (samples1[i] - average1) - (samples2[i] - average2);
			squareError += error * error;
		}
		
		squareError /= samples1.length;
		
		return Math.sqrt(squareError);
	}
	
	// Read in the sound file and write the waveform to a global array.
	// Assumes 16 bit-depth.
	static float[] readSoundFile(final String soundFilePath) {
		
		try {
			AudioInputStream stream = AudioSystem.getAudioInputStream(new File(soundFilePath));
			AudioFormat format = stream.getFormat();
			if (format.getChannels() != 1) {
				print("Audio must have a single channel!");
				System.exit(0);
			} else if (format.getSampleSizeInBits() != 16) {
				print("Audio must be 16 bit format!");
				System.exit(0);
			} else if (format.getEncoding() != Encoding.PCM_SIGNED) {
				print("Audio must be 16-bit Signed format! " + format.getEncoding());
				System.exit(0);
			}
			audioSampleRate = format.getSampleRate();
			
			int bytesToRead = stream.available();
			byte[] bytes = new byte[bytesToRead];
			stream.read(bytes);
			
			int samplesToRead = bytesToRead/2;
			float[] soundSamples = new float[samplesToRead];
			
			// Every pair of bytes makes a sample
			for (int i = 0; i < bytesToRead; i += 2) {
				//short val = (short)(((bytes[i] & 0xFF) << 8) | (bytes[i + 1] & 0xFF));
				short val = (short)(((bytes[i] & 0xFF)) | ((bytes[i + 1] & 0xFF) << 8));
				soundSamples[i/2] = (float)val / Short.MAX_VALUE;
				
				//printSample(soundSamples[i/2]);
			}
			
			return soundSamples;
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		return null;
	}
	
	// Save the global float array as a new WAV file next to the original.
	// Saves as WAV 16-bit signed PCM format.
	static void saveSoundFile(final float[] samples, final String filePath, final String suffixToAppend) {
		
		// Generate the bytes of the WAV file
		final byte[] outputBytes = new byte[samples.length * 2];
		for (int i = 0; i < samples.length; i++) {
			short val = (short)(Math.max(Math.min(samples[i] * Short.MAX_VALUE, Short.MAX_VALUE), Short.MIN_VALUE));
			outputBytes[i*2] = (byte)(val & 0xFF);
			outputBytes[i*2+1] = (byte)(val >>> 8);
		}
		
		AudioFormat outFormat = new AudioFormat(audioSampleRate, 16, 1, true, false);
		InputStream byteInputStream = new ByteArrayInputStream(outputBytes);
		try {
			
			String outFileName = filePath.substring(0, filePath.lastIndexOf('.')) + suffixToAppend + ".wav";
			print("Saving as " + outFileName);
			AudioInputStream stream = new AudioInputStream(byteInputStream, outFormat, outputBytes.length);
			File file = new File(outFileName);
			AudioSystem.write(stream, Type.WAVE, file);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	// Generate a square impulse one unit high of diameter 2*r+1
	static float[] generateSquareImpulse(final int numSamples, final int radius) {
		final float[] arr = new float[numSamples];
		for (int i = numSamples/2 - radius; i <= numSamples/2 + radius; i++) {
			arr[i] = 1.0f;
		}
		return arr;
	}
	
	// Generate a step function one unit high
	static float[] generateStep(final int numSamples) {
		final float[] arr = new float[numSamples];
		for (int i = numSamples/2; i < numSamples; i++) {
			arr[i] = 1.0f;
		}
		return arr;
	}
	
	// Generate a saw tooth function
	static float[] generateSawTooth(final int numSamples, final int samplePeriod) {
		final float[] arr = new float[numSamples];
		for (int i = 0; i < numSamples; i++) {
			arr[i] = (float)(i % samplePeriod) / (samplePeriod - 1);
		}
		return arr;
	}
	
	// Generate random noise
	static float[] generateNoise(final int numSamples) {
		final float[] arr = new float[numSamples];
		for (int i = 0; i < numSamples; i++) {
			arr[i] = (float)Math.random();
		}
		return arr;
	}
	
	// Generate impulse function
	static float[] generateImpulse(final int numSamples) {
		final float[] arr = new float[numSamples];
		arr[numSamples/2] = 1;
		return arr;
	}
	
	// Quantize the given array as if it were saved as a signed 16-bit audio file.
	static void quantizeTo16Bits(final float[] samples) {
		for (int i = 0; i < samples.length; i++) {
			short val = (short)(Math.max(Math.min(samples[i] * Short.MAX_VALUE, Short.MAX_VALUE), Short.MIN_VALUE));
			byte byte1 = (byte)(val & 0xFF);
			byte byte2 = (byte)(val >>> 8);
			short newVal = (short)(((byte1 & 0xFF)) | ((byte2 & 0xFF) << 8));
			samples[i] = (float)newVal / Short.MAX_VALUE;
		}
	}
	
	// Perform a weighted average of the two arrays
	static float[] weightedAverage(final float[] samples1, final float[] samples2, final double a, final double b) {
		final float[] outSamples = new float[samples1.length];
		for (int i = 0; i < outSamples.length; i++) {
			outSamples[i] = (float)(samples1[i] * a + samples2[i] * b);
		}
		return outSamples;
	}
	
	public static void print(Object o) {
		System.out.println(o);
	}
}
