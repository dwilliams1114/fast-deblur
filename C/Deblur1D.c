// Deblur1D.c
// This c file runs the Fast-Method deconvolution algorithm in one dimension.
// gcc Deblur1D.c -o Deblur1D.exe -O && Deblur1D.exe

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Length of the array to deblur:
#define NUM_SAMPLES 30

// Clamp 'val' to within 'min' and 'max'
int clamp(int val, int min, int max) {
	if (val < min) {
		return min;
	} else if (val > max) {
		return max;
	}
	return val;
}

// Fast-Method deconvolution algorithm (by Daniel Williams)
// 'samples' is the original blurred signal.  The deblurred signal is returned in its place.
// 'oldImage' is a blank work buffer.
// 'newImage' is a blank work buffer.
// 'radius' is the radius of deblurring.
// 'iterations' is the number of iterations to run the deconvolution algorithm.
// Note: This code has not been highly optimized.
void deblur1D(float* samples, float* oldImage, float* newImage, const int radius, const int iterations) {

	// Copy 'samples' into 'oldImage'
	memcpy(oldImage, samples, NUM_SAMPLES * sizeof(samples[0]));

	// Perform all the iterations of deconvolution
	for (int iteration = 0; iteration < iterations; iteration++) {
		for (int i = 0; i < NUM_SAMPLES; i++) {
			newImage[i] = samples[clamp(i + radius, 0, NUM_SAMPLES - 1)] +
			              samples[clamp(i - radius, 0, NUM_SAMPLES - 1)] -
			              samples[clamp(i + radius + 1, 0, NUM_SAMPLES - 1)] -
			              samples[clamp(i - radius - 1, 0, NUM_SAMPLES - 1)];
			newImage[i] *= (2 * radius + 1) / 2;
			newImage[i] += (
					oldImage[clamp(i + 2 * radius + 1, 0, NUM_SAMPLES - 1)] +
					oldImage[clamp(i - 2 * radius - 1, 0, NUM_SAMPLES - 1)]
				) / 2;
		}

		// Copy 'newImage' into 'oldImage'
		memcpy(oldImage, newImage, NUM_SAMPLES * sizeof(newImage[0]));
	}

	// Copy 'newImage' into 'samples' as output.
	memcpy(samples, newImage, NUM_SAMPLES * sizeof(newImage[0]));
}

// Blur the given 'samples' with a box kernel of radius 'radius'.
// 'samples' is the array to blur. The blurred values are returned in its place.
// 'workArray' is a blank working array.
// 'radius' is the radius of the box kernel with which to convolve over 'samples'.
void blur1D(float* samples, float* workArray, const int radius) {

	// Average each sample with the surrounding samples.
	for (int i = 0; i < NUM_SAMPLES; i++) {
		float total = 0;
		int count = 0;
		for (int j = -radius; j <= radius; j++) {
			int index = i + j;
			if (index >= 0 && index < NUM_SAMPLES) {
				total += samples[index];
				count++;
			}
		}
		workArray[i] = total / count;
	}
	
	// Copy 'workArray' into 'samples' as output.
	memcpy(samples, workArray, NUM_SAMPLES * sizeof(samples[0]));
}

// Print the elements of an array line-by-line.
void printArr(float* arr, const int length) {
	for (int i = 0; i < length; i++) {
		printf("%f\n", arr[i]);
	}
}

// Program main entry point.
int main() {

	const int blurRadius = 4;
	const int deblurIterations = 20;

	// Signal to blur, then deblur.
	float samples[NUM_SAMPLES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.0f, 0, 0, -2.0f, 3.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// Work buffers of the same length as the signal to deblur.
	float workArray1[NUM_SAMPLES] = { 0 };
	float workArray2[NUM_SAMPLES] = { 0 };

	// Blur the signal
	blur1D(samples, workArray1, blurRadius);

	printf("Blurred array:\n");
	printArr(samples, NUM_SAMPLES);

	// Deblur the signal
	deblur1D(samples, workArray1, workArray2, blurRadius, deblurIterations);

	printf("Reconstructed array:\n");
	printArr(samples, NUM_SAMPLES);

	return 0;
}