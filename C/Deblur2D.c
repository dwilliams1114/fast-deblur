// Deblur2D.c
// This c file runs the Fast-Method deconvolution algorithm in two dimensions.
// To run this file:
// gcc Deblur2D.c -o Deblur2D.exe -O && Deblur2D.exe

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Clamp 'val' to within 'min' and 'max'
int clamp(int val, int min, int max) {
	if (val < min) {
		return min;
	} else if (val > max) {
		return max;
	}
	return val;
}

// Return a pointer to a particular RGB pixel in the given image
uint8_t* getPixel(uint8_t* image, int x, int y, int width, int height) {
	return image + (x + y * width) * 3;
}

// Return a pointer to a list of coordinates that are a distance 'radius' from (0, 0).
// Length of this list is returned in 'lengthOut'.  (This is the length of the array,
// which is twice the number of coordinates.  A coordinate is two numbers.)
// Function for circle-generation using Midpoint Circle algorithm (floating-point version).
// This works for r up to about 500 for granularity 0.01.
// In the returned array, even indexes are X, and odd indexes are Y.
int* computePointsAtRadius(int* lengthOut, const float radius) {

	// Compute an overestimate for the number of pixels that will be in this circle.
	// (This is usually spot-on)
	int pixelCountOverestimate;
	if (radius <= 0.4f) {
		pixelCountOverestimate = 8;
	} else {
		pixelCountOverestimate = 8 * (int)round(1.4142157f * radius - 9.983285E-5f);
	}

	int* coords = malloc(pixelCountOverestimate * sizeof(int));
	if (coords == NULL) {
		printf("Could not allocate memory in computePointsAtRadius!\n");
		exit(1);
	}

	float x = radius;
	int y = 0;
	int i = 0;
	int rRounded = (int)round(radius);

	// Place the top, bottom, left, right points
	coords[i++] = rRounded;
	coords[i++] = 0;
	coords[i++] = -rRounded;
	coords[i++] = 0;

	coords[i++] = 0;
	coords[i++] = rRounded;
	coords[i++] = 0;
	coords[i++] = -rRounded;

	while (i < pixelCountOverestimate) {
		x = (float)sqrt(x * x - 2 * y - 1);
		y++;

		int xRounded = (int)round(x);
		if (xRounded == 0) {
			break;
		}
		if (xRounded < y) {
			break;
		}

		coords[i++] = xRounded;
		coords[i++] = y;
		coords[i++] = -xRounded;
		coords[i++] = y;
		coords[i++] = xRounded;
		coords[i++] = -y;
		coords[i++] = -xRounded;
		coords[i++] = -y;

		// if x == y, then further points would be duplicates
		if (xRounded == y) {
			break;
		}

		coords[i++] = y;
		coords[i++] = xRounded;
		coords[i++] = -y;
		coords[i++] = xRounded;
		coords[i++] = y;
		coords[i++] = -xRounded;
		coords[i++] = -y;
		coords[i++] = -xRounded;
	}

	// 'i' will always be less than or equal to the length of
	// 'coords', which is equal to 'pixelCountOverestimate',
	// assuming the assumptions for this function are met.
	*lengthOut = i; // i is divided by 2 because two numbers make a coordinate.
	return coords;
}

// Fast-Method deconvolution algorithm (by Daniel Williams).
// Blurred input is given in 'image'.
// Deblurred output is returned in 'image'.
// Typically, 'iterations' = 1 is best.
// 'radius' is the blur radius, and may be any value from 1 to 500 in 0.01 increments.
void deblur2D(uint8_t* image, const int width, const int height, const float radius, const int iterations) {
	const size_t imageSize = width * height * 3;

	// Temporarily allocate some working memory
	uint8_t* newImage = malloc(imageSize);
	uint8_t* oldImage = malloc(imageSize);
	memcpy(oldImage, image, imageSize);

	if (oldImage == NULL || newImage == NULL) {
		printf("Could not allocate memory in deblur2D!\n");
		exit(1);
	}

	// Generate three arrays of coordinates where the even indexes are x-coordinates,
	// and the odd indexes are y-coordinates.
	int points1Length = 0;
	int points2Length = 0;
	int points3Length = 0;
	int* points1 = computePointsAtRadius(&points1Length, radius);
	int* points2 = computePointsAtRadius(&points2Length, radius + 1);
	int* points3 = computePointsAtRadius(&points3Length, radius * 2);

	// Repeat whole algorithm N times.
	for (int iteration = 0; iteration < iterations; iteration++) {

		// Iterate over every pixel
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int sum1R = 0;
				int sum1G = 0;
				int sum1B = 0;
				int sum2R = 0;
				int sum2G = 0;
				int sum2B = 0;
				int sum3R = 0;
				int sum3G = 0;
				int sum3B = 0;

				// Sum up the inner ring (radius r)
				for (int i = 0; i < points1Length; i += 2) {
					int newX = clamp(x + points1[i], 0, width - 1);
					int newY = clamp(y + points1[i + 1], 0, height - 1);
					uint8_t* pix = getPixel(image, newX, newY, width, height);
					sum1R += pix[0];
					sum1G += pix[1];
					sum1B += pix[2];
				}

				// Sum up the second inner ring (radius r+1)
				for (int i = 0; i < points2Length; i += 2) {
					int newX = clamp(x + points2[i], 0, width - 1);
					int newY = clamp(y + points2[i + 1], 0, height - 1);
					uint8_t* pix = getPixel(image, newX, newY, width, height);
					sum2R += pix[0];
					sum2G += pix[1];
					sum2B += pix[2];
				}

				// Sum up the outer ring (radius 2r)
				for (int i = 0; i < points3Length; i += 2) {
					int newX = clamp(x + points3[i], 0, width - 1);
					int newY = clamp(y + points3[i + 1], 0, height - 1);
					uint8_t* pix = getPixel(oldImage, newX, newY, width, height);
					sum3R += pix[0];
					sum3G += pix[1];
					sum3B += pix[2];
				}

				// Scale the summed rings according to our algorithm
				sum1R *= 0.67f / 2;
				sum1G *= 0.67f / 2;
				sum1B *= 0.67f / 2;

				sum2R *= -0.67f / 2 * points1Length / points2Length;
				sum2G *= -0.67f / 2 * points1Length / points2Length;
				sum2B *= -0.67f / 2 * points1Length / points2Length;

				sum3R /= (points3Length / 2);
				sum3G /= (points3Length / 2);
				sum3B /= (points3Length / 2);

				// Set the final values of the pixels
				uint8_t* pix = getPixel(newImage, x, y, width, height);
				pix[0] = clamp(sum1R + sum2R + sum3R, 0, 255);
				pix[1] = clamp(sum1G + sum2G + sum3G, 0, 255);
				pix[2] = clamp(sum1B + sum2B + sum3B, 0, 255);
			}
		}

		// Copy 'newImage' into 'oldImage'
		memcpy(oldImage, newImage, imageSize);
	}

	// Copy 'oldImage' into 'image' as output.
	memcpy(image, oldImage, imageSize);

	// Free up memory
	free(points1);
	free(points2);
	free(points3);
	stbi_image_free(oldImage);
	stbi_image_free(newImage);
}

// Perform a disk blur of the given 'image' of radius 'radius'.
// Blurred image is returned in 'image'.
void blur2D(uint8_t* image, const int width, const int height, const float radius) {
	size_t imageSize = width * height * 3;
	uint8_t* tempImage = malloc(imageSize);

	if (tempImage == NULL) {
		printf("Failed to allocate image memory!\n");
		exit(1);
	}

	const int radiusMax = (int)(radius + 1.5f);
	const float radiusSquare = (radius + 0.5f) * (radius + 0.5f);

	// Iterate over every pixel
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int sumR = 0;
			int sumG = 0;
			int sumB = 0;
			int count = 0;

			// Sum over all the pixels inside the disk of given radius
			for (int i = -radiusMax; i <= radiusMax; i++) {
				for (int j = -radiusMax; j <= radiusMax; j++) {
					int xIndex = x + i;
					int yIndex = y + j;

					// Check if we are in image bounds
					if (xIndex >= 0 && xIndex < width && yIndex >= 0 && yIndex < height) {
						if (i*i + j*j <= radiusSquare) {
							uint8_t* pix = getPixel(image, xIndex, yIndex, width, height);
							sumR += pix[0];
							sumG += pix[1];
							sumB += pix[2];
							count++;
						}
					}
				}
			}

			// Assign the average color of the pixel
			uint8_t* pix = getPixel(tempImage, x, y, width, height);
			pix[0] = sumR / count;
			pix[1] = sumG / count;
			pix[2] = sumB / count;
		}
	}

	// Copy 'tempImage' back into 'image'.
	memcpy(image, tempImage, imageSize);

	// Free memory
	stbi_image_free(tempImage);
}

// Program main entry point.
int main() {

	// Radius of the blur and deblur
	const float blurRadius = 16.0f;

	// Load in a color test image
	int width, height, bpp;
	uint8_t* rgbImage = stbi_load("JWST 512.png", &width, &height, &bpp, 3);
	
	// Blur the image using a disk kernel
	printf("Blurring...\n");
	blur2D(rgbImage, width, height, blurRadius);

	// Deblur using the Fast-Method deconvolution
	printf("Deblurring...\n");
	deblur2D(rgbImage, width, height, blurRadius, 1);

	// Save the image
	printf("Writing image...\n");
	stbi_write_png("TestImageOut.png", width, height, 3, rgbImage, width * 3);

	// Free memory
	stbi_image_free(rgbImage);
	printf("Done\n");

	return 0;
}
