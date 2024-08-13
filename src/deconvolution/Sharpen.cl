
// Perform an unsharp mask on the given image.
// Called (indirectly) from GPUAlgorithms.java in the deconvolution project.
kernel void sharpen(
			global uchar* outImage,
			global const uchar* originalImage,
			float radius,
			float weight) {
	
	int x = get_global_id(0);
    int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	int radiusMax = (int)radius;
	
	float sumR = 0;
	float sumG = 0;
	float sumB = 0;
	int count = 0;
	
	// Sum up all the pixels inside the disk
	for (int i = -radiusMax; i <= radiusMax; i++) {
		for (int j = -radiusMax; j <= radiusMax; j++) {
			int xIndex = x + i;
			int yIndex = y + j;
			if (xIndex >= 0 && xIndex < width && yIndex >= 0 && yIndex < height) {
				if (i * i + j * j <= (radius + 0.375f) * (radius + 0.375f)) {
					int imgIndex = (yIndex * width + xIndex) * 3;
					sumB += originalImage[imgIndex + 0];
					sumG += originalImage[imgIndex + 1];
					sumR += originalImage[imgIndex + 2];
					count++;
				}
			}
		}
	}
	
	int i = (y * width + x) * 3;
	
	// Subtract the average color from the original image
	sumB = originalImage[i + 0] - sumB / count;
	sumG = originalImage[i + 1] - sumG / count;
	sumR = originalImage[i + 2] - sumR / count;
	
	// Scale the difference and add it back to the original image
	sumB = originalImage[i + 0] + sumB * weight;
	sumG = originalImage[i + 1] + sumG * weight;
	sumR = originalImage[i + 2] + sumR * weight;
	
	// Set the final color of the pixel
	outImage[i + 0] = (uchar)clamp(sumB, 0.0f, 255.0f);
	outImage[i + 1] = (uchar)clamp(sumG, 0.0f, 255.0f);
	outImage[i + 2] = (uchar)clamp(sumR, 0.0f, 255.0f);
}
