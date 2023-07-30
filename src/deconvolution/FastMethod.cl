
// Deblur the input image using the Fast-Method.
// Called (indirectly) from GPUAlgorithms.java in the deconvolution project.
kernel void fastMethod(
			global uchar* outImage,
			global const uchar* newApproximation,
			global const uchar* originalImage,
			global const int* coords1,
			global const int* coords2,
			global const int* coordsOuter,
			int coords1Count,
			int coords2Count,
			int coordsOuterCount,
			float innerToOuterRatio,
			float innerMult) {
	
	int x = get_global_id(0);
    int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	float gradientB = 0;
	float gradientG = 0;
	float gradientR = 0;
	
	// Integrate over the inner negative ring (radius r+1)
	for (int j = 0; j < coords2Count; j++) {
		int x2 = coords2[j * 2 + 0] + x;
		int y2 = coords2[j * 2 + 1] + y;
		
		// Make sure this is in bounds
		x2 = clamp(x2, 0, width-1);
		y2 = clamp(y2, 0, height-1);
		
		int i = (y2 * width + x2) * 3;
		gradientB -= originalImage[i + 0];
		gradientG -= originalImage[i + 1];
		gradientR -= originalImage[i + 2];
	}
	
	// Scale the negative ring to the same weight as the inner positive ring
	gradientB *= innerToOuterRatio;
	gradientG *= innerToOuterRatio;
	gradientR *= innerToOuterRatio;
	
	// Integrate over the inner positive ring (radius r)
	for (int j = 0; j < coords1Count; j++) {
		int x2 = coords1[j * 2 + 0] + x;
		int y2 = coords1[j * 2 + 1] + y;
		
		// Make sure this is in bounds
		x2 = clamp(x2, 0, width-1);
		y2 = clamp(y2, 0, height-1);
		
		int i = (y2 * width + x2) * 3;
		gradientB += originalImage[i + 0];
		gradientG += originalImage[i + 1];
		gradientR += originalImage[i + 2];
	}
	
	// Calculate the ring suppression
	float outerB = 0;
	float outerG = 0;
	float outerR = 0;
	
	// Sum up the pixels around this pixel
	for (int j = 0; j < coordsOuterCount; j++) {
		int x2 = coordsOuter[j * 2 + 0] + x;
		int y2 = coordsOuter[j * 2 + 1] + y;
		
		// Make sure this is in bounds
		x2 = clamp(x2, 0, width-1);
		y2 = clamp(y2, 0, height-1);
		
		int i = (y2 * width + x2) * 3;
		outerB += newApproximation[i + 0];
		outerG += newApproximation[i + 1];
		outerR += newApproximation[i + 2];
	}
	
	// Calculate the new color of this pixel
	float newB = innerMult * gradientB + outerB / coordsOuterCount;
	float newG = innerMult * gradientG + outerG / coordsOuterCount;
	float newR = innerMult * gradientR + outerR / coordsOuterCount;
	
	// Set the final color of the pixel
	int i = (y * width + x) * 3;
	outImage[i + 0] = (uchar)clamp(newB, 0.0f, 255.0f);
	outImage[i + 1] = (uchar)clamp(newG, 0.0f, 255.0f);
	outImage[i + 2] = (uchar)clamp(newR, 0.0f, 255.0f);
}
