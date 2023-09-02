
// Perform a single iteration of Richardson-Lucy deconvolution
kernel void rlIteration(
		global float* newImage,
		global float* middleImage,
		global const float* originalImage,
		global const float* weights,
		const int kernelWidth,
		const int offset,
		const int mode) {
	
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	int x = get_global_id(0);
    int y = get_global_id(1);
	
	// Sum the pixels around this pixel
	float r = 0;
	float g = 0;
	float b = 0;
	int sampleCount = 0;
	
	if (mode == 0) {
		for (int x2 = -offset; x2 <= offset; x2++) {
			if (x2 + x < 0 || x2 + x >= width) {
				continue;
			}
			for (int y2 = -offset; y2 <= offset; y2++) {
				if (y2 + y < 0 || y2 + y >= height ||
						weights[(y2 + offset) * kernelWidth + (x2 + offset)] == 0) {
					continue;
				}
				
				int i = ((y+y2) * width + (x+x2)) * 3;
				
				r += newImage[i + 0];
				g += newImage[i + 1];
				b += newImage[i + 2];
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
		
		int i = (y * width + x) * 3;
		middleImage[i + 0] = originalImage[i + 0] / r;
		middleImage[i + 1] = originalImage[i + 1] / g;
		middleImage[i + 2] = originalImage[i + 2] / b;
	} else {
		for (int x2 = -offset; x2 <= offset; x2++) {
			if (x2 + x < 0 || x2 + x >= width) {
				continue;
			}
			for (int y2 = -offset; y2 <= offset; y2++) {
				if (y2 + y < 0 || y2 + y >= height ||
						weights[(y2 + offset) * kernelWidth + (x2 + offset)] == 0) {
					continue;
				}
				
				int i = ((y+y2) * width + (x+x2)) * 3;
				
				r += middleImage[i + 0];
				g += middleImage[i + 1];
				b += middleImage[i + 2];
				sampleCount++;
			}
		}
		
		int i = (y * width + x) * 3;
		newImage[i + 0] *= r / sampleCount;
		newImage[i + 1] *= g / sampleCount;
		newImage[i + 2] *= b / sampleCount;
	}
}