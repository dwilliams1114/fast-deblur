// Developed along with DeblurOpenGL.java
// This file runs the fast deconvolution algorithm in GLSL.
// Also requires vshader.glsl.

#version 120

uniform sampler2D tex;			// Image to deblur
uniform sampler2D coords1;		// Used as a 1-dimensional array
uniform sampler2D coords2;		// Used as a 1-dimensional array
uniform sampler2D coordsOuter;	// Used as a 1-dimensional array
uniform int coords1Count;
uniform int coords2Count;
uniform int coordsOuterCount;
uniform bool enableDeblur;		// Boolean
uniform int width;
uniform int height;

const float innerMult = 1.0 / 2.0 * 0.67f;

// Read from the given index of a sampler2D, interpreting it as a 1-dimensional byte buffer,
// where indexes are in pairs.
int readFromArray(sampler2D buffer, int bufferLength, int i) {
	float x = (float(i) + 0.5) / float(bufferLength * 2);
	return int(texture2D(buffer, vec2(x, 0)).r * 127);
}

void main() {
	
	float x = gl_TexCoord[0].x;
	float y = 1.0 - gl_TexCoord[0].y; // Flip upside down
	
	// Bypass all deblurring and just return the image, unmanipulated
	if (!enableDeblur) {
		gl_FragColor = texture2D(tex, vec2(x, y));
		return;
	}
	
	float pixelSizeX = 1.0 / float(width);
	float pixelSizeY = 1.0 / float(height);
	
	float x2 = 0.0;
	float y2 = 0.0;
	vec4 gradient = vec4(0.0, 0.0, 0.0, 0.0);
	vec4 outerCol = vec4(0.0, 0.0, 0.0, 0.0);
	
	// Integrate over the inner negative ring (radius r+1)
	for (int i = 0; i < coords2Count; i++) {
		x2 = readFromArray(coords2, coords2Count, i * 2 + 0) * pixelSizeX + x;
		y2 = readFromArray(coords2, coords2Count, i * 2 + 1) * pixelSizeY + y;
		
		// Clamp coordinates to the image bounds
		x2 = clamp(x2, 0, 0.99999);
		y2 = clamp(y2, 0, 0.99999);
		
		gradient -= texture2D(tex, vec2(x2, y2));
	}
	
	// Scale the negative ring to the same weight as the inner positive ring
	gradient *= float(coords1Count) / float(coords2Count);
	
	// Integrate over the inner positive ring (radius r)
	for (int i = 0; i < coords1Count; i++) {
		x2 = readFromArray(coords1, coords1Count, i * 2 + 0) * pixelSizeX + x;
		y2 = readFromArray(coords1, coords1Count, i * 2 + 1) * pixelSizeY + y;
		
		// Clamp coordinates to the image bounds
		x2 = clamp(x2, 0, 0.99999);
		y2 = clamp(y2, 0, 0.99999);
		
		gradient += texture2D(tex, vec2(x2, y2));
	}
	
	// Integrate over the outer positive ring (radius 2r)
	for (int i = 0; i < coordsOuterCount; i++) {
		x2 = readFromArray(coordsOuter, coordsOuterCount, i * 2 + 0) * pixelSizeX + x;
		y2 = readFromArray(coordsOuter, coordsOuterCount, i * 2 + 1) * pixelSizeY + y;
		
		// Clamp coordinates to the image bounds
		x2 = clamp(x2, 0, 0.99999);
		y2 = clamp(y2, 0, 0.99999);
		
		outerCol += texture2D(tex, vec2(x2, y2));
	}
	
	gl_FragColor = innerMult * gradient + outerCol / float(coordsOuterCount);
	
	//vec4 m = texture2D(tex, vec2(0.99999, 0.99999));
	//gl_FragColor = m;
	
	/*
	int i = int(x * width);
	if (i < coords1Count * 2) {
		float val = readFromArray(coords1, coords1Count, i) / 256.0;
		if (y < 0.5) {
			col.r = val;
		} else {
			col.r = val * 35;
		}
	} else {
		col.b = 0.3;
	}
	*/
	
	/*
	int val = int(texture2D(coords1, vec2(x, 0)).r * 127 + 0.1);
	
	if (y < 1.0 / height) {
		col.r = 1.0;
	} else {
		col.r = float(val) / 256.0;
	}
	//*/
	
	//col += texture2D(tex, vec2(x, y));
	
	//gl_FragColor = col;
}
