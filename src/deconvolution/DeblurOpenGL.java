package deconvolution;

import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.GLPixelBuffer;
import com.jogamp.opengl.util.GLReadBufferUtil;

import gpuTesting.ShaderProgram;

import java.nio.ByteBuffer;

import javax.swing.JOptionPane;

// Created by Daniel Williams
// Created on April 10, 2023
// Last updated on June 6, 2023
// Created to run the Super-Fast Deconvolution Algorithm in OpenGL

public class DeblurOpenGL {
	
	private static GL2 gl;
	private static ShaderProgram shader;
	private static int textureID;
    private static int width;
    private static int height;
    private static boolean isDeblurEnabled = true;
    private static float blurRadius;
	private static int[] textureIdRef;
	private static boolean isInitialized = false;
	private static byte[] nextImageToRender; // Byte array directly from a BufferedImage
	private static double zoom = 1.0;
	private static double panX = 0;
	private static double panY = 0;
	private static int viewportWidth;
	private static int viewportHeight;
	
	// True if we need to update the shader parameters on the next render cycle (width, height, blur radius)
	private static boolean needToUpdateShaderParameters = false;
	
	// GLCanvas used by the swing GUI
	public static GLCanvas glCanvas;
	
	
	// Set up the Canvas and perform OpenGL initialization
	public static void initializeOpenGLCanvas() {
		
		// Prevent multiple initializations
		if (isInitialized) {
			return;
		}
		isInitialized = true;
		
		GLCapabilities caps = new GLCapabilities(GLProfile.getDefault());
	    caps.setSampleBuffers(true);
	    
	    glCanvas = new GLCanvas(caps);
	    glCanvas.setEnabled(false); // Prevent capturing mouse events (pass through)
	    glCanvas.addGLEventListener(new GLEventListener() {
			public void display(GLAutoDrawable arg0) {
				DeblurOpenGL.display(arg0);
			}
			public void dispose(GLAutoDrawable arg0) {
				DeblurOpenGL.dispose(arg0);
			}
			public void init(GLAutoDrawable arg0) {
				DeblurOpenGL.init(arg0);
			}
			public void reshape(GLAutoDrawable arg0, int arg1, int arg2, int arg3, int arg4) {
				DeblurOpenGL.reshape(arg0, arg1, arg2, arg3, arg4);
			}
		});
	}
	
	// Move the image left, right, or zoom in or out
	public static void setTransform(final double zoom, final double panX, final double panY) {
		DeblurOpenGL.zoom = zoom;
		DeblurOpenGL.panX = panX;
		DeblurOpenGL.panY = panY;
	}

	// Set the image parameters for the next image to deblur.
	// (Only needs to be set once for a sequence of similar images.)
	public static void setImageParameters(final int width, final int height, final float deblurRadius) {
		
		// Set the algorithm parameters
		DeblurOpenGL.width = width;
		DeblurOpenGL.height = height;
		DeblurOpenGL.blurRadius = deblurRadius;
		
		// Update the shader values next time it renders
		needToUpdateShaderParameters = true;
	}
	
	// Set whether deblurring is enabled or disabled
	public static void setDeblurEnabled(boolean isEnabled) {
		if (DeblurOpenGL.isDeblurEnabled != isEnabled) {
			DeblurOpenGL.isDeblurEnabled = isEnabled;
			
			// Update the shader values next time it renders
			needToUpdateShaderParameters = true;
		}
	}
	
	// Recompute the convolution kernels for the blur radius.
	// Also update the width and height in the shader on the GPU.
	private static void updateImageParameters() {
		
		if (!isInitialized) {
			System.err.println("DeblurOpenGL must be initialized before setting image parameters!");
			System.exit(1);
		}
		
		// Make sure the graphics pipeline can handle single-byte alignment
		
		gl.glPixelStorei(GL2.GL_UNPACK_ALIGNMENT, 1);
		
		// Send width and height to the shader
		
		int widthLoc = gl.glGetUniformLocation(shader.getId(), "width"); // Access the "uniform int width" from the shader
		gl.glUniform1i(widthLoc, width);
		
		int heightLoc = gl.glGetUniformLocation(shader.getId(), "height"); // Access the "uniform int height" from the shader
		gl.glUniform1i(heightLoc, height);
		
		// Generate the blur kernel with the given radius
		
		final int[] coords1 = Algorithms.generateCircle(blurRadius);
		final int[] coords2 = Algorithms.generateCircle(blurRadius + 1);
		final int[] coordsOuter = Algorithms.generateCircleQuarterDensity(blurRadius * 2); // Skip 75% of pixels for speed.
		final int coords1Count = coords1.length/2;
		final int coords2Count = coords2.length/2;
		final int coordsOuterCount = coordsOuter.length/2;
		
		// Convert to byte arrays so they can be stored in texture data
		
		final byte[] coords1Byte = new byte[coords1.length];
		for (int i = 0; i < coords1.length; i++) {
			coords1Byte[i] = (byte)coords1[i];
		}
		final byte[] coords2Byte = new byte[coords2.length];
		for (int i = 0; i < coords2.length; i++) {
			coords2Byte[i] = (byte)coords2[i];
		}
		final byte[] coordsOuterByte = new byte[coordsOuter.length];
		boolean printedError = false;
		for (int i = 0; i < coordsOuter.length; i++) {
			// Check for byte out of range
			if (!printedError && (coordsOuter[i] < -128 || coordsOuter[i] > 127)) {
				printedError = true;
				System.err.println("Deblur radius is too large for OpenGL implementation");
			}
			coordsOuterByte[i] = (byte)coordsOuter[i];
		}
		
		// Bind the length values to the shader
		
		int coords1CountLoc = gl.glGetUniformLocation(shader.getId(), "coords1Count"); // Access the "uniform int coords1Count" from the shader
		gl.glUniform1i(coords1CountLoc, coords1Count);
		
		int coords2CountLoc = gl.glGetUniformLocation(shader.getId(), "coords2Count"); // Access the "uniform int coords2Count" from the shader
		gl.glUniform1i(coords2CountLoc, coords2Count);
		
		int coordsOuterCountLoc = gl.glGetUniformLocation(shader.getId(), "coordsOuterCount"); // Access the "uniform int coordsOuterCount" from the shader
		gl.glUniform1i(coordsOuterCountLoc, coordsOuterCount);
		
		// Bind the coords as 2D-textures (only pixels where y=0 is used).
		
		int coords1Loc = gl.glGetUniformLocation(shader.getId(), "coords1"); // Access the "uniform sampler2D coords1" from the shader
		gl.glUniform1i(coords1Loc, 1);
		gl.glActiveTexture(GL2.GL_TEXTURE1);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIdRef[1]);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_NEAREST);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_NEAREST);
		gl.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL2.GL_R8_SNORM, coords1.length, 1, 0, GL2.GL_RED, GL2.GL_BYTE, ByteBuffer.wrap(coords1Byte));
		
		int coords2Loc = gl.glGetUniformLocation(shader.getId(), "coords2"); // Access the "uniform sampler2D coords2" from the shader
		gl.glUniform1i(coords2Loc, 2);
		gl.glActiveTexture(GL2.GL_TEXTURE2);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIdRef[2]);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_NEAREST);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_NEAREST);
		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_R8_SNORM, coords2.length, 1, 0, GL2.GL_RED, GL2.GL_BYTE, ByteBuffer.wrap(coords2Byte));
		
		int coordsOuterLoc = gl.glGetUniformLocation(shader.getId(), "coordsOuter"); // Access the "uniform sampler2D coordsOuter" from the shader
		gl.glUniform1i(coordsOuterLoc, 3);
		gl.glActiveTexture(GL2.GL_TEXTURE3);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIdRef[3]);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_NEAREST);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_NEAREST);
		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_R8_SNORM, coordsOuter.length, 1, 0, GL2.GL_RED, GL2.GL_BYTE, ByteBuffer.wrap(coordsOuterByte));
		
		
		// Bind whether to bypass all deblurring
		
		int enableDeblurLoc = gl.glGetUniformLocation(shader.getId(), "enableDeblur"); // Access the "uniform int enableDeblur" from the shader
		gl.glUniform1i(enableDeblurLoc, DeblurOpenGL.isDeblurEnabled ? 1 : 0);
		
		/*
		// Bind the arrays of coordinates to the shader
		int coords1Loc = gl.glGetUniformLocation(shader.getId(), "coords1");
		gl.glUniform1iv(coords1Loc, coords1.length, coords1, 0);
		
		int coords2Loc = gl.glGetUniformLocation(shader.getId(), "coords2");
		gl.glUniform1iv(coords2Loc, coords2.length, coords2, 0);
		
		int coordsOuterLoc = gl.glGetUniformLocation(shader.getId(), "coordsOuter");
		gl.glUniform1iv(coordsOuterLoc, coordsOuter.length, coordsOuter, 0);
		*/
		
		/*
		int[] ubo = new int[1];
		gl.glGenBuffers(1, ubo, 0);
		
		IntBuffer coordsBuffer = IntBuffer.allocate(coords1.length + 1 + coords2.length + 1 + coordsOuterCount);
		coordsBuffer.put(coords1).put(coords1Count).put(coords2).put(coords2Count).put(coordsOuter).put(coordsOuterCount);
		coordsBuffer.flip();
		int coordsBlockIndex = gl.glGetUniformBlockIndex(shader.getId(), "CoordsBuffer");
		gl.glBindBuffer(GL2.GL_UNIFORM_BUFFER, ubo[0]);
		gl.glBufferData(GL2.GL_UNIFORM_BUFFER, coordsBuffer.limit() * Integer.BYTES, coordsBuffer, GL2.GL_STATIC_DRAW);
		gl.glUniformBlockBinding(shader.getId(), coordsBlockIndex, 0);
		gl.glBindBufferBase(GL2.GL_UNIFORM_BUFFER, 0, ubo[0]);
		//*/
	}
	
	// Stores a reference to the bytes of an image to display in the future
	public static void setImageToRender(final byte[] imageData) {
		nextImageToRender = imageData;
	}
	
	// Runs the shader and updates the graphics
	public static void render() {
		
		if (!isInitialized) {
			System.err.println("Need to initialize DeblurOpenGL before usage");
			System.exit(1);
		}
		
		try {
			glCanvas.display();
		} catch (Exception e) {
			e.printStackTrace();
			
			JOptionPane.showMessageDialog(Interface.frame,
					e.getMessage(),
					"OpenGL Error",
					JOptionPane.ERROR_MESSAGE);
			
		}
	}
	
	// Read all pixels from the entire OpenGL GLCanvas (include background and border)
	public static byte[] readBytesFromCanvas() {
		
		// Make the GLCanvas the current OpenGL context
        GLContext glContext = DeblurOpenGL.glCanvas.getContext();
        glContext.makeCurrent();
        
        // Create GLReadBufferUtil instance
        GLReadBufferUtil glReadBufferUtil = new GLReadBufferUtil(false, false);
        
        // Read the pixels from the GLCanvas into a ByteBuffer
        glReadBufferUtil.readPixels(DeblurOpenGL.glCanvas.getGL(), true);

        GLPixelBuffer pixelsBuffer = glReadBufferUtil.getPixelBuffer();
        
        ByteBuffer buf = (ByteBuffer) pixelsBuffer.buffer;
        buf.rewind();
        final byte[] bytes = new byte[buf.remaining()];
        buf.get(bytes);
        
        glContext.release();
        
        return bytes;
	}
	
	// Called automatically when the glCanvas is first created and used
	private static void init(GLAutoDrawable drawable) {
		gl = drawable.getGL().getGL2();
		
		shader = new ShaderProgram(gl);
		try {
			shader.compileShader(gl, "src/deconvolution/vshader.glsl", GL2.GL_VERTEX_SHADER);
			shader.compileShader(gl, "src/deconvolution/fshader.glsl", GL2.GL_FRAGMENT_SHADER);
			shader.link(gl);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		shader.enableShader(gl);
		
		// Allocate some textures
		textureIdRef = new int[4];
		gl.glGenTextures(textureIdRef.length, textureIdRef, 0);
		textureID = textureIdRef[0];
		
		// Bind the texture to the shader
		int texLoc = gl.glGetUniformLocation(shader.getId(), "tex"); // Access the "uniform sampler2D tex" from the shader
		gl.glUniform1i(texLoc, 0);
		gl.glActiveTexture(GL2.GL_TEXTURE0);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureID);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_NEAREST);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_NEAREST);
		
		
		// Set the image width, height, blur radius, and re-compute the convolutional kernels
		updateImageParameters();
	}
	
	// Called automatically on glCanvas reshape
	private static void reshape(GLAutoDrawable drawable, int x, int y, int newWidth, int newHeight) {
		
		DeblurOpenGL.viewportWidth = newWidth;
		DeblurOpenGL.viewportHeight = newHeight;
		
		// Set standard projection
		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glMatrixMode(GL2.GL_MODELVIEW);
	}
	
	// Called automatically on windows closing
	private static void dispose(GLAutoDrawable drawable) {
		shader.delete(gl);
	}
	
	// Render the OpenGL image (called automatically from glCanvas.display())
	private static void display(GLAutoDrawable drawable) {
		
		// Set background color, and make it opaque
		gl.glClearColor(0, 0, 0, 1);
		gl.glClear(GL2.GL_COLOR_BUFFER_BIT);
		
		// If we have changed the image width, height, or blur radius, then perform the necessary updates to shader parameters
		if (needToUpdateShaderParameters) {
			needToUpdateShaderParameters = false;
			updateImageParameters();
		}
		
		if (nextImageToRender != null) {
			
			// Get the buffer from the newest image
			ByteBuffer byteBuffer = ByteBuffer.wrap(nextImageToRender);
			
			// Send the new texture to the GPU
			gl.glActiveTexture(GL2.GL_TEXTURE0);
			gl.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, width, height, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, byteBuffer);
		}

		// Set the view port to center the image
		final int newWidth = (int)(width * zoom);
		final int newHeight = (int)(height * zoom);
		gl.glViewport(
				(int)((viewportWidth - newWidth)/2.0 + panX),
				(int)((viewportHeight - newHeight)/2.0 + panY),
				newWidth,
				newHeight);
		
		// Draw a quad for the shader to render on
		gl.glColor3f(1, 1, 0);
		gl.glBegin(GL2.GL_QUADS);
		gl.glTexCoord2f(0, 1);
		gl.glVertex3f(-1, 1, 1); // Top Left
		gl.glTexCoord2f(1, 1);
		gl.glVertex3f(1, 1, 1); // Top Right
		gl.glTexCoord2f(1, 0);
		gl.glVertex3f(1, -1, 1); // Bottom Right
		gl.glTexCoord2f(0, 0);
		gl.glVertex3f(-1, -1, 1); // Bottom Left
		gl.glEnd();
		gl.glFlush();
	}
	
	public static void print(Object o) {
		System.out.println(o);
	}
}
