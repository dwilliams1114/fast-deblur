package gpuTesting;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.GL2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.Vector;

// Helper program to compile a GLSL shader

public class ShaderProgram {
	
    private int programID = 0;
    private Vector<Integer> shaderIDs;
    
    public ShaderProgram(GL2 gl) {
        shaderIDs = new Vector<Integer>();
        programID = gl.glCreateProgram();
    }
    
    public final void compileShader(GL2 gl, final String srcPath, int shaderType) {
    	int shaderID = gl.glCreateShader(shaderType);
        String[] shaderSrc = readFromFile(srcPath);
        gl.glShaderSource(shaderID, 1, shaderSrc, null, 0);
        gl.glCompileShader(shaderID);
        IntBuffer compileStatusBuffer = IntBuffer.allocate(1);
        gl.glGetShaderiv(shaderID, GL2.GL_COMPILE_STATUS, compileStatusBuffer);
        boolean didFail = compileStatusBuffer.get(0) == GL2.GL_FALSE;
        
        int[] logLength = new int[1];
    	gl.glGetShaderiv(shaderID, GL2.GL_INFO_LOG_LENGTH, logLength, 0);
        if (logLength[0] > 1) {
	        final ByteBuffer infoLog = Buffers.newDirectByteBuffer(logLength[0]);
	        gl.glGetShaderInfoLog(shaderID, infoLog.limit(), null, infoLog);
	        final byte[] infoBytes = new byte[logLength[0]];
	        infoLog.get(infoBytes);
	        System.err.println(new String(infoBytes));
        }
        
        if (didFail) {
        	System.err.println("Failed during compilation.");
            System.exit(1);
        }
        
        shaderIDs.add(shaderID);
        gl.glDeleteShader(0); // Not sure what this does
        gl.glAttachShader(programID, shaderID);
    }
    
    public final void link(GL2 gl) {
        gl.glLinkProgram(programID);
        IntBuffer linkStatusBuffer = IntBuffer.allocate(1);
        gl.glGetProgramiv(programID, GL2.GL_LINK_STATUS, linkStatusBuffer);
        boolean didFail = linkStatusBuffer.get(0) == GL2.GL_FALSE;
        
        int[] logLength = new int[1];
        gl.glGetProgramiv(programID, GL2.GL_INFO_LOG_LENGTH, logLength, 0);
        if (logLength[0] > 1) {
	        byte[] log = new byte[logLength[0]];
	        gl.glGetProgramInfoLog(programID, logLength[0], null, 0, log, 0);
	        System.err.println(new String(log));
        }
        
        if (didFail) {
        	System.err.println("Failed during linking.");
            System.exit(1);
        }
        
        freeShaders(gl);
    }

    public final void enableShader(GL2 gl) {
        gl.glUseProgram(programID);
    }

    public final int getId(){
        return programID;
    }
    
    public void delete(GL2 gl) {
        freeShaders(gl);
        gl.glDeleteProgram(programID);
    }

    private void freeShaders(GL2 gl) {
        for (int shaderId : shaderIDs) {
            gl.glDetachShader(programID, shaderId);
            gl.glDeleteShader(shaderId);
        }
        shaderIDs.clear();
    }
    
    // Return a string where the first element is the contents of the given text file.
    private static String[] readFromFile(String path) {
        File f = new File(path);
        try {
            byte[] bytes = Files.readAllBytes(f.toPath());
            String result = new String();
            for (byte i: bytes) {
                result += (char)i;
            }
            return new String[]{result};
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new String[]{};
    }
};
