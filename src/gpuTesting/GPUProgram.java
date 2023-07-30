package gpuTesting;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

// Created by Daniel Williams
// Created on February 27, 2019
// Last update on December 27, 2020

// This class provides access to OpenCL based general purpose
//   massively parallel processing using the GPU.

// This uses the JOCL (not JogAmp) implementation of Java OpenCL bindings

enum ArrayType {
	BYTE, INT, FLOAT, LONG, DOUBLE, BUFFERED_IMAGE
}

public class GPUProgram {
	
	public static final long READ = CL.CL_MEM_READ_ONLY;
	public static final long WRITE = CL.CL_MEM_WRITE_ONLY;
	public static final long READ_WRITE = CL.CL_MEM_READ_WRITE;
	
	private static cl_command_queue commandQueue;
	private static cl_context context;
	private static cl_device_id device;
	private cl_program program;
	private cl_kernel kernel;
	
	// Whether the GPU has already been initialized
	private static boolean initialized = false;
	
	// Work size for each dimension
	private long[] globalWorkSize;
	
	// Local work size for each dimension
	private long[] localWorkSize = null;
	
	// This temporarily holds the parameters just in case they need to be released
	private cl_mem[] params = new cl_mem[30];
	
	private int writeIndex = 0;
	private int[] writeArgumentNum; 
	private cl_mem[] writeArrayBuffers;
	private ArrayType[] writeArrayType;
	
	private float[][] arrayToFillFloat;
	private int[][] arrayToFillInt;
	private byte[][] arrayToFillByte;
	private BufferedImage[] imageToFill;
	
	// Step 1: This is called first to initialize the GPU
	@SuppressWarnings("deprecation")
	public static void initializeGPU() {
		// Don't initialize twice
		if (initialized) {
			return;
		}
		initialized = true;
		
		//final long deviceType = CL.CL_DEVICE_TYPE_DEFAULT;
		//final long deviceType = CL.CL_DEVICE_TYPE_CPU;
		final long deviceType = CL.CL_DEVICE_TYPE_GPU | CL.CL_DEVICE_TYPE_ACCELERATOR;
		final int deviceIndex = 0;
		
		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);
		
		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
		
		final int platformIndex = 0;
		cl_platform_id platform = platforms[platformIndex];
		
		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
		
		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		// Obtain a device ID 
		cl_device_id devices[] = new cl_device_id[numDevices];
		CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		device = devices[deviceIndex];
		
		// Create a context for the selected device
		context = CL.clCreateContext(contextProperties, 1, new cl_device_id[] { device },
				null, null, null);
		
		//print(getDeviceString(device, CL.CL_DEVICE_NAME));
		//print(getDeviceString(device, CL.CL_DEVICE_VENDOR));
		//print(getDeviceString(device, CL.CL_DRIVER_VERSION));
		//print(getDeviceInt(device, CL.CL_DEVICE_MAX_COMPUTE_UNITS));
		//print(getDeviceInt(device, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY));
		
		// Create a command-queue for the selected device
		try {
			commandQueue = CL.clCreateCommandQueueWithProperties(context, device, null, null);
		} catch (Exception e) { // This is for older systems (OpenCL 1.2)
			commandQueue = CL.clCreateCommandQueue(context, device, 0, null);
		}
	}
	
	// Step 2: This is called to create the kernel from the shader that will be repeatedly executed.
	// kernelName must match that in the shader program.
	public GPUProgram(String kernelName, String directory) {
		try {
			// Load the lines of code for the OpenCL kernel
			final BufferedReader br = new BufferedReader(
					new InputStreamReader(new FileInputStream(directory)));
			String sourceCode = "";
			String line = null;
			while (true) {
				line = br.readLine();
				if (line == null) {
					break;
				}
				sourceCode += line + "\n";
			}
			br.close();
			
			// Create the program
			program = CL.clCreateProgramWithSource(context, 1, new String[] {sourceCode}, null, null);
			
			final String opts = "-cl-mad-enable -cl-fast-relaxed-math -cl-unsafe-math-optimizations";
			
			// Build the program
			CL.clBuildProgram(program, 0, null, opts, null, null);
			
			/* Possible optimization parameters:
			-cl-strict-aliasing
			-cl-mad-enable
			-cl-no-signed-zeros
			
			-cl-finite-math-only
			-cl-fast-relaxed-math
			-w
			-Werror
			*/
			
			// Create the kernel
			kernel = CL.clCreateKernel(program, kernelName, null);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		writeArgumentNum = new int[20]; 
		writeArrayBuffers = new cl_mem[20];
		writeArrayType = new ArrayType[20];
		
		arrayToFillFloat = new float[20][];
		arrayToFillInt = new int[20][];
		arrayToFillByte = new byte[20][];
		imageToFill = new BufferedImage[20];
		
		// Set all of the write indices to -1
		for (int i = 0; i < writeArgumentNum.length; i++) {
			writeArgumentNum[i] = -1;
		}
	}
	
	// Step 3: Set the sizes of the global work groups.
	// This is effectively the array size and dimension that the GPU will iterate through
	public void setGlobalWorkGroupSizes(long ... workSizes) {
		globalWorkSize = workSizes;
		if (localWorkSize == null) {
			localWorkSize = new long[workSizes.length];
			for (int i = 0; i < workSizes.length; i++) {
				// Try to guess the best local work group size
				if (workSizes[i] % 7 == 0) {
					localWorkSize[i] = 7;
				} else if (workSizes[i] % 4 == 0) {
					localWorkSize[i] = 4;
				} else if (workSizes[i] % 3 == 0) {
					localWorkSize[i] = 3;
				} else if (workSizes[i] % 2 == 0) {
					localWorkSize[i] = 2;
				} else {
					localWorkSize[i] = 1;
				}
			}
		}
	}
	
	// Step 3b (optional): Set the sizes of the local work groups.
	// These values must be multiples of the global work group sizes.
	// The global work groups are broken up into these local work groups.
	public void setLocalWorkGroupSizes(long ... workSizes) {
		localWorkSize = workSizes;
	}
	
	// Step 4: Set the arguments for the given kernel.
	// Arguments only need to be set if they have changed!
	// Array arguments only need to be set if their reference has changed!
	public void setArgument(int argNum, Object arg, long accessType) {
		
		if (argNum < 0) {
			new Exception("Kernel argument must positive").printStackTrace();
			System.exit(1);
			return;
		}
		
		if (!initialized) {
			new Exception("GPU has not been initialized").printStackTrace();
			System.exit(1);
			return;
		}
		
		if (arg instanceof Float)  {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_float, Pointer.to(new float[] { (float)arg }));
		} else if (arg instanceof Integer) {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_int, Pointer.to(new int[] { (int)arg }));
		} else if (arg instanceof Long) {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_long, Pointer.to(new long[] { (long)arg }));
		} else if (arg instanceof float[]) {
			final float[] array = (float[])arg;
			
			// Release this memory if it has changed
			if (params[argNum] != null) {
	        	CL.clReleaseMemObject(params[argNum]);
	        }
			params[argNum] = CL.clCreateBuffer(context, accessType, 
					array.length * Sizeof.cl_float, null, null);
			
			if (accessType == CL.CL_MEM_READ_ONLY || accessType == CL.CL_MEM_READ_WRITE) {
				// This step takes a long time (90 milliseconds)
		        CL.clEnqueueWriteBuffer(commandQueue, params[argNum], true, 0,
		        		array.length * Sizeof.cl_float, Pointer.to(array), 0, null, null);
		        
		        CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(params[argNum]));
			}
			
			if (accessType == CL.CL_MEM_WRITE_ONLY || accessType == CL.CL_MEM_READ_WRITE) {
				int index = findWriteArgumentIndex(argNum);
				
				// If this is the first time this argument has been written to
				if (index == -1) {
					writeArgumentNum[writeIndex] = argNum;
		        	writeArrayBuffers[writeIndex] = params[argNum];
		        	writeArrayType[writeIndex] = ArrayType.FLOAT;
		        	arrayToFillFloat[writeIndex] = array;
		        	writeIndex++;
				} else { // This has been written to before
					writeArgumentNum[index] = argNum;
		        	writeArrayBuffers[index] = params[argNum];
		        	writeArrayType[index] = ArrayType.FLOAT;
		        	arrayToFillFloat[index] = array;
				}
	        }
		} else if (arg instanceof int[]) {
			final int[] array = (int[])arg;
			
			// Release this memory if it has changed
			if (params[argNum] != null) {
	        	CL.clReleaseMemObject(params[argNum]); // TODO is this necessary for memory conservation?
	        }
			params[argNum] = CL.clCreateBuffer(context, accessType,
					array.length * Sizeof.cl_int, null, null);
			
			if (accessType == CL.CL_MEM_READ_ONLY) {
				// This step takes a long time
		        CL.clEnqueueWriteBuffer(commandQueue, params[argNum], true, 0,
		        		array.length * Sizeof.cl_int, Pointer.to(array), 0, null, null);
		        
		        CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(params[argNum]));
		        
			} else if (accessType == CL.CL_MEM_WRITE_ONLY || accessType == CL.CL_MEM_READ_WRITE) {
				int index = findWriteArgumentIndex(argNum);
				
				// If this is the first time this argument has been written to
				if (index == -1) {
		        	writeArgumentNum[writeIndex] = argNum;
		        	writeArrayBuffers[writeIndex] = params[argNum];
		        	writeArrayType[writeIndex] = ArrayType.INT;
		        	arrayToFillInt[writeIndex] = array;
		        	writeIndex++;
				} else { // This has been written to before
					writeArgumentNum[index] = argNum;
		        	writeArrayBuffers[index] = params[argNum];
		        	writeArrayType[index] = ArrayType.INT;
		        	arrayToFillInt[index] = array;
				}
	        }
		} else if (arg instanceof byte[]) {
			final byte[] array = (byte[])arg;
			
			// Release this memory if it has changed
			if (params[argNum] != null) {
	        	CL.clReleaseMemObject(params[argNum]); // TODO is this necessary for memory conservation?
	        }
			params[argNum] = CL.clCreateBuffer(context, accessType,
					array.length * Sizeof.cl_uchar, null, null);
			
			if (accessType == CL.CL_MEM_READ_ONLY) {
				// This step takes a long time
		        CL.clEnqueueWriteBuffer(commandQueue, params[argNum], true, 0,
		        		array.length * Sizeof.cl_uchar, Pointer.to(array), 0, null, null);
		        
		        CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(params[argNum]));
		        
			} else if (accessType == CL.CL_MEM_WRITE_ONLY || accessType == CL.CL_MEM_READ_WRITE) {
				int index = findWriteArgumentIndex(argNum);
				
				// If this is the first time this argument has been written to
				if (index == -1) {
					writeArgumentNum[writeIndex] = argNum;
		        	writeArrayBuffers[writeIndex] = params[argNum];
		        	writeArrayType[writeIndex] = ArrayType.BYTE;
		        	arrayToFillByte[writeIndex] = array;
		        	writeIndex++;
				} else { // This has been written to before
					writeArgumentNum[index] = argNum;
		        	writeArrayBuffers[index] = params[argNum];
		        	writeArrayType[index] = ArrayType.BYTE;
		        	arrayToFillByte[index] = array;
				}
	        }
		} else if (arg instanceof BufferedImage) {
			BufferedImage image = (BufferedImage)arg;
			if (image.getType() != BufferedImage.TYPE_INT_RGB) {
				new Exception("Image must be of type TYPE_INT_RGB").printStackTrace();
				System.exit(1);
				return;
			}
			
			// Release this memory if it has changed
			if (params[argNum] != null) {
	        	CL.clReleaseMemObject(params[argNum]);
	        }
			params[argNum] = CL.clCreateBuffer(context, accessType, 
					image.getWidth() * image.getHeight() * Sizeof.cl_int, null, null);
			
			if (accessType == CL.CL_MEM_READ_ONLY) {
				final DataBufferInt dataBuffer = (DataBufferInt) image.getRaster().getDataBuffer();
				final int[] array = dataBuffer.getData();
				
				// This step takes a long time (90 milliseconds)
		        CL.clEnqueueWriteBuffer(commandQueue, params[argNum], true, 0,
		        		array.length * Sizeof.cl_int, Pointer.to(array), 0, null, null);
		        CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(params[argNum]));
	        
			} else if (accessType == CL.CL_MEM_WRITE_ONLY || accessType == CL.CL_MEM_READ_WRITE) {
				int index = findWriteArgumentIndex(argNum);
				
				// If this is the first time this argument has been written to
				if (index == -1) {
					writeArgumentNum[writeIndex] = argNum;
		        	writeArrayBuffers[writeIndex] = params[argNum];
		        	writeArrayType[writeIndex] = ArrayType.BUFFERED_IMAGE;
		        	imageToFill[writeIndex] = image;
		        	writeIndex++;
				} else { // This has been written to before
					writeArgumentNum[index] = argNum;
		        	writeArrayBuffers[index] = params[argNum];
		        	writeArrayType[index] = ArrayType.BUFFERED_IMAGE;
		        	imageToFill[index] = image;
				}
	        }
		} else if (arg == null) {
			new Exception("Argument must not be null").printStackTrace();
			System.exit(1);
		} else {
			new Exception("Invalid argument type: " + arg.getClass().getSimpleName()).printStackTrace();
			System.exit(1);
		}
	}
	
	// Step 5: This is called to process the data.
	// The given WRITE arrays will be populated with data.
	public void executeKernel() {
		
		// Check if the work sizes are compatible
		if (localWorkSize.length != globalWorkSize.length) {
			new Exception("Dimension of local work group must equal dimension of global work group!").printStackTrace();
			System.exit(1);
			return;
		}
		
		// TODO this call is not necessary upon every execution
		// TODO this is only necessary when the pointers change
		for (int i = 0; i < writeIndex; i++) {
			CL.clSetKernelArg(kernel, writeArgumentNum[i], Sizeof.cl_mem, Pointer.to(writeArrayBuffers[i]));
		}
		
		// This does the actual processing
		CL.clEnqueueNDRangeKernel(commandQueue, kernel, globalWorkSize.length,
				null, globalWorkSize, localWorkSize, 0, null, null);
		
		// Wait for the computation to finish (not necessary though)
		CL.clFinish(commandQueue);
		
		// Fill in the float arrays
		for (int i = 0; i < writeIndex; i++) {
			final ArrayType type = writeArrayType[i];
			
			if (type == ArrayType.FLOAT) {
				CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffers[i], CL.CL_TRUE, 0,
						Sizeof.cl_float * arrayToFillFloat[i].length,
						Pointer.to(arrayToFillFloat[i]), 0, null, null);
			} else if (type == ArrayType.INT) {
				CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffers[i], CL.CL_TRUE, 0,
						Sizeof.cl_int * arrayToFillInt[i].length,
						Pointer.to(arrayToFillInt[i]), 0, null, null);
			} else if (type == ArrayType.BYTE) {
				CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffers[i], CL.CL_TRUE, 0,
						Sizeof.cl_char * arrayToFillByte[i].length,
						Pointer.to(arrayToFillByte[i]), 0, null, null);
			} else if (type == ArrayType.LONG) {
				new Exception("Long arrays not implemented yet");
				System.exit(1);
			} else if (type == ArrayType.DOUBLE) {
				new Exception("Double arrays not implemented yet");
				System.exit(1);
			} else if (type == ArrayType.BUFFERED_IMAGE) {
				DataBufferInt dataBuffer = (DataBufferInt) imageToFill[i].getRaster().getDataBuffer();
				final int[] imageData = dataBuffer.getData();
				CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffers[i], CL.CL_TRUE, 0,
						Sizeof.cl_float * imageData.length,
						Pointer.to(imageData), 0, null, null);
			} else {
				new Exception("Invalid buffer mapping!");
				System.exit(1);
			}
		}
		
		//CL.clFinish(commandQueue);
		
		/*
		// This takes a long time.
		// Here, data is sent back to the RAM.
		if (arrayToFillFloat != null) {
			CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffer, CL.CL_TRUE, 0,
					Sizeof.cl_float * arrayToFillFloat.length,
					Pointer.to(arrayToFillFloat), 0, null, null);
		} else if (arrayToFillInt != null) {
			CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffer, CL.CL_TRUE, 0,
					Sizeof.cl_uint * arrayToFillInt.length,
					Pointer.to(arrayToFillInt), 0, null, null);
		} else if (arrayToFillByte != null) {
			CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffer, CL.CL_TRUE, 0,
					Sizeof.cl_uchar * arrayToFillByte.length,
					Pointer.to(arrayToFillByte), 0, null, null);
		} else if (imageToFill != null) {
			DataBufferInt dataBuffer = (DataBufferInt) imageToFill.getRaster().getDataBuffer();
			final int[] imageData = dataBuffer.getData();
			CL.clEnqueueReadBuffer(commandQueue, writeArrayBuffer, CL.CL_TRUE, 0,
					Sizeof.cl_float * imageData.length,
					Pointer.to(imageData), 0, null, null);
		} else {
			new Exception("No output argument set!");
			System.exit(1);
		}
		*/
	}
	
	// Return whether this kernel argument has already been set
	public boolean isArgumentSet(int num) {
		return params[num] != null;
	}
	
	// Step 6: Delete unused data
	// Call this to release all resources that were used for this instance
	public void dispose() {
		
		// Finish all operations on the command queue
		if (commandQueue != null) {
			CL.clFinish(commandQueue);
			CL.clFlush(commandQueue);
			CL.clReleaseCommandQueue(commandQueue);
		}
		
		// Release all of the arguments
		for (int i = 0; i < params.length; i++) {
			if (params[i] != null) {
				CL.clReleaseMemObject(params[i]);
				params[i] = null;
			}
		}
		
		// Release the main things
		CL.clReleaseKernel(kernel);
		CL.clReleaseProgram(program);
		
		if (context != null) {
			CL.clReleaseContext(context);
		}
		
		// Clear the data for Java
		writeArrayBuffers = null;
		arrayToFillFloat = null;
		arrayToFillInt = null;
		imageToFill = null;
		kernel = null;
		commandQueue = null;
		context = null;
		initialized = false;
	}
	
	// Get the number of bytes of global memory in this GPU
	public static long getGlobalMemory() {
		if (device == null) {
			new Exception("GPU has not been initialized").printStackTrace();
			System.exit(1);
		}
		long[] value = {0};
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_GLOBAL_MEM_SIZE,
				Sizeof.cl_ulong, Pointer.to(value), new long[] {value.length});
		return value[0];
	}
	
	// Get the number of bytes of global memory in this GPU
	public static long getMaxMemAllocSize() {
		if (device == null) {
			new Exception("GPU has not been initialized").printStackTrace();
			System.exit(1);
		}
		long[] value = {0};
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE,
				Sizeof.cl_ulong, Pointer.to(value), new long[] {value.length});
		return value[0];
	}
	
	/*
	// Returns the value of the device info parameter with the given name
	private static String getDeviceString(cl_device_id device, int paramName) {
		// Obtain the length of the string that will be queried
		long size[] = new long[1];
		CL.clGetDeviceInfo(device, paramName, 0, null, size);
		
		// Create a buffer of the appropriate size and fill it with the info
		byte buffer[] = new byte[(int) size[0]];
		CL.clGetDeviceInfo(device, paramName, buffer.length,
				Pointer.to(buffer), null);
		
		// Create a string from the buffer (excluding the trailing \0 byte)
		return new String(buffer, 0, buffer.length - 1);
	}
	
	// Get some integer value from the device properties
	private static int getDeviceInt(cl_device_id device, int paramName) {
		int values[] = new int[1];
		CL.clGetDeviceInfo(device, paramName, Sizeof.cl_int * 1, Pointer.to(values), null);
		return values[0];
	}
	*/
	
	// Determine the index in the write array of the given argument number
	private int findWriteArgumentIndex(int argNum) {
		for (int i = 0; i < writeArgumentNum.length; i++) {
			if (writeArgumentNum[i] == argNum) {
				return i;
			} else if (writeArgumentNum[i] == -1) {
				return -1;
			}
		}
		
		return -1;
	}
	
	// Return whether the GPU has been initialized
	public static boolean isInitialized() {
		return initialized;
	}
	
	static void print(Object o) {
		System.out.println(o);
	}
}
