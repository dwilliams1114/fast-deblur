package gpuAbstraction;

// This class specifies a range of values which will be copied to the GPU and copied back.
// It goes with GPUProgram.

public class GPURange {
	protected final int start; // Includes 'start'
	protected final int end;   // Excludes 'end'
	protected final int size;
	
	public GPURange(int start, int end){
		this.start = start;
		this.end = end;
		this.size = end - start;
		
		if (size <= 0 || start < 0) {
			new Exception("GPURange must have a positive count. (Got " + size + ")").printStackTrace();
			System.exit(1);
		}
	}
	
	@Override
	public String toString() {
		return "[" + start + ", " + end + ")";
	}
}
