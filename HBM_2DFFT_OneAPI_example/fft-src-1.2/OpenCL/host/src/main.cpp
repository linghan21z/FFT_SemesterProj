/*
Copyright (c) 2020, 2021 by BittWare, Inc., A Molex Company 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


/*
 * This host sets up 16 HBM memories required by the OpenCL 2D FFT implementation.
 * Data accessed as two complex pairs from each HBM memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <CL/cl_ext_intelfpga.h>


// Define the size of the transform. Each dimension is the same.
#define SIZE 1024

// Define number of HBMs in use, this must be 16 to match the OpenCL kernel.
#define HBMS 16
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id *device; 
cl_context context = NULL;
cl_command_queue queue; 
cl_program program = NULL;
cl_kernel kernel; 


// OpenCL memory buffers
cl_mem input_buf[HBMS]; 

// Buffers for input and output to kernel
scoped_aligned_ptr<float> input[HBMS];
scoped_aligned_ptr<float> output[HBMS]; 

// Output buffer
scoped_array<float> ref_output; 

// Control whether the fast emulator should be used.
bool use_fast_emulator = false;

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
	// Initialize OpenCL.
	if(!init_opencl()) {
	return -1;
	}

	// Initialize the problem data.
	// Requires the number of devices to be known.
	init_problem();

	// Run the kernel.
	run();

	// Free the resources allocated
	cleanup();

	return 0;
}

// Initializes the OpenCL objects.
bool init_opencl() {
	cl_int status;

	printf("Initializing OpenCL\n");

	if(!setCwdToExeDir()) {
	return false;
	}

	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");

	if(platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device.
	device = (getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  %s\n", getDeviceName(device[i]).c_str());
	}

	// Create the context.
	context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the program for all device. Use the first device as the
	// representative device (assuming all device are of the same type).
  
	std::string binary_file = getBoardBinaryFile("device/fft2d_mx", device[0]);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");


	// Command queue.
	queue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Kernel.
	const char *kernel_name = "FFT_2d_hbm";
	kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");


	// 1024x1024 FFT with complex pair input data and double buffered, then shared over
	// 16 HBM memories.
	#define BUFFER_SIZE (1024*1024*2*4*2/16)

	// Input buffers.
	for (int h = 0; h < 16; h++)
	{
		printf("allocate hbm%d resources\n",h);
		input_buf[h] = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_HETEROGENEOUS_INTELFPGA, 
			BUFFER_SIZE * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input A");
		input[h].reset(2*2*SIZE*SIZE/HBMS);
		output[h].reset(2*2*SIZE*SIZE/HBMS);
	}

	printf("init done\n");
	return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  // Initialize input into 2DFFT
  for (int j = 0; j <(SIZE); j+=HBMS)
  {
	  for (int i = 0; i <(SIZE); i++)
	  for (int h = 0; h < HBMS; h++)
	  {
     		// complex input
		float r = sin((float)(j+h) / 32.0f)* sin((float)i / 32.0f);
		int index = i + (SIZE * (j + h));
		int hbm_index = (i + (SIZE * (j >> 4)))*2;
		input[h][hbm_index] = r;//i;//r;
		input[h][hbm_index + 1] = 0.0f;	
	  }    
   }
}

void run() {
    cl_int status;

    const double start_time = getCurrentTimestamp();

    // Launch the problem for each device.
    cl_event kernel_event;
    cl_event finish_event;


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event;

    printf("Buffers enqueued\n");

    // Set kernel arguments.
    unsigned argi = 0;

    for (int h = 0; h < HBMS; h++)
    {
        status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_buf[h]);
    	checkError(status, "Failed to set argument %d", argi - 1);
    }

    for (int h = 0; h < HBMS; h++)
    {
        status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_buf[h]);
    	checkError(status, "Failed to set argument %d", argi - 1);
    }

    // Populate input data
    for (int h = 0; h < HBMS; h++)
    {
        status = clEnqueueWriteBuffer(queue, input_buf[h], CL_TRUE,
         0, (2*2*SIZE*SIZE/HBMS) * sizeof(float), input[h], 0, NULL, &write_event);
        checkError(status, "Failed to transfer input ");
    }

    int fft_pass = 0;
    const size_t global_work_size = 1;
    #define ITS 1
    for (int it = 0; it < ITS; it++)
    {
      status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,&global_work_size, NULL, 0, NULL, &kernel_event);
      clWaitForEvents(1, &kernel_event);
    }
    checkError(status, "Failed to launch kernel");

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    double secs = nanoSeconds*1e-9;
    // 1024 x 1024 complex numbers
    // 2 passes
    // read and write
        double gbytes = 2* ((2*1024*1024*2*4)/secs)*1e-9*ITS;
        printf("OpenCl Execution time is: %0.3f milliseconds \n",nanoSeconds / 1000000.0);
        printf("Estimated bandwidth GBytes/Sec = %f\n",gbytes);

    	printf("kernel run!\n");
	// Read the result. This the final operation. 
	for (int h = 0; h < HBMS; h++)
	    status = clEnqueueReadBuffer(queue, input_buf[h], CL_TRUE,
        	0, (2*SIZE*SIZE/HBMS) * sizeof(float), output[h], 0, NULL,NULL);


	const double end_time = getCurrentTimestamp();

	// Wall-clock time taken.
	printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
}

// Free the resources allocated during initialization
void cleanup() {
	if(kernel) {
		clReleaseKernel(kernel);
	}
	if(queue) {
		clReleaseCommandQueue(queue);
	}
	for (int h = 0; h < HBMS; h++)
	{
		if(input_buf[h]) {
			clReleaseMemObject(input_buf[h]);
		}
	}
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
}

