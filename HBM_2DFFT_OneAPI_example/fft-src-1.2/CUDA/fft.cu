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
	This code benchmarks FFT performance on a cuda device. It requires the cuda SDK and cufft to be 
	installed. 
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <iostream>

// includes, project
#include <cufft.h>

// Complex data type
typedef float2 Complex;

void run1DTest(int argc, char** argv, const int nx, const int batch);
void run2DTest(int argc, char** argv, const int nx, const int ny, const int batch);


// Queury device
void Query() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
   
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

   size_t free,total;
   cudaMemGetInfo(&free,&total);
   printf("free mem = %f MBYtes : total mem = %f MBytes\n",(float)free/(1024*1024),(float)total/(1024*1024));
  }
  printf("Choose device 0\n"); 
  cudaSetDevice(0);
}


int main(int argc, char** argv)
{
    // Query available cuda devices
    Query();
    // Run 1D benchmarkds
    printf("1D FFTs batch 1024\n");
    run1DTest(argc, argv, 16384*2, 1024);
    run1DTest(argc, argv, 16384, 1024);
    run1DTest(argc, argv, 8192, 1024);
    run1DTest(argc, argv, 4096, 1024);
    run1DTest(argc, argv, 2048, 1024);
    run1DTest(argc, argv, 1024, 1024);
    run1DTest(argc, argv, 512, 1024);
    run1DTest(argc, argv, 256, 1024);
    run1DTest(argc, argv, 128, 1024);

    // Run 2D benchmarks with no batching.
    printf("\n2D FFTs no batch\n");
    run2DTest(argc, argv, 4096, 4096, 1);
    run2DTest(argc, argv, 2048, 2048, 1);
    run2DTest(argc, argv, 1024, 1024, 1);
    run2DTest(argc, argv, 512, 512, 1);
    run2DTest(argc, argv, 256, 256, 1);
    run2DTest(argc, argv, 128, 128, 1);
    run2DTest(argc, argv, 64, 64, 1);
    run2DTest(argc, argv, 32, 32, 1);
    run2DTest(argc, argv, 16, 16, 1);

    // Run 2D benchmarks with batching enabled.
    printf("\n2D FFTs batch 20\n");
    run2DTest(argc, argv, 4096, 4096, 20);
    run2DTest(argc, argv, 2048, 2048, 20);
    run2DTest(argc, argv, 1024, 1024, 20);
    run2DTest(argc, argv, 512, 512, 20);
    run2DTest(argc, argv, 256, 256, 20);
    run2DTest(argc, argv, 128, 128, 20);
    run2DTest(argc, argv, 64, 64, 20);
    run2DTest(argc, argv, 32, 32, 20);
    run2DTest(argc, argv, 16, 16, 20);
}


#define NRANK 2
void run2DTest(int argc, char** argv, const int nx, const int ny, const int batch)
{
	const int ISTRIDE = 2;
	const int OSTRIDE = 2;
	const int IX = (nx+2);
	const int IY = (ny+1);
	const int OX = (nx+3);
	const int OY = (ny+4);
	const int IDIST = (IX*IY*ISTRIDE+3);
	const int ODIST = (OX*OY*OSTRIDE+5);
	int isize = IDIST * batch;
	int osize = ODIST * batch;
	int inembed[NRANK] = {IX, IY}; //  pointer that indicates storage dimensions of input data
	int onembed[NRANK] = {OX, OY}; //  pointer that indicates storage dimensions of output data
	// Running CUDA 2D FFT test.
	cufftHandle plan;
	cufftComplex *data;
	cufftComplex *data_out;
	int n[NRANK];
	n[0] = nx;
	n[1] = ny;

	cudaMalloc((void**)&data, sizeof(cufftComplex)*isize);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	cudaMalloc((void**)&data_out, sizeof(cufftComplex)*osize);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	// Allocate host memory for the filter
	Complex* h_filter_kernel = (Complex*)malloc(sizeof(Complex) * (isize));
	Complex* h_filter_kernel_out = (Complex*)malloc(sizeof(Complex) * (osize));
	    // Initalize the memory for the filter
	    for (unsigned int i = 0; i < isize; ++i) {
	        h_filter_kernel[i].x = 1;
	        h_filter_kernel[i].y = 0;
	    }

   	// Copy host memory to device
   	cudaMemcpy(data,h_filter_kernel, nx*ny*batch*sizeof(cufftComplex), cudaMemcpyHostToDevice);

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to copy data from host\n");
		return;	
	}

	float memsettime;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Create a 2D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		 inembed,ISTRIDE,IDIST,
		 onembed,OSTRIDE,ODIST,
					  CUFFT_C2C,batch) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return;	
	}
	
	cudaDeviceSynchronize();

	clock_t begin = clock();	
	cudaEventRecord(start,0);
	if (cufftExecC2C(plan, data, data_out,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return;		
	}
	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;	
	}
	cudaEventRecord(stop,0);
	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	// Copy device memory to host
	cudaMemcpy( h_filter_kernel_out , data_out, nx*ny*batch*sizeof(Complex),
               	   cudaMemcpyDeviceToHost);


	cudaEventElapsedTime(&memsettime, start, stop);

	cudaEventDestroy(start);

	cudaEventDestroy(stop);

	printf("Total bytes processed in GBytes/Sec - average time - dimension - batch size= : %f : %f : %d : %d\n",(1e-09*(float)(nx*ny*batch*sizeof(Complex)*4))/time_spent,time_spent/batch,nx,batch);


	if (cudaDeviceSynchronize() != cudaSuccess){
  		
	fprintf(stderr, "Cuda error: Failed to synchronize\n");
	   	return;
	}	

	//for (int i = 0; i < 10; i++)
	//	printf("data = (%f,%f)\n", h_filter_kernel_out[i].x, h_filter_kernel_out[i].y);

	cufftDestroy(plan);
	cudaFree(data);
	cudaFree(data_out);

	free(h_filter_kernel);
    	free(h_filter_kernel_out);
}



void run1DTest(int argc, char** argv, const int nx, const int batch)
{
	// Running CUDA 2D FFT test.
	cufftHandle plan;
	cufftComplex *data;
	cufftComplex *data_out;
	int n[1];
	n[0] = nx;

	cudaMalloc((void**)&data, sizeof(cufftComplex)*nx*batch);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	cudaMalloc((void**)&data_out, sizeof(cufftComplex)*nx*batch);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	// Allocate host memory for the filter
	Complex* h_filter_kernel = (Complex*)malloc(sizeof(Complex) * (nx*batch));
	Complex* h_filter_kernel_out = (Complex*)malloc(sizeof(Complex) * (nx*batch));
	    // Initalize the memory for the filter
	    for (unsigned int i = 0; i < nx*batch; ++i) {
	        h_filter_kernel[i].x = 1;
	        h_filter_kernel[i].y = 0;
	        h_filter_kernel_out[i].x = 0xffffff;
	        h_filter_kernel_out[i].y = 0;
	    }

   	// Copy host memory to device
   	cudaMemcpy(data,h_filter_kernel, nx*batch*sizeof(cufftComplex), cudaMemcpyHostToDevice);

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to copy data from host\n");
		return;	
	}

	float memsettime;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Create a 1D FFT plan. */
	if (cufftPlanMany(&plan, 1, n,
					  NULL, 1, 0,
					  NULL, 1, 0,
					  CUFFT_C2C,batch) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return;	
	}
	
	cudaDeviceSynchronize();
	clock_t start_cpu = clock();
	cudaEventRecord(start,0);
	if (cufftExecC2C(plan, data, data_out,CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return;		
	}
	cudaEventRecord(stop,0);
	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;	
	}
	clock_t end_cpu = clock();
	double time_spent = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;


	for (int i = 0; i < 10; i++)
	{
		h_filter_kernel_out[i].x = 0xfffff;
		h_filter_kernel_out[i].y = 0;
	}

	
	// Copy device memory to host
	cudaMemcpy( h_filter_kernel_out , data_out, nx*batch*sizeof(Complex),
               	   cudaMemcpyDeviceToHost);
	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to copy data to host\n");
		return;	
	}


	cudaEventElapsedTime(&memsettime, start, stop);

	cudaEventDestroy(start);

	cudaEventDestroy(stop);

	
	printf("Total bytes processed in GBytes/Sec - average time - dimension - batch size= : %f : %f : %d : %d\n",(1e-09*(float)(nx*batch*sizeof(Complex)*2))/time_spent,time_spent/batch,nx,batch);


	if (cudaDeviceSynchronize() != cudaSuccess){  		
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	   	return;
	}	

	cufftDestroy(plan);
	cudaFree(data);
	cudaFree(data_out);

	free(h_filter_kernel);
    	free(h_filter_kernel_out);
}



