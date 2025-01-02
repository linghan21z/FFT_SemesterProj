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

#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "FFT_2d_hbm.h"

#include <array>
#include <iostream>
#include <stdio.h>

#include "CompareToFFTW.h"

void StripeIntoHMBs(fftw_complex *input,
	float* HBM0,
	float* HBM1,
	float* HBM2,
	float* HBM3,
	float* HBM4,
	float* HBM5,
	float* HBM6,
	float* HBM7,
	float* HBM8,
	float* HBM9,
	float* HBM10,
	float* HBM11,
	float* HBM12,
	float* HBM13,
	float* HBM14,
	float* HBM15)
{
	// FFT processes 16 rows in parallel.
	for (int y = 0; y < HBM_FFT_SIZE; y+=16)
		for (int x = 0; x < HBM_FFT_SIZE; x++)
		{
			for (int hh = 0; hh < 32; hh++)
			{
				int h = hh/2;
				int index = ((y+h)*HBM_FFT_SIZE) + x;
				float val;
				if ((hh&0x1) == 0)
					val = input[index][0];
				else
					val = input[index][1];
				int hbm_index = ((y/16)*(HBM_FFT_SIZE*2)) + (x*2) + (hh&0x1);
				float hbm_input = val;
				switch(h)
				{
					case 0 : HBM0[hbm_index] = hbm_input; break;
					case 1 : HBM1[hbm_index] = hbm_input; break;
					case 2 : HBM2[hbm_index] = hbm_input; break;
					case 3 : HBM3[hbm_index] = hbm_input; break;
					case 4 : HBM4[hbm_index] = hbm_input; break;
					case 5 : HBM5[hbm_index] = hbm_input; break;
					case 6 : HBM6[hbm_index] = hbm_input; break;
					case 7 : HBM7[hbm_index] = hbm_input; break;
					case 8 : HBM8[hbm_index] = hbm_input; break;
					case 9 : HBM9[hbm_index] = hbm_input; break;
					case 10 : HBM10[hbm_index] = hbm_input; break;
					case 11 : HBM11[hbm_index] = hbm_input; break;
					case 12 : HBM12[hbm_index] = hbm_input; break;
					case 13 : HBM13[hbm_index] = hbm_input; break;
					case 14 : HBM14[hbm_index] = hbm_input; break;
					case 15 : HBM15[hbm_index] = hbm_input; break;
					default : break;		
				}
			}
		}
}

void MergeHMBs(fftw_complex *output,
	float *HBM0,
	float *HBM1,
	float *HBM2,
	float *HBM3,
	float *HBM4,
	float *HBM5,
	float *HBM6,
	float *HBM7,
	float *HBM8,
	float *HBM9,
	float *HBM10,
	float *HBM11,
	float *HBM12,
	float *HBM13,
	float *HBM14,
	float *HBM15)
{
	// FFT processes 16 rows in parallel.
	for (int y = 0; y < HBM_FFT_SIZE; y+=16)
		// HBM memory is in the format of float4, so one complex pair per array index.
		for (int x = 0; x < HBM_FFT_SIZE; x++)
		{
			fftw_complex val;
			for (int hh = 0; hh < 32; hh++)
			{
				int h = hh/2;
				int index = ((y+h)*HBM_FFT_SIZE) + x;
				int hbm_index = ((y/16)*(HBM_FFT_SIZE*2)) + (x*2) + (hh&0x1);
				float hbm_output;

				switch(h)
				{
					case 0 : hbm_output = HBM0[hbm_index]; break;
					case 1 : hbm_output = HBM1[hbm_index]; break;
					case 2 : hbm_output = HBM2[hbm_index]; break;
					case 3 : hbm_output = HBM3[hbm_index]; break;
					case 4 : hbm_output = HBM4[hbm_index]; break;
					case 5 : hbm_output = HBM5[hbm_index]; break;
					case 6 : hbm_output = HBM6[hbm_index]; break;
					case 7 : hbm_output = HBM7[hbm_index]; break;
					case 8 : hbm_output = HBM8[hbm_index]; break;
					case 9 : hbm_output = HBM9[hbm_index]; break;
					case 10 : hbm_output = HBM10[hbm_index]; break;
					case 11 : hbm_output = HBM11[hbm_index]; break;
					case 12 : hbm_output = HBM12[hbm_index]; break;
					case 13 : hbm_output = HBM13[hbm_index]; break;
					case 14 : hbm_output = HBM14[hbm_index]; break;
					case 15 : hbm_output = HBM15[hbm_index]; break;
					default : break;		
				}

				if ((hh&0x1) == 0)
					output[index][0] = hbm_output;
				else
					output[index][1] = hbm_output;
				
			}
		}
}

float *HBMs_FFT_A[16];


// Create arrays for 1024x1024 2D FFT split over 16 HBM memories
// Data is striped one line after the other.
void PrepareFFTData(fftw_complex *input,float *HBM[16])
{
	  size_t byte_alignment = 1024; //Alignment affects data access speed
			// A 1024-byte alignment guarantees that rows of the FFT data or chunks processed by HBM modules are properly aligned for efficient parallel access.
			// It often matches the size of cache lines, memory pages, or the transfer size of direct memory access (DMA) units in such systems.
	  //aligned_alloc is a C standard library function for allocating memory with specific alignment requirements.
	  //Each HBM[i] is a pointer to dynamically allocated memory that will hold data for one High Bandwidth Memory (HBM) bank.
	  											 //This calculates the total memory needed for one HBM:
	  HBM[0] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  		//The result of aligned_alloc is a void* (generic pointer), which needs to be cast to the specific type float* because HBM is an array of float*.
	  HBM[1] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[2] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[3] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[4] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[5] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[6] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[7] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[8] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[9] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[10] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[11] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[12] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[13] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[14] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  HBM[15] = (float*)aligned_alloc(byte_alignment,sizeof(float)*4*(HBM_FFT_SIZE*HBM_FFT_SIZE/16));
	  StripeIntoHMBs(input,HBM[0],HBM[1],HBM[2],HBM[3],HBM[4],HBM[5],HBM[6],HBM[7],HBM[8],HBM[9],HBM[10],HBM[11],HBM[12],HBM[13],HBM[14],HBM[15]);
}


void RemoveFFTDataStriping(fftw_complex *output_fpga,float *HBM[16])
{
	  MergeHMBs(output_fpga,HBM[0],HBM[1],HBM[2],HBM[3],HBM[4],HBM[5],HBM[6],HBM[7],HBM[8],HBM[9],HBM[10],HBM[11],HBM[12],HBM[13],HBM[14],HBM[15]);	  
	  free(HBM[0]);
	  free(HBM[1]);
	  free(HBM[2]);
	  free(HBM[3]);
	  free(HBM[4]);
	  free(HBM[5]);
	  free(HBM[6]);
	  free(HBM[7]);
	  free(HBM[8]);
	  free(HBM[9]);
	  free(HBM[10]);
	  free(HBM[11]);
	  free(HBM[12]);
	  free(HBM[13]);
	  free(HBM[14]);
	  free(HBM[15]);
}

// Create a raw output inmage
void GenerateRawImage(fftw_complex *image, char *filename)
{
	unsigned short *raw_image = new unsigned short[HBM_FFT_SIZE*HBM_FFT_SIZE];
	// Use Red channel for real, blue for imag
	float max = 0; float min = 0;
	for (int i = 0; i < (HBM_FFT_SIZE*HBM_FFT_SIZE); i++)
	{
		float mag = (sqrt(image[i][0]*image[i][0] + image[i][1]*image[i][1]));
		if (mag > max) max = mag;
		if (mag < min) min = mag;
		if (max > 1.0f) // clamp
			max = 1.0f;
		if (min < -1.0f) // clamp
			min = -1.0f;
	}
	float scale = 65535.0f/(max-min);
	
	for (int i = 0; i < (HBM_FFT_SIZE*HBM_FFT_SIZE); i++)
	{
		float r = image[i][0];
		float im = image[i][1];
		float mag = (sqrt(image[i][0]*image[i][0] + image[i][1]*image[i][1]));
		mag = mag*scale;
		if (mag > 65535) mag = 65535;
		unsigned short j = 65535-mag;
		raw_image[i] = j;
	}
	FILE *raw;
	if ((raw = fopen(filename,"wb+")))
	{
		 fwrite(raw_image,sizeof(unsigned short),HBM_FFT_SIZE*HBM_FFT_SIZE,raw);
	}
	else
	{
		printf("Failed to create output image\n");
	}
	fclose(raw);
	delete[] raw_image;
	
}

#ifndef LIBRARY_ONLY
int main() {
	  // Create complex input and output arrays.
	  fftw_complex *input = new fftw_complex[HBM_FFT_SIZE*HBM_FFT_SIZE];
	  fftw_complex *output_fpga_a = new fftw_complex[HBM_FFT_SIZE*HBM_FFT_SIZE];
	  fftw_complex *output_cpu = new fftw_complex[HBM_FFT_SIZE*HBM_FFT_SIZE];


	  // Fill arrays with something!
	  for (int i = 0; i < HBM_FFT_SIZE*HBM_FFT_SIZE; i++)
	  {
		  int x , y;
		  x = i % 1024;
		  y = i / 1024;
		  // create disc
		  bool disc = false;
		  int disc_size = 10;
		  if (sqrt((x-512)*(x-512) +  (y-512)*(y-512)) < disc_size)
			  disc = true;
		  input[i][0] = disc?1.0f:0.0f;
		  input[i][1] = 0.0f;
		  output_fpga_a[i][0] = 0.0f;
		  output_fpga_a[i][1] = 0.0f;
		  output_cpu[i][0] = 0.0f;
		  output_cpu[i][1] = 0.0f;
	  }

	  GenerateRawImage(input,(char*)"image_in.raw");

	  // Prepare two sets of HBM inputs.
	  PrepareFFTData(input,HBMs_FFT_A); //allocate memory for each HBM bank
	  					//Then use StripeIntoHMBs() to fetch data into corresponding HBM banks

	  FFT2D_in_DPCPP_HBMs(HBMs_FFT_A);

	  RemoveFFTDataStriping(output_fpga_a,HBMs_FFT_A);
	  			//use MergeHMBs() to get data from HBM banks to output
				//then free(HBM[]) banks

	  // Compare results to fftw implementation
	  Compare(input,output_fpga_a,output_cpu,HBM_FFT_SIZE,(char*)"fpga_vs_fftw_A.txt");

	  GenerateRawImage(output_fpga_a,(char*)"image_out.raw");
	  printf("2D FFT complete!\n");
	  delete[] input;
	  delete[] output_fpga_a;
	  delete[] output_cpu;
	
  return 0;
}
#endif
