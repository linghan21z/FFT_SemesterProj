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

#include "CompareToFFTW.h"
#include "math.h"


// Find the maximum difference seen so far and update a histogram of the errors in 
// percentage of difference between FPGA and FFTW implementation.

#define DIFFERENCES_TO_DISPLAY 0
int differences = 0;

void UpdateMaxDiff(float a, float b, float &max,unsigned int histo[100],int index)
{
	 float diff = 0;
	 // Ignore sub normal or zero numbers for difference calculation.
	 unsigned int i = *((unsigned int*)(&a));
  	 // Remove this check to include small numbers in differences
         if ((fabs(a) > 0.001) || (fabs(b) > 0.001))
	 {
	   if (i&0x7f800000) // Then not subnormal or zero.
	   {
		 diff = (a - b);
		 diff = fabs(diff/a)*100.0f;
	   }
	   else
	   {
		 diff = 0;
	   }
	 }
	 if (diff > 99) diff = 99;
	 histo[(int)(diff)]++;
	 if (diff > max)
	 {
		 max = diff;
		 if (differences < DIFFERENCES_TO_DISPLAY)
		 {
			differences++;
			printf("index=%d: cpu=%f : fpga=%f: diff =%f\n",index,a,b,diff);
		 }
	 }
	 
}

// Run an FFTW equivalent 2D FFT and compare the results.
void Compare(fftw_complex *input,fftw_complex *output_fpga,fftw_complex *output_cpu,int SIZE,char *str)
{
	// Generate output file with FFT comparison result
	FILE *file = fopen(str,"w+");
	if (!file)
	{
		printf("Failed to create comparison output file (%s%d)\n",__FILE__,__LINE__);
		exit(1);
	}
	// Create FFTW plan for 2D FFT
	fftw_plan p = fftw_plan_dft_2d(SIZE,SIZE,input,output_cpu,FFTW_FORWARD, FFTW_ESTIMATE );

	// Run the FFTW 2D FFT
	fftw_execute(p);

	// Measure maximum percentage
	float max_diff = 0;

	// Initialise Histogram
	 unsigned int errors = 0;
	 unsigned int histo[100];
	 for (int i = 0; i < 100; i++)
		 histo[i] = 0;

	 // Compare values and store in file
	 // Note there will be differences due to floating point operator order and the implementation
	 // of the floating point logic in the diiferent architectures.
	 for (int i = 0; i < SIZE*SIZE; i++)
	 {
		 fftw_complex fpga;
		 fpga[0] = output_fpga[i][0];
		 fpga[1] = output_fpga[i][1];
		 fftw_complex cpu;
		 cpu[0] = output_cpu[i][0];
		 cpu[1] = output_cpu[i][1];

		 UpdateMaxDiff(cpu[0],fpga[0],max_diff,histo,i);
		 UpdateMaxDiff(cpu[1],fpga[1],max_diff,histo,i);
	 }

	 fprintf(file,"Distribution of differences to the nearest percentile...\n");
	 for (int i = 0; i < 100; i++)
	 {
		 fprintf(file,"\terrors in percentile %d = %d\n",i,histo[i]);
	 }
	 fclose(file);
	 printf("For FFT comparison results see file %s\n",str);
}



