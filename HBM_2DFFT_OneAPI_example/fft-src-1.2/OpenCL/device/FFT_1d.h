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

#ifndef FFT_1D_H
#define FFT_1D_H

#include "types.h"
#define MAX_DELAY 512

/*
	1024 point 1D FFT
*/
#define FFT_1D_PIPE_LINE_DELAY (511+512+512)

/*
	1024 1D FFT pipeline stage
*/
void Stage_fpga(float2 a, float2 b, float2* a_out, float2* b_out, unsigned short index, short Delay, short A, short n, short mask,
	float2 delay_a[MAX_DELAY],
	float2 delay_b[MAX_DELAY * 2],
	float2 delay_b_n[MAX_DELAY],
	unsigned short delay_index[MAX_DELAY]);

void final_delay(float2 a, float2 b, float2* a_out, float2* b_out, unsigned short index, short Delay, short A, short n, short mask,
	float2 delay_a[MAX_DELAY],
	float2 delay_b[MAX_DELAY * 2],
	float2 delay_b_n[MAX_DELAY],
	unsigned short delay_index[MAX_DELAY]);

/*
	1024 1D FFT pipeline. One word in one word out after FFT_1D_PIPE_LINE_DELAY
*/
float4 FFT_1d_1024_pipeline(float4 in,
	float2 delay_a[11][MAX_DELAY], // Gets delayed LOG2N
	float2 delay_b[11][MAX_DELAY * 2], // Gets delayed LOG2N*2
	float2 delay_b_n[11][MAX_DELAY], // Gets delayed LOG"N
	unsigned short delay_index[11][MAX_DELAY], // Delay index by LOGN!
	float2 A[2048],
	float2 B[2048],
	float2 QA[1024],
	float2 QB[1024],
	unsigned short i);

void FFT_1d_1024(float4* in, float4* out);

/*
 1024 FFT output sequence test to check pipeline works for multiple FFT's in a row
*/
__kernel
void FFT_1d_1024_sequence_test(__HBMf4__("HBM0") in, __HBMf4__("HBM0") out, int number_of_sequential_ffts);

#endif
