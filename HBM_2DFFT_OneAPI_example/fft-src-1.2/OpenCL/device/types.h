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


#ifndef TYPES_H
#define TYPES_H

/*
	Defines for C++ to ignore OpenCL key words
*/
#ifndef INTEL_CL
#define __kernel
#define __attribute__(x)
#endif

#ifndef INTEL_CL
#include <stdio.h>

/*
	Local C++ version of float2 and float4 vectors used by FPGA OpenCL code
*/

typedef struct
{
	float s[2];
	float& operator[](int index) {
		return s[index];
	}
}float2;

typedef struct
{
	float s[4];
	float& operator[](int index) {
		return s[index];
	}
}float4;
#endif

// #defines for declaring HBM attributes for FPGA compilation.
// Ports are declared as restricted and volatile to reduce logic overhead and routing congestion around HBMs
#ifndef INTEL_CL
#define __HBM__(__X__) float2*
#define __HBMf4__(__X__) float4*
#else
#define __HBM__(__X__) __global __attribute((buffer_location(__X__))) volatile float2 *restrict
#define __HBMf4__(__X__) __global __attribute((buffer_location(__X__))) volatile float4 *restrict
#endif

/*
	Define for size of FFT
	FFT uses sliding windows and pre-calculated FFT sizes. Therefore, the size of the FFT cannot
	be dynamically altered.
*/
#define SIZE 1024

#endif
