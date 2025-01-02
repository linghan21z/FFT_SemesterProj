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

#ifndef FFT_2D_HBM_H  //These ensure the header is included only once in a compilation unit, 
#define FFT_2D_HBM_H  //avoiding duplicate definitions.

using namespace sycl;  //brings SYCL constructs into the global namespace.

/*
	Include type definitions for HBM and FFT
*/
#include "FFT_1d.h"
// No Sycl equivalent for heterogeneous memories yet. 目前尚无与异构存储器等效的 Sycl
#define __HBMf4__(x) buffer<vec<float, 4>, 1>
//The macro __HBMf4__ defines a buffer type for SYCL, that stores 4-component floating-point vectors (vec<float, 4>).
//(1) indicates the dimension of the buffer. data is arranged in a simple linear array (like a single row of numbers).

#define HBM_FFT_SIZE 1024

// Define the problem size per HBM
//each HBM handles a portion of the problem. 
//The division by 16 implies the workload is split across 16 HBMs.
//calculates the total number of elements in the array for one HBM
constexpr size_t fft_array_size_hbm = (HBM_FFT_SIZE * HBM_FFT_SIZE / (16) );

// Create Typedef array structure for each HBM.  
//defines a fixed-size array containing fft_array_size_hbm #number of elements, 
//where each element is a vector of 4 floats.
typedef std::array<vec<float, 4>, fft_array_size_hbm> fft_Array_HBM;
//a pair of complex numbers (each complex number has 2 floats: real and imaginary)

// dpc++ function prototype for 2D FFT.
//Its input is a pointer to an array of 16 float* buffers, 
//where each corresponds to an HBM bank.
void FFT2D_in_DPCPP_HBMs(float *HBM[16]);

//16 HBM banks, which are used to split and process data in parallel

#endif
 