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

#ifndef FFT_2D_HBM_H
#define FFT_2D_HBM_H

/*
	Include type definitions for HBM and FFT
*/
#include "types.h"
#include "FFT_1d.h"


__kernel
void FFT_2d_hbm(
	__HBMf4__("HBM0")  in0,
	__HBMf4__("HBM1")  in1,
	__HBMf4__("HBM2")  in2,
	__HBMf4__("HBM3")  in3,
	__HBMf4__("HBM4")  in4,
	__HBMf4__("HBM5")  in5,
	__HBMf4__("HBM6")  in6,
	__HBMf4__("HBM7")  in7,
	__HBMf4__("HBM8")  in8,
	__HBMf4__("HBM9")  in9,
	__HBMf4__("HBM10")  in10,
	__HBMf4__("HBM11")  in11,
	__HBMf4__("HBM12")  in12,
	__HBMf4__("HBM13")  in13,
	__HBMf4__("HBM14")  in14,
	__HBMf4__("HBM15")  in15,
	__HBMf4__("HBM0")  out0,
	__HBMf4__("HBM1")  out1,
	__HBMf4__("HBM2")  out2,
	__HBMf4__("HBM3")  out3,
	__HBMf4__("HBM4")  out4,
	__HBMf4__("HBM5")  out5,
	__HBMf4__("HBM6")  out6,
	__HBMf4__("HBM7")  out7,
	__HBMf4__("HBM8")  out8,
	__HBMf4__("HBM9")  out9,
	__HBMf4__("HBM10")  out10,
	__HBMf4__("HBM11")  out11,
	__HBMf4__("HBM12")  out12,
	__HBMf4__("HBM13")  out13,
	__HBMf4__("HBM14")  out14,
	__HBMf4__("HBM15")  out15);

__kernel
void FFT_2d_hbm_b(
	__HBMf4__("HBM16")  in0,
	__HBMf4__("HBM17")  in1,
	__HBMf4__("HBM18")  in2,
	__HBMf4__("HBM19")  in3,
	__HBMf4__("HBM10")  in4,
	__HBMf4__("HBM21")  in5,
	__HBMf4__("HBM22")  in6,
	__HBMf4__("HBM23")  in7,
	__HBMf4__("HBM24")  in8,
	__HBMf4__("HBM25")  in9,
	__HBMf4__("HBM26")  in10,
	__HBMf4__("HBM27")  in11,
	__HBMf4__("HBM28")  in12,
	__HBMf4__("HBM29")  in13,
	__HBMf4__("HBM30")  in14,
	__HBMf4__("HBM31")  in15,
	__HBMf4__("HBM16")  out0,
	__HBMf4__("HBM17")  out1,
	__HBMf4__("HBM18")  out2,
	__HBMf4__("HBM19")  out3,
	__HBMf4__("HBM20")  out4,
	__HBMf4__("HBM21")  out5,
	__HBMf4__("HBM22")  out6,
	__HBMf4__("HBM23")  out7,
	__HBMf4__("HBM24")  out8,
	__HBMf4__("HBM25")  out9,
	__HBMf4__("HBM26")  out10,
	__HBMf4__("HBM27")  out11,
	__HBMf4__("HBM28")  out12,
	__HBMf4__("HBM29")  out13,
	__HBMf4__("HBM30")  out14,
	__HBMf4__("HBM31")  out15);

#endif
