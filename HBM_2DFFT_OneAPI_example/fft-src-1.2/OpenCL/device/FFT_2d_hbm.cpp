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

#include "FFT_2d_hbm.h"
/*
	This code peforms a 2D FFT using 16 parallel 1D FFTs.
	Each FFT is fed by a single HBM using one pseudo port whilst the output is written to the other pseudo port.
	However, the HBM ports are transposed by the algorithm ready for the next pass.

	See the standalone transpose code "FFT_transpose_hbm.cpp" for further explanation of how the transpose works
*/

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
	__HBMf4__("HBM15")  out15)
{

	/*
		Burst buffer for output to HBMs.
		Used to improve burst length problems introduced from transpose function.

		One buffer for each of the 16 HBM outputs. Each buffer can store 4 rows of input
	*/
#pragma max_concurrency 1
	for (short pass = 0; pass < 2; pass++)
	{

	// Double buffer
#ifndef INTEL_CL
	float4* BurstBufferA0; BurstBufferA0 = new float4[4 * SIZE / 2];
	float4* BurstBufferA1; BurstBufferA1 = new float4[4 * SIZE / 2];
	float4* BurstBufferA2; BurstBufferA2 = new float4[4 * SIZE / 2];
	float4* BurstBufferA3; BurstBufferA3 = new float4[4 * SIZE / 2];
	float4* BurstBufferA4; BurstBufferA4 = new float4[4 * SIZE / 2];
	float4* BurstBufferA5; BurstBufferA5 = new float4[4 * SIZE / 2];
	float4* BurstBufferA6; BurstBufferA6 = new float4[4 * SIZE / 2];
	float4* BurstBufferA7; BurstBufferA7 = new float4[4 * SIZE / 2];
	float4* BurstBufferA8; BurstBufferA8 = new float4[4 * SIZE / 2];
	float4* BurstBufferA9; BurstBufferA9 = new float4[4 * SIZE / 2];
	float4* BurstBufferA10; BurstBufferA10 = new float4[4 * SIZE / 2];
	float4* BurstBufferA11; BurstBufferA11 = new float4[4 * SIZE / 2];
	float4* BurstBufferA12; BurstBufferA12 = new float4[4 * SIZE / 2];
	float4* BurstBufferA13; BurstBufferA13 = new float4[4 * SIZE / 2];
	float4* BurstBufferA14; BurstBufferA14 = new float4[4 * SIZE / 2];
	float4* BurstBufferA15; BurstBufferA15 = new float4[4 * SIZE / 2];

	float4* BurstBufferB0; BurstBufferB0 = new float4[4 * SIZE / 2];
	float4* BurstBufferB1; BurstBufferB1 = new float4[4 * SIZE / 2];
	float4* BurstBufferB2; BurstBufferB2 = new float4[4 * SIZE / 2];
	float4* BurstBufferB3; BurstBufferB3 = new float4[4 * SIZE / 2];
	float4* BurstBufferB4; BurstBufferB4 = new float4[4 * SIZE / 2];
	float4* BurstBufferB5; BurstBufferB5 = new float4[4 * SIZE / 2];
	float4* BurstBufferB6; BurstBufferB6 = new float4[4 * SIZE / 2];
	float4* BurstBufferB7; BurstBufferB7 = new float4[4 * SIZE / 2];
	float4* BurstBufferB8; BurstBufferB8 = new float4[4 * SIZE / 2];
	float4* BurstBufferB9; BurstBufferB9 = new float4[4 * SIZE / 2];
	float4* BurstBufferB10; BurstBufferB10 = new float4[4 * SIZE / 2];
	float4* BurstBufferB11; BurstBufferB11 = new float4[4 * SIZE / 2];
	float4* BurstBufferB12; BurstBufferB12 = new float4[4 * SIZE / 2];
	float4* BurstBufferB13; BurstBufferB13 = new float4[4 * SIZE / 2];
	float4* BurstBufferB14; BurstBufferB14 = new float4[4 * SIZE / 2];
	float4* BurstBufferB15; BurstBufferB15 = new float4[4 * SIZE / 2];

#else
	float4 BurstBufferA0[4 * SIZE / 2];
	float4 BurstBufferB0[4 * SIZE / 2];
	float4 BurstBufferA1[4 * SIZE / 2];
	float4 BurstBufferB1[4 * SIZE / 2];
	float4 BurstBufferA2[4 * SIZE / 2];
	float4 BurstBufferB2[4 * SIZE / 2];
	float4 BurstBufferA3[4 * SIZE / 2];
	float4 BurstBufferB3[4 * SIZE / 2];
	float4 BurstBufferA4[4 * SIZE / 2];
	float4 BurstBufferB4[4 * SIZE / 2];
	float4 BurstBufferA5[4 * SIZE / 2];
	float4 BurstBufferB5[4 * SIZE / 2];
	float4 BurstBufferA6[4 * SIZE / 2];
	float4 BurstBufferB6[4 * SIZE / 2];
	float4 BurstBufferA7[4 * SIZE / 2];
	float4 BurstBufferB7[4 * SIZE / 2];
	float4 BurstBufferA8[4 * SIZE / 2];
	float4 BurstBufferB8[4 * SIZE / 2];
	float4 BurstBufferA9[4 * SIZE / 2];
	float4 BurstBufferB9[4 * SIZE / 2];
	float4 BurstBufferA10[4 * SIZE / 2];
	float4 BurstBufferB10[4 * SIZE / 2];
	float4 BurstBufferA11[4 * SIZE / 2];
	float4 BurstBufferB11[4 * SIZE / 2];
	float4 BurstBufferA12[4 * SIZE / 2];
	float4 BurstBufferB12[4 * SIZE / 2];
	float4 BurstBufferA13[4 * SIZE / 2];
	float4 BurstBufferB13[4 * SIZE / 2];
	float4 BurstBufferA14[4 * SIZE / 2];
	float4 BurstBufferB14[4 * SIZE / 2];
	float4 BurstBufferA15[4 * SIZE / 2];
	float4 BurstBufferB15[4 * SIZE / 2];
#endif

	/*
		Delay buffers for FFT calculation
	*/
	
	float2 delay_a[16][11][MAX_DELAY]; // Gets delayed LOG2N
	float2 delay_b[16][11][MAX_DELAY * 2]; // Gets delayed LOG2N*2
	float2 delay_b_n[16][11][MAX_DELAY]; // Gets delayed LOG"N
	unsigned short delay_index[16][11][MAX_DELAY]; // Delay index by LOGN!
	float2 A0[2048];
	float2 A1[2048];
	float2 A2[2048];
	float2 A3[2048];
	float2 A4[2048];
	float2 A5[2048];
	float2 A6[2048];
	float2 A7[2048];
	float2 A8[2048];
	float2 A9[2048];
	float2 A10[2048];
	float2 A11[2048];
	float2 A12[2048];
	float2 A13[2048];
	float2 A14[2048];
	float2 A15[2048];

	float2 B0[2048];
	float2 B1[2048];
	float2 B2[2048];
	float2 B3[2048];
	float2 B4[2048];
	float2 B5[2048];
	float2 B6[2048];
	float2 B7[2048];
	float2 B8[2048];
	float2 B9[2048];
	float2 B10[2048];
	float2 B11[2048];
	float2 B12[2048];
	float2 B13[2048];
	float2 B14[2048];
	float2 B15[2048];

	float2 QA0[1024];
	float2 QA1[1024];
	float2 QA2[1024];
	float2 QA3[1024];
	float2 QA4[1024];
	float2 QA5[1024];
	float2 QA6[1024];
	float2 QA7[1024];
	float2 QA8[1024];
	float2 QA9[1024];
	float2 QA10[1024];
	float2 QA11[1024];
	float2 QA12[1024];
	float2 QA13[1024];
	float2 QA14[1024];
	float2 QA15[1024];

	float2 QB0[1024];
	float2 QB1[1024];
	float2 QB2[1024];
	float2 QB3[1024];
	float2 QB4[1024];
	float2 QB5[1024];
	float2 QB6[1024];
	float2 QB7[1024];
	float2 QB8[1024];
	float2 QB9[1024];
	float2 QB10[1024];
	float2 QB11[1024];
	float2 QB12[1024];
	float2 QB13[1024];
	float2 QB14[1024];
	float2 QB15[1024];

#define loop_size ((SIZE * (SIZE / (16 * 2))))
#define burst_buffer_overhead (4*SIZE/2)

	// HBM size mask to used on total loop index

#define BUFFER_OFFSET 0x8000



	float4 SlidingWindowInA[8][32];
	/*
		2D FFT performs two passes over the same code. Note it is assumed that HBMs are in pairs, 0,1 : 2,3 : 4,5 etc...
		Therefore, each input HBM is matched to another output HBM to create a double buffer configuration of shared
		data. Data is not shared between other pairs, hence sliding window transpose.

		The input and output offset of each HBM memory must be toggled between passess.

		The size of the offset is (SIZE*SIZE/2)/NO_HBMS. The division by 2 is because data is store in pairs, in order to
		keep the FFT pipelines busy 100% of the time.

		The two passes cannot unfortunately be combined into a single loop as the transpose of the data and the latency of the pipeline
		will cause the data to be read by the second pass before it is ready to be processed.

	*/
	

		unsigned short x, y;
		x = 0; y = 0;
		// Ignore any burst buffer dependencies using #pragma ivdep
#pragma ivdep
		for (int l = 0; l < (loop_size)+burst_buffer_overhead + 8 + FFT_1D_PIPE_LINE_DELAY;l++)
		{
			float4 fin[16];
			// Which buffer?
			bool a_or_b = ((l - FFT_1D_PIPE_LINE_DELAY) & 0x800) ? true : false;

			if (l < (loop_size + FFT_1D_PIPE_LINE_DELAY))
			{
				unsigned int HBM_offset = ((y >> 4) * (SIZE / 2));
				unsigned int HBM_address = ((x >> 1) + HBM_offset);
				HBM_address += pass == 0 ? 0 : BUFFER_OFFSET;

#ifndef INTEL_CL
				// Only of x86 going out of bounds is not an issue for FPGA
				HBM_address = HBM_address > (SIZE * SIZE / (16)) ? 0 : HBM_address;
#endif
				fin[0] = FFT_1d_1024_pipeline(in0[HBM_address], delay_a[0], delay_b[0], delay_b_n[0], delay_index[0], A0, B0, QA0, QB0, l);
				fin[1] = FFT_1d_1024_pipeline(in1[HBM_address], delay_a[1], delay_b[1], delay_b_n[1], delay_index[1], A1, B1, QA1, QB1, l);
				fin[2] = FFT_1d_1024_pipeline(in2[HBM_address], delay_a[2], delay_b[2], delay_b_n[2], delay_index[2], A2, B2, QA2, QB2, l);
				fin[3] = FFT_1d_1024_pipeline(in3[HBM_address], delay_a[3], delay_b[3], delay_b_n[3], delay_index[3], A3, B3, QA3, QB3, l);
				fin[4] = FFT_1d_1024_pipeline(in4[HBM_address], delay_a[4], delay_b[4], delay_b_n[4], delay_index[4], A4, B4, QA4, QB4, l);
				fin[5] = FFT_1d_1024_pipeline(in5[HBM_address], delay_a[5], delay_b[5], delay_b_n[5], delay_index[5], A5, B5, QA5, QB5, l);
				fin[6] = FFT_1d_1024_pipeline(in6[HBM_address], delay_a[6], delay_b[6], delay_b_n[6], delay_index[6], A6, B6, QA6, QB6, l);
				fin[7] = FFT_1d_1024_pipeline(in7[HBM_address], delay_a[7], delay_b[7], delay_b_n[7], delay_index[7], A7, B7, QA7, QB7, l);
				fin[8] = FFT_1d_1024_pipeline(in8[HBM_address], delay_a[8], delay_b[8], delay_b_n[8], delay_index[8], A8, B8, QA8, QB8, l);
				fin[9] = FFT_1d_1024_pipeline(in9[HBM_address], delay_a[9], delay_b[9], delay_b_n[9], delay_index[9], A9, B9, QA9, QB9, l);
				fin[10] = FFT_1d_1024_pipeline(in10[HBM_address], delay_a[10], delay_b[10], delay_b_n[10], delay_index[10], A10, B10, QA10, QB10, l);
				fin[11] = FFT_1d_1024_pipeline(in11[HBM_address], delay_a[11], delay_b[11], delay_b_n[11], delay_index[11], A11, B11, QA11, QB11, l);
				fin[12] = FFT_1d_1024_pipeline(in12[HBM_address], delay_a[12], delay_b[12], delay_b_n[12], delay_index[12], A12, B12, QA12, QB12, l);
				fin[13] = FFT_1d_1024_pipeline(in13[HBM_address], delay_a[13], delay_b[13], delay_b_n[13], delay_index[13], A13, B13, QA13, QB13, l);
				fin[14] = FFT_1d_1024_pipeline(in14[HBM_address], delay_a[14], delay_b[14], delay_b_n[14], delay_index[14], A14, B14, QA14, QB14, l);
				fin[15] = FFT_1d_1024_pipeline(in15[HBM_address], delay_a[15], delay_b[15], delay_b_n[15], delay_index[15], A15, B15, QA15, QB15, l);
			}

			unsigned short burst_buffer_in_address;
			burst_buffer_in_address = (l - FFT_1D_PIPE_LINE_DELAY) & 0x7ff;// ((x >> 1) + ((y / 16) * (SIZE / 2))) & 0x7ff;

			unsigned short l_x, l_y;
			unsigned int index = (l - (FFT_1D_PIPE_LINE_DELAY)) & 0x7ff;

			// Index is designed to read buffer rows column first
			// 8 words are read in x to create 16x16 transpose window
			// X is therfore bits (2 to 0) and bits * downto 5.
			// Y is bits (4 to 3)
			// For 16 word bursts 4 blocks must now be read from burst buffer to produce 32 float4's or 16 HBM words
			l_x = (index & 0x7) + ((index >> 5) * 8);
			l_y = (index >> 3) & 0x3;

			unsigned int buffer_index = l_x + (l_y * SIZE / 2);
			float4 bout[16];
			if ((l - FFT_1D_PIPE_LINE_DELAY) >= 0)
			{
				if (a_or_b)
				{
					bout[0] = BurstBufferA0[buffer_index];
					bout[1] = BurstBufferA1[buffer_index];
					bout[2] = BurstBufferA2[buffer_index];
					bout[3] = BurstBufferA3[buffer_index];
					bout[4] = BurstBufferA4[buffer_index];
					bout[5] = BurstBufferA5[buffer_index];
					bout[6] = BurstBufferA6[buffer_index];
					bout[7] = BurstBufferA7[buffer_index];
					bout[8] = BurstBufferA8[buffer_index];
					bout[9] = BurstBufferA9[buffer_index];
					bout[10] = BurstBufferA10[buffer_index];
					bout[11] = BurstBufferA11[buffer_index];
					bout[12] = BurstBufferA12[buffer_index];
					bout[13] = BurstBufferA13[buffer_index];
					bout[14] = BurstBufferA14[buffer_index];
					bout[15] = BurstBufferA15[buffer_index];
					BurstBufferB0[burst_buffer_in_address] = fin[0];
					BurstBufferB1[burst_buffer_in_address] = fin[1];
					BurstBufferB2[burst_buffer_in_address] = fin[2];
					BurstBufferB3[burst_buffer_in_address] = fin[3];
					BurstBufferB4[burst_buffer_in_address] = fin[4];
					BurstBufferB5[burst_buffer_in_address] = fin[5];
					BurstBufferB6[burst_buffer_in_address] = fin[6];
					BurstBufferB7[burst_buffer_in_address] = fin[7];
					BurstBufferB8[burst_buffer_in_address] = fin[8];
					BurstBufferB9[burst_buffer_in_address] = fin[9];
					BurstBufferB10[burst_buffer_in_address] = fin[10];
					BurstBufferB11[burst_buffer_in_address] = fin[11];
					BurstBufferB12[burst_buffer_in_address] = fin[12];
					BurstBufferB13[burst_buffer_in_address] = fin[13];
					BurstBufferB14[burst_buffer_in_address] = fin[14];
					BurstBufferB15[burst_buffer_in_address] = fin[15];
				}
				else
				{
					bout[0] = BurstBufferB0[buffer_index];
					bout[1] = BurstBufferB1[buffer_index];
					bout[2] = BurstBufferB2[buffer_index];
					bout[3] = BurstBufferB3[buffer_index];
					bout[4] = BurstBufferB4[buffer_index];
					bout[5] = BurstBufferB5[buffer_index];
					bout[6] = BurstBufferB6[buffer_index];
					bout[7] = BurstBufferB7[buffer_index];
					bout[8] = BurstBufferB8[buffer_index];
					bout[9] = BurstBufferB9[buffer_index];
					bout[10] = BurstBufferB10[buffer_index];
					bout[11] = BurstBufferB11[buffer_index];
					bout[12] = BurstBufferB12[buffer_index];
					bout[13] = BurstBufferB13[buffer_index];
					bout[14] = BurstBufferB14[buffer_index];
					bout[15] = BurstBufferB15[buffer_index];
					BurstBufferA0[burst_buffer_in_address] = fin[0];
					BurstBufferA1[burst_buffer_in_address] = fin[1];
					BurstBufferA2[burst_buffer_in_address] = fin[2];
					BurstBufferA3[burst_buffer_in_address] = fin[3];
					BurstBufferA4[burst_buffer_in_address] = fin[4];
					BurstBufferA5[burst_buffer_in_address] = fin[5];
					BurstBufferA6[burst_buffer_in_address] = fin[6];
					BurstBufferA7[burst_buffer_in_address] = fin[7];
					BurstBufferA8[burst_buffer_in_address] = fin[8];
					BurstBufferA9[burst_buffer_in_address] = fin[9];
					BurstBufferA10[burst_buffer_in_address] = fin[10];
					BurstBufferA11[burst_buffer_in_address] = fin[11];
					BurstBufferA12[burst_buffer_in_address] = fin[12];
					BurstBufferA13[burst_buffer_in_address] = fin[13];
					BurstBufferA14[burst_buffer_in_address] = fin[14];
					BurstBufferA15[burst_buffer_in_address] = fin[15];
				}
			}


			unsigned short buff_choice = (l - (burst_buffer_overhead + FFT_1D_PIPE_LINE_DELAY)) & 0x7;
#pragma unroll
			for (int block = 0; block < 8; block += 1)
			{
#pragma unroll
				for (int p = 0; p < 16; p += 2)
				{
					SlidingWindowInA[block][p] = SlidingWindowInA[block][p + 2];
					SlidingWindowInA[block][p + 1] = SlidingWindowInA[block][p + 3];
				}
#pragma unroll
				for (int p = 16; p < 32; p += 2)
				{
					short update = buff_choice == (block);
					SlidingWindowInA[block][p + 0] = update ? bout[(p - 16)] : SlidingWindowInA[block][p + 2];
					SlidingWindowInA[block][p + 1] = update ? bout[(p - 16) + 1] : SlidingWindowInA[block][p + 3];
				}
			}

			// Split in to two rows to write to HBMs in parallel
			float4 output_row[16];
			if (l >= (burst_buffer_overhead + FFT_1D_PIPE_LINE_DELAY + 8))
			{
#pragma unroll
				for (int p = 0; p < 8; p++)
				{
					float4 a, b;
					a = SlidingWindowInA[p][(p << 1)];
					b = SlidingWindowInA[p][(p << 1) + 1];
					output_row[(p << 1)][0] = a[0];
					output_row[(p << 1)][1] = a[1];
					output_row[(p << 1)][2] = b[0];
					output_row[(p << 1)][3] = b[1];
					output_row[(p << 1) + 1][0] = a[2];
					output_row[(p << 1) + 1][1] = a[3];
					output_row[(p << 1) + 1][2] = b[2];
					output_row[(p << 1) + 1][3] = b[3];
				}
			}

			if (l >= (burst_buffer_overhead + FFT_1D_PIPE_LINE_DELAY + 8))
			{
				// Address out is now in bursts for 32
				short l_x, l_y;
				// Index is designed to read buffer rows column first
				// 8 words are read in x to create 16x16 transpose window
				// X is therfore bits (2 to 0) and bits * downto 5.
				// Y is bits (4 to 3)
				int index = l - (burst_buffer_overhead + FFT_1D_PIPE_LINE_DELAY + 8);
				l_x = (index & 0x1f) + ((index >> (6 + 5)) * 32);
				l_y = (index >> 5) & (0x3ff >> 4);
				int addr_out = l_x + (l_y * SIZE / 2);
				addr_out += pass == 0 ? BUFFER_OFFSET : 0;

				out0[addr_out] = output_row[0];
				out1[addr_out] = output_row[1];
				out2[addr_out] = output_row[2];
				out3[addr_out] = output_row[3];
				out4[addr_out] = output_row[4];
				out5[addr_out] = output_row[5];
				out6[addr_out] = output_row[6];
				out7[addr_out] = output_row[7];
				out8[addr_out] = output_row[8];
				out9[addr_out] = output_row[9];
				out10[addr_out] = output_row[10];
				out11[addr_out] = output_row[11];
				out12[addr_out] = output_row[12];
				out13[addr_out] = output_row[13];
				out14[addr_out] = output_row[14];
				out15[addr_out] = output_row[15];
			}

			x = x != (SIZE - 2) ? x + 2 : 0; // Two pixels at a time.
			y = x == 0 ? y + 16 : y; // 16 rows at a time, 1 por HBM
		}
#ifndef INTEL_CL
		delete BurstBufferA0;
		delete BurstBufferA1;
		delete BurstBufferA2;
		delete BurstBufferA3;
		delete BurstBufferA4;
		delete BurstBufferA5;
		delete BurstBufferA6;
		delete BurstBufferA7;
		delete BurstBufferA8;
		delete BurstBufferA9;
		delete BurstBufferA10;
		delete BurstBufferA11;
		delete BurstBufferA12;
		delete BurstBufferA13;
		delete BurstBufferA14;
		delete BurstBufferA15;


		delete BurstBufferB0;
		delete BurstBufferB1;
		delete BurstBufferB2;
		delete BurstBufferB3;
		delete BurstBufferB4;
		delete BurstBufferB5;
		delete BurstBufferB6;
		delete BurstBufferB7;
		delete BurstBufferB8;
		delete BurstBufferB9;
		delete BurstBufferB10;
		delete BurstBufferB11;
		delete BurstBufferB12;
		delete BurstBufferB13;
		delete BurstBufferB14;
		delete BurstBufferB15;
#endif
	}


}
