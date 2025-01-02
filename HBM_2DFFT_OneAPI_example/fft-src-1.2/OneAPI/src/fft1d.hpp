/**
Copyright (c) 2020, 2021 by BittWare, Inc., A Molex Company 

This source code is provided to you (the Licensee) under license by BittWare, a Molex Company.

To view or use this source code, the Licensee must accept a Software License Agreement (viewable at developer.bittware.com),
which is commonly provided as a click-through license agreement.  The terms of the Software License Agreement govern all use
and distribution of this file unless an alternative superseding license has been executed with BittWare.  This source code
and its derivatives may not be distributed to third parties in source code form.  Software including or derived from this
source code, including derivative works thereof created by Licensee, may be distributed to third parties with BITTWARE
hardware only and in executable form only.

The click-through license is available here: https://developer.bittware.com/software_license.txt
**/


/*
  FFT to use delay paths to group pairs accordingly and avoid M20K access problems
*/
#ifndef FFT1D_H
#define FFT1D_H

#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

using namespace cl::sycl;

#define float2 vec<float, 2>

template <int unique_identifer,int __SIZE__,int __LOGN__> 
class BITTWARE_FFT
{

	private:
		int __MAX_FFT_SIZE__;
		int __MAX_FFT_SIZE_LOG_N__;

		[[intelfpga::private_copies(__LOGN__+1)]] const float weights_32[16][2];
		// This includes twiddle addresses.
		const short address_reorder_LUT_32[32];
		const unsigned short BIT_MASK[__LOGN__ + 1];
		// Mask for square sizes
		const unsigned short BIT_MASK_DQ[__LOGN__ + 1];

	public:
		BITTWARE_FFT():
			__MAX_FFT_SIZE__(32),
			__MAX_FFT_SIZE_LOG_N__(5),
			weights_32{{ 1.000000,0.000000 }, {0.980785,-0.195090} ,{0.923880,-0.382683}, {0.831470,-0.555570},
					   {0.707107,-0.707107}, {0.555570,-0.831470} ,{0.382683,-0.923880}, {0.195090,-0.980785},
					   {0.000000,-1.000000}, {-0.195090,-0.980785}, {-0.382683,-0.923880}, {-0.555570,-0.831470},
					   {-0.707107,-0.707107} ,{-0.831470,-0.555570} ,{-0.923880,-0.382683}, {-0.980785,-0.195090}
					 },
			// This includes twiddle addresses.
			address_reorder_LUT_32{	0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31 },
			BIT_MASK{ 0x1,0x3,0x7,0xf,0x1f,0x3f },
			// Mask for square sizes
			BIT_MASK_DQ{ 0x1,0x3,0xf,0x3f,0xff,0x3ff }
		{

		}

    ~BITTWARE_FFT(){}

	static constexpr int Latency = (__SIZE__ / 2) +		 // Reorder delay
		((__SIZE__ / 2) - 1) + // FFT pipeline delay
		(__SIZE__ / 2);		 // Final shuffle

    void Radix2ButterFly(float2 a, float2 b, float2* a_out, float2* b_out, float2 w)
    {
        float real, imag;
        float real2, imag2;
        real = a[0];
        imag = a[1];
        real2 = b[0];
        imag2 = b[1];
        float real_w, imag_w;
        real_w = w[0];
        imag_w = w[1];
        float real_r, imag_r;
        real_r = real2 * real_w - imag2 * imag_w;
        imag_r = real2 * imag_w + imag2 * real_w;
        float2 outa,outb;
        outa[0] = real + real_r;
        outa[1] = imag + imag_r;
        outb[0] = real - real_r;
        outb[1] = imag - imag_r;
        (*a_out) = outa;
        (*b_out) = outb;
    }

    /*
    Delay inputs long enough that pairs can align.
    E.g. Stage 1 will wait for 2 words,
    Stage 2 will wait for 4 words,
    Stage 4 will wait for 8 words, etc...
    */

    // Need to delay 2 clocks to have pairs in place	
	inline void Stage(float2 a, float2 b, float2* a_out, float2* b_out, short index,const short Delay, const short A, const short n,
        float2 delay_a[__SIZE__/2],
        float2 delay_b[__SIZE__],
        float2 delay_b_n[__SIZE__ / 2],
        short delay_index[__SIZE__ / 2],
		bool dir
    )
    {
        float2 new_a, new_b;
        // select appropriate delay path depending where we are in the current index
        // Swap input halfway through
		short mask = index == 0 ? 1 : Delay;
        short MASK = mask;// 0x1;
        short current_index = delay_index[0];

		index = index + (Delay - A + 1);
		if (Delay == 0)
        {
            current_index = index;
            new_a = a;
            new_b = b;
        }
        else
        {
            if ((current_index & MASK))
            {
                new_a = delay_b[0];
                new_b = delay_b_n[0];
            }
            else
            {
                new_a = delay_a[0];
                new_b = a;
            }
        }


        // Now calculate butterfly

        float2 w;
		if (__SIZE__ == 32)
		{
			w[0] = dir?-weights_32[(current_index * n) % (__SIZE__ / 2)][0] :weights_32[(current_index * n) % (__SIZE__ / 2)][0];
			w[1] = dir?-weights_32[(current_index * n) % (__SIZE__ / 2)][1] :weights_32[(current_index * n) % (__SIZE__ / 2)][1];
		}

        if (index >= Delay)
        {
			Radix2ButterFly(new_a, new_b, a_out, b_out, w);
        }
   
        // Create sliding window
        if (Delay != 0)
        {
            #pragma unroll
            for (int p = 0; p < (Delay - 1); p++)
            {
                delay_a[p] = delay_a[p + 1];
                delay_b_n[p] = delay_b_n[p + 1];
                delay_index[p] = delay_index[p + 1];
            }
            delay_a[Delay - 1] = a;
            delay_b_n[Delay - 1] = b;
            delay_index[Delay - 1] = index;


            #pragma unroll
            for (int p = 0; p < ((Delay << 1) - 1); p++)
            {
                delay_b[p] = delay_b[p + 1];
            }
            delay_b[(Delay << 1) - 1] = b;
        }
    }


	[[intelfpga::register]] float2 delay_a[__LOGN__][__SIZE__ / 2]; // Gets delayed LOG2N
	[[intelfpga::register]] float2 delay_b[__LOGN__][__SIZE__]; // Gets delayed LOG2N*2
	[[intelfpga::register]] float2 delay_b_n[__LOGN__][__SIZE__ / 2]; // Gets delayed LOG"N
	[[intelfpga::register]] short delay_index[__LOGN__][__SIZE__ / 2]; // Delay index by LOGN!
	
	inline void fft_butterfly_pipeline(short i, float2 in_a, float2 in_b, float2* out_a, float2* out_b, bool dir)
	{
        [[intelfpga::register]] float2 pairs[__LOGN__ + 1][2];
        pairs[0][0] = in_a;
		pairs[0][1] = in_b;
		#pragma unroll
		for (int s = 0; s < __LOGN__;s++)
		{
			Stage(pairs[s][0], pairs[s][1], &pairs[s + 1][0], &pairs[s + 1][1], i, s==0?0:1 << (s - 1), 1 << (s), (__SIZE__ >> (s + 1)), delay_a[s], delay_b[s], delay_b_n[s], delay_index[s], dir);
		}

		*out_a = pairs[__LOGN__][0];
		*out_b = pairs[__LOGN__][1];
	}

	float2 complex0[__SIZE__ * 2];
	inline void reorder(short i, float4 c, float2* c0_out, float2* c1_out)
	{
		short addr_in = i;
		short addr_out = i+ __SIZE__/2;
		unsigned short mask = BIT_MASK[__LOGN__ - 1];
		unsigned short mask_inv = ~mask;
		unsigned short mask2 = BIT_MASK[__LOGN__ ];


		float2 c0, c1;
		c0[0] = c[0];
		c0[1] = c[1];
		c1[0] = c[2];
		c1[1] = c[3];
		

		short a0_in, a1_in;
		a0_in = (addr_in << 1) & mask2;
		a1_in = ((addr_in << 1) + 1) & mask2;
		complex0[a0_in] = c0;
		complex0[a1_in] = c1;


		// c0 and c1 are always from opposing halves of the input data.
		short a0_out, a1_out;
		a0_out = (((addr_out << 1) & mask_inv) + address_reorder_LUT_32[(addr_out << 1) & mask]) & mask2;
		a1_out = (((addr_out << 1) & mask_inv) + address_reorder_LUT_32[((addr_out << 1) + 1) & mask]) & mask2;
		if (__SIZE__ == 32)
		{			
			*c0_out = complex0[a0_out];
			*c1_out = complex0[a1_out];
		}
	}

	// Delay pairs to be inorder. Only need to wait until half way through.
	float2 delay0[__SIZE__];
	float2 delay1[__SIZE__];
	inline void final_shuffle(short index,float2 c0, float2 c1, float2* c_out0, float2* c_out1)
	{
		// Create shift register sliding window
		unsigned short mask = BIT_MASK[__LOGN__-1];
		unsigned short mask_inv = ~mask;

		delay0[index & mask] = c0; 
		delay1[index & mask] = c1;

		short out_index = (index&(mask>>1));
		short bank_offset = (((index &(~BIT_MASK[__LOGN__ - 2]))&mask) != 0) ? 0 : (__SIZE__ >> 1);
		if ((out_index&BIT_MASK[__LOGN__ - 2]) < 8)
		{
			*c_out0 = delay0[bank_offset + (out_index << 1) & mask];
			*c_out1 = delay0[bank_offset + (1 + (out_index << 1)) & mask];
		}
		else // Use second half
		{
			*c_out0 = delay1[bank_offset + ((out_index-8) << 1) & mask];
			*c_out1 = delay1[bank_offset + (1 + ((out_index-8) << 1)) & mask];
		}
	}

	/*
		Simple 2D transpose for input pairs.
	*/
	float4 double_buffer[__SIZE__ * __SIZE__];
	unsigned short x = 0;
	unsigned short y = 0;
	inline void TransposePipeline(short i, float4 in, float4* out)
	{
		int mask = BIT_MASK_DQ[__LOGN__];
		int input_addr = i & mask;
		int output_addr_offset = (i & ~(mask>>1)) ? 0 : (__SIZE__ * __SIZE__ >> 1);
		
		float4 a, b;
		int addr_a;
		int addr_b;
		addr_a = ((x >> 1) * __SIZE__) + (y >> 1);
		addr_b = (((x >> 1) + 1) * __SIZE__) + (y >> 1);
		a = double_buffer[output_addr_offset + addr_a];
		b = double_buffer[output_addr_offset + addr_b];
		float4 word;
		if (y & 0x1)
		{
			word[0] = a[2];
			word[1] = a[3];
			word[2] = b[2];
			word[3] = b[3];
		}
		else
		{
			word[0] = a[0];
			word[1] = a[1];
			word[2] = b[0];
			word[3] = b[1];
		}

		*out = word;
		double_buffer[input_addr] = in;
		x = x != ((__SIZE__ >> 1) - 1) ? x + 1:0;
		y = (x == 0) ? (y != (__SIZE__ - 1)) ? y + 1 : 0 : y;
	}
	/*
		FFT pipeline
	*/
	inline void fft_pipline(short i,float4 in, float4* out, bool dir)
	{
		unsigned short mask = BIT_MASK[__LOGN__ - 1];        
		unsigned short mask_inv = ~mask;

		unsigned short mask_in = BIT_MASK[__LOGN__ - 2];
		unsigned short mask_in_inv = ~mask_in;


		[[intelfpga::register]] float2 pairs[2][2];
		reorder(i,in, &pairs[0][0], &pairs[0][1]);
		if (i >= (__SIZE__ / 2))
		{
			
			fft_butterfly_pipeline((i - (__SIZE__ / 2)), pairs[0][0], pairs[0][1], &pairs[1][0], &pairs[1][1], dir);
		}
		float2 c0, c1;
		if (i >= ((__SIZE__ / 2) +
			((__SIZE__ / 2) - 1)))
		{
			
			short shuf_i = i - ((__SIZE__ / 2) + ((__SIZE__ / 2) - 1));
			final_shuffle(shuf_i,pairs[1][0], pairs[1][1], &c0, &c1);
		}
		float4 final;
		final[0] = c0[0];
		final[1] = c0[1];
		final[2] = c1[0];
		final[3] = c1[1];
		*out = final;
	}


	/*
		Performs a 1D FFT.
	*/
	void fft(float4* in, float4* out,bool dir)
	{
		unsigned short mask = BIT_MASK[__LOGN__ - 1];
		unsigned short mask_inv = ~mask;

		unsigned short mask_in = BIT_MASK[__LOGN__ - 2];
		unsigned short mask_in_inv = ~mask_in;

		for (int i = 0; i < ((__SIZE__ / 2)  + Latency); i++)
		{
			float4 out_temp;
			float4 input;
			if (i < (__SIZE__ / 2))
				input = in[i];
			fft_pipline(i,input, &out_temp,dir);
			if (i >= Latency)
				out[i - Latency] = out_temp;
		}
	}



	/*
		Performs a 1D FFT accross the rows of a symetric plane.
	*/
    void fft_plane(float4* in, float4* out)
    {
        [[intelfpga::register]] float2 pairs[2][2];
		unsigned short mask = BIT_MASK[__LOGN__ - 1];
		unsigned short mask_inv = ~mask;

		unsigned short mask_in = BIT_MASK[__LOGN__ - 2];
		unsigned short mask_in_inv = ~mask_in;

		[[intelfpga::ivdep(__SIZE__)]] 
       		for (int i = 0; i < (((__SIZE__ / 2) * __SIZE__) + Latency ); i++)
		{
			float4 out_temp;
			float4 input;
			if (i < ((__SIZE__ / 2) * __SIZE__))
			{
				input = in[i];				
			}
			else
			{
				input[0] = 0;
				input[1] = 0;
				input[2] = 0;
				input[3] = 0;
			}
			fft_pipline(i, input, &out_temp, 0);
         
			if (i >= Latency)
			{
				out[i - Latency] = out_temp;
			}
		}

	}



};


#endif
