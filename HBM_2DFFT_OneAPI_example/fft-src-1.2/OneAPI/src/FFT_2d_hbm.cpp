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

#include <CL/sycl.hpp>
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

#include "FFT_2d_hbm.h"

// Directly include 1D FFT code.
#include "FFT_1d_pipeline.cpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

/*
This code peforms a 2D FFT using 16 parallel 1D FFTs.
Each FFT is fed by a single HBM using one pseudo port whilst the output is written to the other pseudo port.
However, the HBM ports are transposed by the algorithm ready for the next pass.

See the standalone transpose code "FFT_transpose_hbm.cpp" for further explanation of how the transpose works
*/


// problem input size
constexpr size_t kInSize = 1024*1024;
constexpr double kInputMB = (kInSize * sizeof(int)) / (1024 * 1024);

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}


#define INTEL_CL
#define ONE_API_KERNEL //marks that the kernel code is using Intel's oneAPI framework.

// Convenience data access definitions 这个其实后面没用上
//Instead of repeatedly typing access::mode::read,you can use the shorter dp_read or dp_write for readability.
constexpr access::mode dp_read = access::mode::read; //dp_read represents the read access mode for buffers in SYCL.
constexpr access::mode dp_write = access::mode::write;

#define BUFFER_OFFSET 0x8000 //defines BUFFER_OFFSET as a hexadecimal value (0x8000 = 32,768 in decimal).

// output message for runtime exceptions
#define EXCEPTION_MSG \
  "    If you are targeting an FPGA hardware, please ensure that an FPGA board is plugged to the system, \n\
        set up correctly and compile with -DFPGA  \n\
    If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR.\n"

//The cl::sycl::INTEL::lsu refers to Intel-specific FPGA features.
//The burst_coalesce<true> parameter optimizes memory transactions by combining (coalescing) smaller memory accesses into larger bursts.
using BurstCoalescedLSU = cl::sycl::INTEL::lsu<cl::sycl::INTEL::burst_coalesce<true>>;

#define loop_HBM_FFT_SIZE ((HBM_FFT_SIZE * (HBM_FFT_SIZE / (16 * 2))))
#define burst_buffer_overhead (4*HBM_FFT_SIZE/2)
//allocate extra space in memory buffers to accommodate burst-mode optimizations or avoid memory overflows.



static void ReportTime(const std::string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();

  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds\n";
}


/**
 * Example fast large 2D FFT using HBMs to provide very high throughput.
 */
//utilizing multiple HBM banks for parallel data processing. 
void FFT2D_in_DPCPP_HBMs(float *HBM[16])
{
	// create device selector for the device of your interest
	#ifdef FPGA_EMULATOR
  	// DPC++ extension: FPGA emulator selector on systems without FPGA card
	  INTEL::fpga_emulator_selector dselector;
	#else
	  // DPC++ extension: FPGA selector on systems with FPGA card
	  INTEL::fpga_selector dselector;
	#endif

	event submit_event; 

	//The try block in C++ is part of exception handling. 
	//It is used to catch and manage errors that might occur during the execution of code within the try block. 
	//If an exception is thrown, the catch block will handle it, preventing the program from crashing.
	try { //try block ensures safe execution of Queue creation and Buffer creation
		
		// the kernels and encapsulates all the states needed for execution
	  	auto prop_list = property_list{property::queue::enable_profiling()};
		// create the devices queue with the selector above and the exception
		// handler to catch async runtime errors the device queue is used to enqueue
		queue q(dselector, dpc_common::exception_handler, prop_list);

		// Use Verbose SYCL 1.2. syntax for output bugffer
		//Defines a 1D range representing the number of items (HBM_FFT_SIZE² / 16)
		range<1> num_items{ HBM_FFT_SIZE * HBM_FFT_SIZE / (16)};
		// Setup device buffers, which align to HBM[]
		buffer<float4, 1> buffer_in0((float4*)HBM[0], num_items);
		buffer<float4, 1> buffer_in1((float4*)HBM[1], num_items);
		buffer<float4, 1> buffer_in2((float4*)HBM[2], num_items);
		buffer<float4, 1> buffer_in3((float4*)HBM[3], num_items);
		buffer<float4, 1> buffer_in4((float4*)HBM[4], num_items);
		buffer<float4, 1> buffer_in5((float4*)HBM[5], num_items);
		buffer<float4, 1> buffer_in6((float4*)HBM[6], num_items);
		buffer<float4, 1> buffer_in7((float4*)HBM[7], num_items);
		buffer<float4, 1> buffer_in8((float4*)HBM[8], num_items);
		buffer<float4, 1> buffer_in9((float4*)HBM[9], num_items);
		buffer<float4, 1> buffer_in10((float4*)HBM[10], num_items);
		buffer<float4, 1> buffer_in11((float4*)HBM[11], num_items);
		buffer<float4, 1> buffer_in12((float4*)HBM[12], num_items);
		buffer<float4, 1> buffer_in13((float4*)HBM[13], num_items);
		buffer<float4, 1> buffer_in14((float4*)HBM[14], num_items);
		buffer<float4, 1> buffer_in15((float4*)HBM[15], num_items);

		//sets up a SYCL kernel submission, preparing memory access patterns for input and output buffers	
		submit_event = q.submit([&](handler& h)
		{
			// Create Sycl Accessors with Buffer Location
			sycl::ONEAPI::accessor_property_list PL0(sycl::INTEL::buffer_location<0>);
			sycl::ONEAPI::accessor_property_list PL1(sycl::INTEL::buffer_location<1>);
			sycl::ONEAPI::accessor_property_list PL2(sycl::INTEL::buffer_location<2>);
			sycl::ONEAPI::accessor_property_list PL3(sycl::INTEL::buffer_location<3>);
			sycl::ONEAPI::accessor_property_list PL4(sycl::INTEL::buffer_location<4>);
			sycl::ONEAPI::accessor_property_list PL5(sycl::INTEL::buffer_location<5>);
			sycl::ONEAPI::accessor_property_list PL6(sycl::INTEL::buffer_location<6>);
			sycl::ONEAPI::accessor_property_list PL7(sycl::INTEL::buffer_location<7>);
			sycl::ONEAPI::accessor_property_list PL8(sycl::INTEL::buffer_location<8>);
			sycl::ONEAPI::accessor_property_list PL9(sycl::INTEL::buffer_location<9>);
			sycl::ONEAPI::accessor_property_list PL10(sycl::INTEL::buffer_location<10>);
			sycl::ONEAPI::accessor_property_list PL11(sycl::INTEL::buffer_location<11>);
			sycl::ONEAPI::accessor_property_list PL12(sycl::INTEL::buffer_location<12>);
			sycl::ONEAPI::accessor_property_list PL13(sycl::INTEL::buffer_location<13>);
			sycl::ONEAPI::accessor_property_list PL14(sycl::INTEL::buffer_location<14>);
			sycl::ONEAPI::accessor_property_list PL15(sycl::INTEL::buffer_location<15>);

			//sycl::accessor is created for each input buffer. These allow the kernel 
			//to read data from the corresponding buffers with specific memory locations
			sycl::accessor in0_acc(buffer_in0,h,sycl::read_only,PL0);
			sycl::accessor in1_acc(buffer_in1,h,sycl::read_only,PL1);
			sycl::accessor in2_acc(buffer_in2,h,sycl::read_only,PL2);
			sycl::accessor in3_acc(buffer_in3,h,sycl::read_only,PL3);
			sycl::accessor in4_acc(buffer_in4,h,sycl::read_only,PL4);
			sycl::accessor in5_acc(buffer_in5,h,sycl::read_only,PL5);
			sycl::accessor in6_acc(buffer_in6,h,sycl::read_only,PL6);
			sycl::accessor in7_acc(buffer_in7,h,sycl::read_only,PL7);
			sycl::accessor in8_acc(buffer_in8,h,sycl::read_only,PL8);
			sycl::accessor in9_acc(buffer_in9,h,sycl::read_only,PL9);
			sycl::accessor in10_acc(buffer_in10,h,sycl::read_only,PL10);
			sycl::accessor in11_acc(buffer_in11,h,sycl::read_only,PL11);
			sycl::accessor in12_acc(buffer_in12,h,sycl::read_only,PL12);
			sycl::accessor in13_acc(buffer_in13,h,sycl::read_only,PL13);
			sycl::accessor in14_acc(buffer_in14,h,sycl::read_only,PL14);
			sycl::accessor in15_acc(buffer_in15,h,sycl::read_only,PL15);

			//created for each output buffer, allowing the kernel to write results back to the buffers
			sycl::accessor out0_acc(buffer_in0,h,sycl::write_only,PL0);
			sycl::accessor out1_acc(buffer_in1,h,sycl::write_only,PL1);
			sycl::accessor out2_acc(buffer_in2,h,sycl::write_only,PL2);
			sycl::accessor out3_acc(buffer_in3,h,sycl::write_only,PL3);
			sycl::accessor out4_acc(buffer_in4,h,sycl::write_only,PL4);
			sycl::accessor out5_acc(buffer_in5,h,sycl::write_only,PL5);
			sycl::accessor out6_acc(buffer_in6,h,sycl::write_only,PL6);
			sycl::accessor out7_acc(buffer_in7,h,sycl::write_only,PL7);
			sycl::accessor out8_acc(buffer_in8,h,sycl::write_only,PL8);
			sycl::accessor out9_acc(buffer_in9,h,sycl::write_only,PL9);
			sycl::accessor out10_acc(buffer_in10,h,sycl::write_only,PL10);
			sycl::accessor out11_acc(buffer_in11,h,sycl::write_only,PL11);
			sycl::accessor out12_acc(buffer_in12,h,sycl::write_only,PL12);
			sycl::accessor out13_acc(buffer_in13,h,sycl::write_only,PL13);
			sycl::accessor out14_acc(buffer_in14,h,sycl::write_only,PL14);
			sycl::accessor out15_acc(buffer_in15,h,sycl::write_only,PL15);

			h.single_task<class kernel_FFT_2d_hbm_stage_1>([=]()
			// Restrict alliasing of pointers.
			[[intel::kernel_args_restrict]]
			{			
				[[intel::max_concurrency(1)]]
				for (int loop = 0; loop < 2; loop++)
				{
					unsigned char pass = loop&0x1;
					// Double buffer
					float4 BurstBufferA0[4 * HBM_FFT_SIZE];
					float4 BurstBufferA1[4 * HBM_FFT_SIZE];
					float4 BurstBufferA2[4 * HBM_FFT_SIZE];
					float4 BurstBufferA3[4 * HBM_FFT_SIZE];
					float4 BurstBufferA4[4 * HBM_FFT_SIZE];
					float4 BurstBufferA5[4 * HBM_FFT_SIZE];
					float4 BurstBufferA6[4 * HBM_FFT_SIZE];
					float4 BurstBufferA7[4 * HBM_FFT_SIZE];
					float4 BurstBufferA8[4 * HBM_FFT_SIZE];
					float4 BurstBufferA9[4 * HBM_FFT_SIZE];
					float4 BurstBufferA10[4 * HBM_FFT_SIZE];
					float4 BurstBufferA11[4 * HBM_FFT_SIZE];
					float4 BurstBufferA12[4 * HBM_FFT_SIZE];
					float4 BurstBufferA13[4 * HBM_FFT_SIZE];
					float4 BurstBufferA14[4 * HBM_FFT_SIZE];
					float4 BurstBufferA15[4 * HBM_FFT_SIZE];

					/*
						Delay buffers for FFT calculation
						Expand dimensions 1 and 2.

						16 copies one for each parallel FFT.
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

					float4 SlidingWindowInA[8][32];
					/*
						2D FFT performs two passes over the same code. Note it is assumed that HBMs are in pairs, 0,1 : 2,3 : 4,5 etc...
						Therefore, each input HBM is matched to another output HBM to create a double buffer configuration of shared
						data. Data is not shared between other pairs, hence sliding window transpose.

						The input and output offset of each HBM memory must be toggled between passess.

						The HBM_FFT_SIZE of the offset is (HBM_FFT_SIZE*HBM_FFT_SIZE/2)/NO_HBMS. The division by 2 is because data is store in pairs, in order to
						keep the FFT pipelines busy 100% of the time.

						The two passes cannot unfortunately be combined into a single loop as the transpose of the data and the latency of the pipeline
						will cause the data to be read by the second pass before it is ready to be processed.

					*/

					unsigned short x, y;
					x = 0; y = 0;
					const unsigned int loopsize = (loop_HBM_FFT_SIZE)+burst_buffer_overhead + 8 + FFT_1D_PIPE_LINE_DELAY;
					// Ignore any burst buffer dependencies using #pragma ivdep
					[[intel::ivdep]]
					for (int l = 0; l < loopsize ;l++)
					{
						float4 fin[16];
						// Which buffer?
						bool a_or_b = ((l - FFT_1D_PIPE_LINE_DELAY) & 0x800) ? true : false;
						unsigned int HBM_offset = ((y >> 4) * (HBM_FFT_SIZE / 2));
						unsigned int HBM_address = ((x >> 1) + HBM_offset);

						HBM_address += pass == 0 ? 0 : BUFFER_OFFSET;

						// Run fully pipelined 1D FFTs in parallel
						if (l < (loop_HBM_FFT_SIZE + FFT_1D_PIPE_LINE_DELAY))
						{
							fin[0] = FFT_1d_1024_pipeline(in0_acc[HBM_address], delay_a[0], delay_b[0], delay_b_n[0], delay_index[0], A0, QA0, QB0, l);
							fin[1] = FFT_1d_1024_pipeline(in1_acc[HBM_address], delay_a[1], delay_b[1], delay_b_n[1], delay_index[1], A1, QA1, QB1, l);
							fin[2] = FFT_1d_1024_pipeline(in2_acc[HBM_address], delay_a[2], delay_b[2], delay_b_n[2], delay_index[2], A2, QA2, QB2, l);
							fin[3] = FFT_1d_1024_pipeline(in3_acc[HBM_address], delay_a[3], delay_b[3], delay_b_n[3], delay_index[3], A3, QA3, QB3, l);
							fin[4] = FFT_1d_1024_pipeline(in4_acc[HBM_address], delay_a[4], delay_b[4], delay_b_n[4], delay_index[4], A4, QA4, QB4, l);
							fin[5] = FFT_1d_1024_pipeline(in5_acc[HBM_address], delay_a[5], delay_b[5], delay_b_n[5], delay_index[5], A5, QA5, QB5, l);
							fin[6] = FFT_1d_1024_pipeline(in6_acc[HBM_address], delay_a[6], delay_b[6], delay_b_n[6], delay_index[6], A6, QA6, QB6, l);
							fin[7] = FFT_1d_1024_pipeline(in7_acc[HBM_address], delay_a[7], delay_b[7], delay_b_n[7], delay_index[7], A7, QA7, QB7, l);
							fin[8] = FFT_1d_1024_pipeline(in8_acc[HBM_address], delay_a[8], delay_b[8], delay_b_n[8], delay_index[8], A8, QA8, QB8, l);
							fin[9] = FFT_1d_1024_pipeline(in9_acc[HBM_address], delay_a[9], delay_b[9], delay_b_n[9], delay_index[9], A9, QA9, QB9, l);
							fin[10] = FFT_1d_1024_pipeline(in10_acc[HBM_address], delay_a[10], delay_b[10], delay_b_n[10], delay_index[10], A10, QA10, QB10, l);
							fin[11] = FFT_1d_1024_pipeline(in11_acc[HBM_address], delay_a[11], delay_b[11], delay_b_n[11], delay_index[11], A11, QA11, QB11, l);
							fin[12] = FFT_1d_1024_pipeline(in12_acc[HBM_address], delay_a[12], delay_b[12], delay_b_n[12], delay_index[12], A12, QA12, QB12, l);
							fin[13] = FFT_1d_1024_pipeline(in13_acc[HBM_address], delay_a[13], delay_b[13], delay_b_n[13], delay_index[13], A13, QA13, QB13, l);
							fin[14] = FFT_1d_1024_pipeline(in14_acc[HBM_address], delay_a[14], delay_b[14], delay_b_n[14], delay_index[14], A14, QA14, QB14, l);
							fin[15] = FFT_1d_1024_pipeline(in15_acc[HBM_address], delay_a[15], delay_b[15], delay_b_n[15], delay_index[15], A15, QA15, QB15, l);
						}

						#define BURST_BUFFER_MASK 0x7ff

						unsigned short burst_buffer_in_address;
						burst_buffer_in_address = (l - FFT_1D_PIPE_LINE_DELAY) & BURST_BUFFER_MASK;// ((x >> 1) + ((y / 16) * (HBM_FFT_SIZE / 2))) & 0x7ff;

						unsigned short l_x, l_y;
						unsigned int index = (l - (FFT_1D_PIPE_LINE_DELAY)) & BURST_BUFFER_MASK;

						// Index is designed to read buffer rows column first
						// 8 words are read in x to create 16x16 transpose window
						// X is therfore bits (2 to 0) and bits * downto 5.
						// Y is bits (4 to 3)
						// For 16 word bursts 4 blocks must now be read from burst buffer to produce 32 float4's or 16 HBM words
						l_x = (index & 0x7) + ((index >> 5) * 8);
						l_y = (index >> 3) & 0x3;

						unsigned int buffer_index = l_x + (l_y * HBM_FFT_SIZE / 2);
						float4 bout[16];
						unsigned int double_buffer_offset_in = a_or_b ? (4 * HBM_FFT_SIZE / 2) : 0;
						unsigned int double_buffer_offset_out = a_or_b ? 0 : (4 * HBM_FFT_SIZE / 2);

						buffer_index = (buffer_index&BURST_BUFFER_MASK) + double_buffer_offset_in;
						burst_buffer_in_address = (burst_buffer_in_address&BURST_BUFFER_MASK) + double_buffer_offset_out;

						if ((l - FFT_1D_PIPE_LINE_DELAY) >= 0)
						{
							/*
								Burst buffer for output to HBMs.
								Used to improve burst length problems introduced from transpose function.

								One buffer for each of the 16 HBM outputs. Each buffer can store 4 rows of input
							*/
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
								//short update = buff_choice == (block);
								if (buff_choice == block)
								{
									SlidingWindowInA[block][p + 0] = bout[(p - 16)];
									SlidingWindowInA[block][p + 1] = bout[(p - 16) + 1];
								}
								else
								{
									if (p != 30)
									{
										SlidingWindowInA[block][p + 0] = SlidingWindowInA[block][p + 2];
										SlidingWindowInA[block][p + 1] = SlidingWindowInA[block][p + 3];
									}
								}
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
							int addr_out = l_x + (l_y * HBM_FFT_SIZE / 2);
							addr_out += pass == 0 ? BUFFER_OFFSET : 0;
							
							out0_acc[addr_out] = output_row[0];
							out1_acc[addr_out] = output_row[1];
							out2_acc[addr_out] = output_row[2];
							out3_acc[addr_out] = output_row[3];
							out4_acc[addr_out] = output_row[4];
							out5_acc[addr_out] = output_row[5];
							out6_acc[addr_out] = output_row[6];
							out7_acc[addr_out] = output_row[7];
							out8_acc[addr_out] = output_row[8];
							out9_acc[addr_out] = output_row[9];
							out10_acc[addr_out] = output_row[10];
							out11_acc[addr_out] = output_row[11];
							out12_acc[addr_out] = output_row[12];
							out13_acc[addr_out] = output_row[13];
							out14_acc[addr_out] = output_row[14];
							out15_acc[addr_out] = output_row[15];
						}

						x = x != (HBM_FFT_SIZE - 2) ? x + 2 : 0; // Two pixels at a time.
						y = x == 0 ? y + 16 : y; // 16 rows at a time, 1 por HBM
					}
				}
			});
		 });
	} catch (cl::sycl::exception const &e) {
		// catch the exception from devices that are not supported.
		std::cout << "An exception is caught when creating a device queue.\n" << e.what()
		      << std::endl;
		std::cout << EXCEPTION_MSG;
		std::terminate();
	}
	ReportTime("FPGA kernel ran for ",submit_event);
}
