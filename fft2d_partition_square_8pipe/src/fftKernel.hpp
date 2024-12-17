#include <math.h>

#define _USE_MATH_DEFINES //this macro enables mathematical constants like M_PI from <cmath>.
#include <cmath>

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

#include "fft2d.hpp"

// Validate Accessor Creation
// std::cout << "Buffer buf_in0 size: " << buf_in0.get_range().size() << std::endl;

// event fetch_event;
// fetch_event = runFetchKernel<kLogN, kLogParallelism, 
//                 FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
//                 FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
//                 float>(q);
// fetch_event.wait();

class FetchKernel_accessor;
template <int logn, size_t log_points, 
          typename PipeOut0, typename PipeOut1, typename PipeOut2, typename PipeOut3, 
          typename PipeOut4, typename PipeOut5, typename PipeOut6, typename PipeOut7, 
          typename T>
sycl::event runFetchKernel(sycl::queue &q, int fftIteration,
            // sycl::buffer<ac_complex<float>, 1> &buf_in0,
            // sycl::buffer<ac_complex<float>, 1> &buf_in1,
            // sycl::buffer<ac_complex<float>, 1> &buf_inout0,
            // sycl::buffer<ac_complex<float>, 1> &buf_inout1
            ac_complex<T>* input_data_0, ac_complex<T>* input_data_1,
            ac_complex<T>* temp_data_0, ac_complex<T>* temp_data_1,
            ac_complex<T>* output_data_0, ac_complex<T>* output_data_1) {

#define MEM_CHANNELS
 //using mem_channel 
#if defined(NO_INTERLEAVING) && defined(MEM_CHANNELS)
    sycl::range<1> num_items((1 << logn) * (1 << logn) / 2);

    sycl::buffer buf_in0(input_data_0, num_items, {sycl::property::buffer::mem_channel{1}});
    sycl::buffer buf_in1(input_data_1, num_items, {sycl::property::buffer::mem_channel{2}});
    sycl::buffer buf_inout0(temp_data_0, num_items, {sycl::property::buffer::mem_channel{3}});
    sycl::buffer buf_inout1(temp_data_1, num_items, {sycl::property::buffer::mem_channel{4}});
    sycl::buffer buf_out0(output_data_0, num_items, {sycl::property::buffer::mem_channel{5}});
    sycl::buffer buf_out1(output_data_1, num_items, {sycl::property::buffer::mem_channel{6}});
#else
    sycl::buffer buf_in0(input_data_0);
    sycl::buffer buf_in1(input_data_1);
    sycl::buffer buf_inout0(temp_data_0);
    sycl::buffer buf_inout1(temp_data_1);
    sycl::buffer buf_out0(output_data_0);
    sycl::buffer buf_out1(output_data_1);
#endif

	sycl::event fetch_event = q.submit([&](sycl::handler &h) {
        sycl::accessor input_data_mem0(buf_in0, h, sycl::read_write);
        sycl::accessor input_data_mem1(buf_in1, h, sycl::read_write);
        sycl::accessor temp_data_mem0(buf_inout0, h, sycl::read_write);
        sycl::accessor temp_data_mem1(buf_inout1, h, sycl::read_write);
    
		h.single_task<FetchKernel_accessor>([=]() [[intel::kernel_args_restrict]] {
			constexpr int kN = (1 << logn); //The size of the matrix, N×N
			constexpr int kPoints = (1 << log_points); //num of points fetched in one operation.

			constexpr int kWorkGroupSize = kN; //size of the work group
			constexpr int kIterations = kN * kN / kPoints / kWorkGroupSize; //num of iterations required to process the entire matrix.

			for (int i = 0; i < kIterations; i++) {
      // Local memory for storing 8 rows 
      ac_complex<T> buf0[kPoints / 2 * kN]; //matrix data is fetched row by row and stored in the buf array.
      ac_complex<T> buf1[kPoints / 2 * kN];
      
        for (int work_item = 0; work_item < kWorkGroupSize / 2; work_item++) {
  //These are just computing the index or offset need to "read data"
          // Each read fetches 8 matrix points
          int x = (i * kN + work_item) << log_points; //matrix offset for the current work item.         
          int where, where_global;
          where = x;
          where_global = where;   

          if (fftIteration == 0) {
  #pragma unroll
            for (int k = 0; k < kPoints; k++) { //where bitwise 2^7-1(111 1111) +k
              buf0[(where & ((1 << (logn + log_points)) - 1)) + k] = //just fill in every kpoints bits(k control)
                  input_data_mem0[where_global + k];

              // buf1[(where1 & ((1 << (logn + log_points)) - 1)) + k] = 
              //     src[where_global1 + k];
              buf1[(where & ((1 << (logn + log_points)) - 1)) + k] = 
                  input_data_mem1[where_global + k];
            }
          } else {
  #pragma unroll
            for (int k = 0; k < kPoints; k++) { //where bitwise 2^7-1(111 1111) +k
            buf0[(where & ((1 << (logn + log_points)) - 1)) + k] = //just fill in every kpoints bits(k control)
                temp_data_mem0[where_global + k];

            // buf1[(where1 & ((1 << (logn + log_points)) - 1)) + k] = 
            //     src[where_global1 + k];
            buf1[(where & ((1 << (logn + log_points)) - 1)) + k] = 
                temp_data_mem1[where_global + k];
            }
          }  
        }
      

      //Above is buf[] = src[]
      //Below is to_pipe[] = buf[]
      for (int work_item = 0; work_item < (kWorkGroupSize>>3); work_item++) {
        int row = 0;
        int col = work_item & (kN / kPoints - 1);

        // Stream fetched data over 8 channels to the FFT engine
        std::array<ac_complex<T>, kPoints> 
                  to_pipe0, to_pipe1, to_pipe2, to_pipe3, to_pipe4, to_pipe5, to_pipe6, to_pipe7; //sent to the FFT engine in chunks
#pragma unroll
        for (int k = 0; k < kPoints; k++) { //Each chunk contains kPoints data points
          to_pipe0[k] = buf0[row * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe1[k] = buf0[(row + 1) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe2[k] = buf0[(row + 2) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe3[k] = buf0[(row + 3) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
         
          to_pipe4[k] = buf1[row * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe5[k] = buf1[(row + 1) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe6[k] = buf1[(row + 2) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
          to_pipe7[k] = buf1[(row + 3) * kN + BitReversed<log_points>(k) * kN / kPoints + col];
        } //data is bit-reversed before being written to the PipeOut pipe
        PipeOut0::write(to_pipe0);
        PipeOut1::write(to_pipe1);
        PipeOut2::write(to_pipe2);
        PipeOut3::write(to_pipe3);

        PipeOut4::write(to_pipe4);
        PipeOut5::write(to_pipe5);
        PipeOut6::write(to_pipe6);
        PipeOut7::write(to_pipe7);
      }         //writing data to a pipe in an Intel FPGA kernel, part of the SYCL (DPC++) programming model.
    }
		});
	});
  return fetch_event;
}


// Launch FFT kernel
// event fft_event;
// fft_event = runFFTKernel<kLogN, kLogParallelism, 
//               FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
//               FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
//               FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
//               FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
//               float>(q, inverse);
// fft_event.wait();


class FFTKernel_accessor;
template <int logn, size_t log_points, 
          typename PipeIn0, typename PipeIn1, typename PipeIn2, typename PipeIn3,
          typename PipeIn4, typename PipeIn5, typename PipeIn6, typename PipeIn7,
          typename PipeOut0, typename PipeOut1, typename PipeOut2, typename PipeOut3,
          typename PipeOut4, typename PipeOut5, typename PipeOut6, typename PipeOut7,
          typename T>
sycl::event runFFTKernel(sycl::queue &q, int inverse) {
  // Submit the kernel to the queue
  sycl::event fft_event = q.submit([&](sycl::handler &h) {
    h.single_task<class FFTKernel_accessor>([=]() [[intel::kernel_args_restrict]] {
      constexpr int kN = (1 << logn);
      constexpr int kPoints = (1 << log_points);

      ac_complex<T> fft_delay_elements0[kN + kPoints * (logn - 2)]; //sliding window for data reordering
                                            //The logn - 2 portion relates to the stages of FFT pipeline that require reordering.
			ac_complex<T> fft_delay_elements1[kN + kPoints * (logn - 2)];
			ac_complex<T> fft_delay_elements2[kN + kPoints * (logn - 2)];
			ac_complex<T> fft_delay_elements3[kN + kPoints * (logn - 2)];
			ac_complex<T> fft_delay_elements4[kN + kPoints * (logn - 2)];                                      
			ac_complex<T> fft_delay_elements5[kN + kPoints * (logn - 2)];
			ac_complex<T> fft_delay_elements6[kN + kPoints * (logn - 2)];
			ac_complex<T> fft_delay_elements7[kN + kPoints * (logn - 2)];
    
      // FFT kernel loop
      for (unsigned i = 0; i < kN * (kN / kPoints) / 8 + kN / kPoints - 1; i++) { // kN/kPoints determines how many iterations are needed to process all points
      std::array<ac_complex<T>, kPoints> data0, data1, data2, data3,
                                         data4, data5, data6, data7; //process batch

      // Read data from channels (input pipe)
      if (i < kN * (kN / kPoints) / 8) {
        data0 = PipeIn0::read();  //Reading from PipeIn:This retrieves a batch of kPoints FFT points.
        data1 = PipeIn1::read();
        data2 = PipeIn2::read();
        data3 = PipeIn3::read();
        data4 = PipeIn4::read();
        data5 = PipeIn5::read();
        data6 = PipeIn6::read();
        data7 = PipeIn7::read();
      } else { //Padding with zeros: Once all input data has processed, the remaining iterations output zeros, 
        data0 = std::array<ac_complex<T>, kPoints>{0}; //which allows the pipeline to "drain" (i.e., output the remaining computed results).
        data1 = std::array<ac_complex<T>, kPoints>{0};
        data2 = std::array<ac_complex<T>, kPoints>{0};
        data3 = std::array<ac_complex<T>, kPoints>{0};
        data4 = std::array<ac_complex<T>, kPoints>{0};
        data5 = std::array<ac_complex<T>, kPoints>{0};
        data6 = std::array<ac_complex<T>, kPoints>{0};
        data7 = std::array<ac_complex<T>, kPoints>{0};
      }       //Padding with Zeros: Once all the real data has been processed, zeros are fed into the pipeline as padding. 
              //These zeros don't affect the final FFT result, but they trigger the pipeline to continue processing and push any remaining valid results out of the pipeline.
              //Flushing the Pipeline: This ensures that the entire pipeline gets emptied, and all valid results are "flushed out" to the output.

      // Perform one FFT step
      data0 = FFTStep<logn>(data0, i % (kN / kPoints), fft_delay_elements0, inverse);
      data1 = FFTStep<logn>(data1, i % (kN / kPoints), fft_delay_elements1, inverse);
      data2 = FFTStep<logn>(data2, i % (kN / kPoints), fft_delay_elements2, inverse);
      data3 = FFTStep<logn>(data3, i % (kN / kPoints), fft_delay_elements3, inverse);
      data4 = FFTStep<logn>(data4, i % (kN / kPoints), fft_delay_elements4, inverse);
      data5 = FFTStep<logn>(data5, i % (kN / kPoints), fft_delay_elements5, inverse);
      data6 = FFTStep<logn>(data6, i % (kN / kPoints), fft_delay_elements6, inverse);
      data7 = FFTStep<logn>(data7, i % (kN / kPoints), fft_delay_elements7, inverse);
          //i%(kN / kPoints) provides the current index within the current FFT block, 
          //fft_delay_elements sliding window handles data storage and shifts across iterations.
      
      // Write result to channels (output pipe)
      if (i >= kN / kPoints - 1) { //write data after i reaches kN/kPoints - 1, =3
        PipeOut0::write(data0); //ensures all necessary FFT steps for each batch have been completed before the output is sent.
        PipeOut1::write(data1);
        PipeOut2::write(data2);
        PipeOut3::write(data3);
        PipeOut4::write(data4);
        PipeOut5::write(data5);
        PipeOut6::write(data6);
        PipeOut7::write(data7);
      }
    }
    });
  });
  return fft_event;
}



// event transpose_event;
// transpose_event = runTransposeKernel<kLogN, kLogParallelism, 
//                 FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
//                 FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
//                 float>(q);
// transpose_event.wait();

class TransposeKernel_accessor;
template <int logn, size_t log_points, 
          typename PipeIn0, typename PipeIn1, typename PipeIn2, typename PipeIn3,
          typename PipeIn4, typename PipeIn5, typename PipeIn6, typename PipeIn7, 
          typename T>
sycl::event runTransposeKernel(sycl::queue &q, int fftIteration,
            // sycl::buffer<ac_complex<float>, 1> &buf_inout0,
            // sycl::buffer<ac_complex<float>, 1> &buf_inout1,
            // sycl::buffer<ac_complex<float>, 1> &buf_out0,
            // sycl::buffer<ac_complex<float>, 1> &buf_out1) {
            ac_complex<T>* input_data_0, ac_complex<T>* input_data_1,
            ac_complex<T>* temp_data_0, ac_complex<T>* temp_data_1,
            ac_complex<T>* output_data_0, ac_complex<T>* output_data_1) {

#define MEM_CHANNELS
 //using mem_channel 
#if defined(NO_INTERLEAVING) && defined(MEM_CHANNELS)
    sycl::range<1> num_items((1 << logn) * (1 << logn) / 2);

    sycl::buffer buf_in0(input_data_0, num_items, {sycl::property::buffer::mem_channel{1}});
    sycl::buffer buf_in1(input_data_1, num_items, {sycl::property::buffer::mem_channel{2}});
    sycl::buffer buf_inout0(temp_data_0, num_items, {sycl::property::buffer::mem_channel{3}});
    sycl::buffer buf_inout1(temp_data_1, num_items, {sycl::property::buffer::mem_channel{4}});
    sycl::buffer buf_out0(output_data_0, num_items, {sycl::property::buffer::mem_channel{5}});
    sycl::buffer buf_out1(output_data_1, num_items, {sycl::property::buffer::mem_channel{6}});
#else
    sycl::buffer buf_in0(input_data_0);
    sycl::buffer buf_in1(input_data_1);
    sycl::buffer buf_inout0(temp_data_0);
    sycl::buffer buf_inout1(temp_data_1);
    sycl::buffer buf_out0(output_data_0);
    sycl::buffer buf_out1(output_data_1);
#endif

	sycl::event transpose_event = q.submit([&](sycl::handler &h) {
        sycl::accessor temp_data_mem0(buf_inout0, h, sycl::read_write);
        sycl::accessor temp_data_mem1(buf_inout1, h, sycl::read_write);
        sycl::accessor output_data_mem0(buf_out0, h, sycl::read_write);
        sycl::accessor output_data_mem1(buf_out1, h, sycl::read_write);
        
		h.single_task<TransposeKernel_accessor>([=]() [[intel::kernel_args_restrict]] {
			constexpr int kN = (1 << logn);
			constexpr int kWorkGroupSize = kN;
			constexpr int kPoints = (1 << log_points);
			constexpr int kIterations = kN * kN / kPoints / kWorkGroupSize;

			for (int t = 0; t < kIterations; t++) {
				//Data Buffering
				// ac_complex<T> buf[kPoints * kN];
				ac_complex<T> buf0[kPoints / 2 * kN]; //The kernel buffers(local array) kPoints rows of FFT results at a time,before being written to the output.
				ac_complex<T> buf1[kPoints / 2 * kN]; 
				
				for (int work_item = 0; work_item < (kWorkGroupSize >> 3); work_item++) {
					//Reading Data from Pipe
					std::array<ac_complex<T>, kPoints> from_pipe0 = PipeIn0::read(); //reads kPoints complex numbers from the input pipe,correspond to one row of FFT output data.
					std::array<ac_complex<T>, kPoints> from_pipe1 = PipeIn1::read();
					std::array<ac_complex<T>, kPoints> from_pipe2 = PipeIn2::read();
					std::array<ac_complex<T>, kPoints> from_pipe3 = PipeIn3::read(); 
					std::array<ac_complex<T>, kPoints> from_pipe4 = PipeIn4::read(); 
					std::array<ac_complex<T>, kPoints> from_pipe5 = PipeIn5::read();
					std::array<ac_complex<T>, kPoints> from_pipe6 = PipeIn6::read();
					std::array<ac_complex<T>, kPoints> from_pipe7 = PipeIn7::read();

	#pragma unroll
					for (int k = 0; k < kPoints; k++) {
						//Buffering the Read Data
						buf0[kPoints * work_item + k] = from_pipe0[k]; 
						buf0[kPoints * work_item + kN + k] = from_pipe1[k];
						buf0[kPoints * work_item + 2 * kN + k] = from_pipe2[k]; 
						buf0[kPoints * work_item + 3 * kN + k] = from_pipe3[k]; 

						buf1[kPoints * work_item + 0 * kN + k] = from_pipe4[k]; 
						buf1[kPoints * work_item + 1 * kN + k] = from_pipe5[k]; 
						buf1[kPoints * work_item + 2 * kN + k] = from_pipe6[k]; 
						buf1[kPoints * work_item + 3 * kN + k] = from_pipe7[k];
					
					}//Each row stored in local buf, with consecutive rows stacked vertically in memory.
				}

				//This part does not need any change, just store dest[]=buf[kPoints*kN] same as before
				//similar to "Fetch", buf[]=src[], no change
				for (int work_item = 0; work_item < kWorkGroupSize / 2; work_item++) {
					//Transpose Logic
					int colt = work_item; //colt (column index)
					int revcolt = BitReversed<logn>(colt); //bit-reversed column index, transpose write operation to ensure the FFT results are placed correctly
					//Calculating Memory Offsets
					int i = (t * kN + work_item) >> logn;
					int where = colt * kN + i * kPoints; //the memory location (where)the transposed data will be written
					// if (mangle) where = MangleBits<logn>(where); //applies the alternative memory layout
          
          if (fftIteration == 0) {
#pragma unroll
            for (int k = 0; k < kPoints; k++) {
						//Writing the Transposed Data            
              temp_data_mem0[where + k] = buf0[k * kN + revcolt];
              temp_data_mem1[where + k] = buf1[k * kN + revcolt];
            }
          } else {
	#pragma unroll
            for (int k = 0; k < kPoints; k++) {           
              output_data_mem0[where + k] = buf0[k * kN + revcolt];
              output_data_mem1[where + k] = buf1[k * kN + revcolt];
            }  
          }
				}
			}
		});
	});
  return transpose_event;
}
