#ifndef __FFT2D_HPP__  //include guard, preventing the header file from being included multiple times
#define __FFT2D_HPP__  //Once it is determined that the header has not been included before, this line defines the macro to indicate that this header has now been processed.

#define _USE_MATH_DEFINES
#include <cmath>

#include <sycl/ext/intel/ac_types/ac_complex.hpp>  //extension lib support for complex number types
#include <sycl/ext/intel/fpga_extensions.hpp>  //extension lib offer additional functionalities to optimize implementations for FPGAs
#include <sycl/sycl.hpp>  //core SYCL lib provides C++ interface for writing cross-platform parallel programs.

// Large twiddle factors tables  大型旋转因子表
#include "twiddle_factors.hpp"

// Complex single-precision floating-point feedforward FFT / iFFT engine
// Configurable in 4 or 8-parallel versions.
//
// See Mario Garrido, Jesús Grajal, M. A. Sanchez, Oscar Gustafsson:
// Pipeline Radix-2k Feedforward FFT Architectures.
// IEEE Trans. VLSI Syst. 21(1): 23-32 (2013))
//
// The log(points) of the transform must be a compile-time constant argument.
// This FFT engine processes 4 or 8 points for each invocation.
//
// The entry point of the engine is the 'FFTStep' function. This function
// passes 4 or 8 data points through a fixed sequence of processing blocks
// (butterfly, rotation, swap, reorder, multiplications, etc.) and produces
// 4 or 8 output points towards the overall FFT transform.
//
// The engine is designed to be invoked from a loop in a single work-item task.
// When compiling a single work-item task, the compiler leverages pipeline
// parallelism and overlaps the execution of multiple invocations of this
// function. A new instance can start processing every clock cycle

// FFT butterfly building block
template <size_t points, typename T>  // <number of points, numeric type-precision>
std::array<ac_complex<T>, points> Butterfly( //Return Type:this func. returns an std::array of ac_complex<T> type with points elements 
    std::array<ac_complex<T>, points> data) { //Input: The function takes a single argument, data
  std::array<ac_complex<T>, points> res; //an array res to store the results of the butterfly operation.
#pragma unroll //a directive to the compiler, hinting that the loop should be unrolled to optimize performance
  for (int k = 0; k < points; k++) {
    if (k % 2 == 0) {
      res[k] = data[k] + data[k + 1]; //even values of k, addition part of the butterfly operation
    } else {
      res[k] = data[k - 1] - data[k]; //odd values of k, subtraction part of butfly
    }
  }
  return res;
}

// Swap real and imaginary components in preparation for inverse transform
template <size_t points, typename T>
std::array<ac_complex<T>, points> SwapComplex(
    std::array<ac_complex<T>, points> data) {
  std::array<ac_complex<T>, points> res;
#pragma unroll
  for (int k = 0; k < points; k++) {
    res[k].r() = data[k].i(); //swaps the real and imaginary components of each complex number.
    res[k].i() = data[k].r();
  }
  return res;
}

// FFT trivial rotation building block
template <size_t points, typename T>
std::array<ac_complex<T>, points> TrivialRotate(
    std::array<ac_complex<T>, points> data) {
#pragma unroll
  for (int k = 0; k < (points / 4); k++) { //loop iterates through every 4th element
    ac_complex<T> tmp = data[k * 4 + 3]; //index 3,7,11...
    data[k * 4 + 3].r() = tmp.i();  //applies a 90-degree rotation counterclockwise in the complex plane 
    data[k * 4 + 3].i() = -tmp.r(); //eg. 3i+2 -> -2i+3
  }
  return data;
}

// FFT data swap building block associated with trivial rotations
template <size_t points, typename T>
std::array<ac_complex<T>, points> TrivialSwap(
    std::array<ac_complex<T>, points> data) {
#pragma unroll
  for (int k = 0; k < (points / 4); k++) {
    ac_complex<T> tmp = data[k * 4 + 1]; //index 1,5,9...
    data[k * 4 + 1] = data[k * 4 + 2];  //1<->2, 5<->6...
    data[k * 4 + 2] = tmp;
  }
  return data;
}

// FFT data swap building block associated with complex rotations
//even positions get data from the first half, odd positions get data from the second half
template <size_t points, typename T>
std::array<ac_complex<T>, points> Swap(
    std::array<ac_complex<T>, points> data) {
  std::array<ac_complex<T>, points> tmp;
#pragma unroll
  for (int k = 0; k < (points / 2); k++) {
    tmp[k + k] = data[k];  //Assign even-indexed values
    tmp[points - 1 - k - k] = data[points - 1 - k]; //Assign mirrored odd-indexed values
  }
  return tmp;
}

// This function "delays" the input by 'depth' steps
// Input 'data' from invocation N would be returned in invocation N + depth
// The 'shift_reg' sliding window is shifted by 1 element at every invocation
template <typename T>
ac_complex<T> Delay(
    ac_complex<T> data, int depth, ac_complex<T> *shift_reg) {
  shift_reg[depth] = data; //the newest input at the "last" position of the shift window
  return shift_reg[0]; //returns oldest value in the window
}

// FFT data reordering building block. Implements the reordering depicted below
// (for depth = 2). The first valid outputs are in invocation 4
// Invocation count: 0123...          01234567...
// data[0]         : GECA...   ---->      DBCA...
// data[1]         : HFDB...   ---->      HFGE...
//Original Data: G E C A H F D B (alternating values across even and odd indices).
//Reordered Data: D B C A H F G E (swapping certain pairs of data, while delaying others).
template <size_t points, typename T>
std::array<ac_complex<T>, points> ReorderData(
    std::array<ac_complex<T>, points> data, int depth, ac_complex<T> *shift_reg,
    bool toggle) {
  // Use disconnected segments of length 'depth + 1' elements starting at
  // 'shift_reg' to implement the delay elements. At the end of each FFT step,
  // the contents of the entire buffer is shifted by 1 element

#pragma unroll //First Delay and Data Assignment Loop
  for (int k = 0; k < points; k++) { //applies the Delay function to odd indices: data[1], data[3], ...
    data[k * 2 + 1] = Delay(data[k * 2 + 1], depth,
                            shift_reg + (k * 2 + 1 - 1) / 2 * (depth + 1));
  }

  if (toggle) {
#pragma unroll //Conditional Swapping (toggle flag)
    for (int k = 0; k < points; k += 2) {
      ac_complex<T> tmp = data[k]; //swaps adjacent pairs of elements 
      data[k] = data[k + 1];  //data[0] and data[1], data[2] and data[3]...
      data[k + 1] = tmp;
    }
  }

#pragma unroll //Second Delay and Data Assignment Loop
  for (int k = 0; k < points; k += 2) { //applies Delay function to even-indexed data points
    data[k] = Delay(data[k], depth, shift_reg + (points + k) / 2 * (depth + 1));
  }

  return data; //modified data array, which is now reordered and delayed
}

// Produces the twiddle factor associated with a processing stream 'stream',
// at a specified 'stage' during a step 'index' of the computation
//Twiddle Factor: In FFT, twiddle factors are complex exponentials of the form 
//e^(−2πik/n) , where k is an index, and n is the FFT size(8-point,4-point...)
// If there are precomputed twiddle factors for the given FFT size, use them
// This saves hardware resources, because it avoids evaluating 'cos' and 'sin'
// functions
template <int size, size_t points, typename T> 
ac_complex<T> Twiddle(int index, int stage, int stream) {
  //current index of data point in FFT, stage of FFT compu, different parallel computs in fft pipeline
  ac_complex<T> twid;

  constexpr int kTwiddleStages = 5; //number of stages, constexpr can be evaluated at compile time

  // Coalesces the twiddle tables for indexed access
  // Macros defined in twiddle_factors.hpp, contain precomputed values for different FFT sizes.
  constexpr T twiddles_cos_8_points[kTwiddleStages][6][512] = COS8;
  constexpr T twiddles_sin_8_points[kTwiddleStages][6][512] = SIN8;
  constexpr T twiddles_cos_4_points[kTwiddleStages][3][1024] = COS4;
  constexpr T twiddles_sin_4_points[kTwiddleStages][3][1024] = SIN4;

  // Use the precomputed twiddle factors, if available - otherwise, compute them
  int twid_stage = stage >> 1; //shifts the bits of stage one position to right(/2),
    //stages are often halved because the FFT process breaks down the problem into smaller subproblems
  if constexpr (size <= (1 << (kTwiddleStages * 2 + 2))) { //shift binary 1 to the left by 12 positions=2^12=4096
    if constexpr (points == 8) {
      twid.r() =
          twiddles_cos_8_points[twid_stage][stream]
                               [index *
                                ((1 << (kTwiddleStages * 2 + 2)) / size)];
      twid.i() =
          twiddles_sin_8_points[twid_stage][stream]
                               [index *
                                ((1 << (kTwiddleStages * 2 + 2)) / size)];
    } else {
      twid.r() =
          twiddles_cos_4_points[twid_stage][stream]
                               [index *
                                ((1 << (kTwiddleStages * 2 + 2)) / size)];
      twid.i() =
          twiddles_sin_4_points[twid_stage][stream]
                               [index *
                                ((1 << (kTwiddleStages * 2 + 2)) / size)];
    }
  } else { //Dynamic Twiddle Factor Calculation
    // This would generate hardware consuming a large number of resources
    // Instantiated only if precomputed twiddle factors are unavailable
    constexpr double kTwoPi = 2.0f * M_PI;
    int multiplier;

    // The latter 3 streams will generate the second half of the elements
    // In that case phase = 1

    int phase = 0;
    if (stream >= 3) {
      stream -= 3;
      phase = 1;
    }
    switch (stream) {
      case 0:
        multiplier = 2;
        break;
      case 1:
        multiplier = 1;
        break;
      case 2:
        multiplier = 3;
        break;
      default:
        multiplier = 0;
    }
    int pos =
        (1 << (stage - 1)) * multiplier *
        ((index + (size / 8) * phase) & (size / 4 / (1 << (stage - 1)) - 1));
    double theta = -1.0f * kTwoPi / size * (pos & (size - 1));
    twid.r() = cos(theta);
    twid.i() = sin(theta);
  }
  return twid;
}

// FFT complex rotation building block
template <int size, size_t points, typename T>
std::array<ac_complex<T>, points> ComplexRotate(
    std::array<ac_complex<T>, points> data, int index, int stage) {
#pragma unroll
  for (int group = 0; group < points / 4; group++) { //how many sets of 4 complex numbers exist in the data array
#pragma unroll        //point=4 or 8,so group=0, or 0,1
    for (int k = 1; k < 4; k++) { //k=1,2,3
      int stream = group * 3 + k - 1; //stream can only be =0-1-2; 3-4-5.
      data[k + group * 4] *= Twiddle<size, points, T>(index, stage, stream);
    } //data[1]-2-3,5-6-7
  }
  return data;
}

// Process "points" input points towards and a FFT/iFFT of size N, N >= points
// (in order input, bit reversed output). Apply all input points in N / points
// consecutive invocations. Obtain all outputs in N /8 consecutive invocations
// starting with invocation N /points - 1 (outputs are delayed). Multiple
// back-to-back transforms can be executed
//
// 'data' encapsulates points complex single-precision floating-point input
// points 'step' specifies the index of the current invocation
// 'fft_delay_elements' is an array representing a sliding window of size
// N+points*(log(N)-2) 'inverse' toggles between the direct and inverse
// transform
template <int logn, size_t points, typename T>
std::array<ac_complex<T>, points> FFTStep(
    std::array<ac_complex<T>, points> data, int step,
    ac_complex<T> *fft_delay_elements, bool inverse) {
  constexpr int size = 1 << logn; //total size of FFT as N=2^(logn)

  // Swap real and imaginary components if doing an inverse transform
  if (inverse) {
    data = SwapComplex(data);
  }

  // Stage 0 of feed-forward FFT
  data = Butterfly(data);
  data = TrivialRotate(data);
  data = TrivialSwap(data);

  int stage;
  constexpr int kInitStages = points == 8 ? 2 : 1;
  if constexpr (points == 8) {
    // Stage 1
    data = Butterfly(data);
    data = ComplexRotate<size>(data, step & (size / points - 1), 1);
    data = Swap(data);

    stage = 2;
  } else {
    stage = 1;
  }

  // Next stages alternate two computation patterns - represented as
  // a loop to avoid code duplication. Instruct the compiler to fully unroll
  // the loop to increase the amount of pipeline parallelism and allow feed
  // forward execution

#pragma unroll
  for (; stage < logn - 1; stage++) {
    bool complex_stage = stage & 1;  //odd stages 3, 5, ...checks the LSB of the stage variable.

    // Figure out the index of the element processed at this stage
    // Subtract (add modulo size / 8) the delay incurred as data travels
    // from one stage to the next
    int data_index = (step + (1 << (logn - 1 - stage))) & (size / points - 1);

    data = Butterfly(data);

    if (complex_stage) { //odd stages 3, 5
      data = ComplexRotate<size>(data, data_index, stage);
    }

    data = Swap(data);

    // Compute the delay of this stage
    int delay = 1 << (logn - 2 - stage);

    // Reordering multiplexers must toggle every 'delay' steps
    bool toggle = data_index & delay;

    // Assign unique sections of the buffer for the set of delay elements at each stage
    ac_complex<T> *head_buffer = fft_delay_elements + size -
                                 (1 << (logn - stage + kInitStages)) +
                                 points * (stage - kInitStages);

    data = ReorderData(data, delay, head_buffer, toggle);

    if (!complex_stage) { //not odd stages, even 2,4...
      data = TrivialRotate(data);
    }
  }

  // Stage logn - 1
  data = Butterfly(data);

// Shift the contents of the sliding window to accommodate new data. 
// The hardware is capable of shifting the entire contents in parallel 
// if the loop is unrolled. More important, when unrolling this loop 
// each transfer maps to a trivial loop-carried dependency
#pragma unroll
  for (int ii = 0; ii < size + points * (logn - 2) - 1; ii++) {
    fft_delay_elements[ii] = fft_delay_elements[ii + 1];
  }

  if (inverse) {
    data = SwapComplex(data);
  }

  return data;
}

// This utility function bit-reverses an integer 'x' of width 'bits'.
template <int bits>
int BitReversed(int x) {
  int y = 0;
#pragma unroll
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1; //bitwise AND to isolate LSB of x,then sets the LSB of y
    x >>= 1; 
  }
  return y; //eg, If call BitReversed<4>(13) x=1101, finally get y=1011, is bit-reversed of x
}

/* Accesses to DDR memory are efficient if the memory locations are accessed
 * in order. There is significant overhead when accesses are not in order.
 * The penalty is higher if the accesses stride a large number of locations.
 *
 * This function provides the mapping for an alternative memory layout. This
 * layout preserves some amount of linearity when accessing elements from the
 * same matrix row, while bringing closer locations from the same matrix
 * column. The matrix offsets are represented using 2 * log(N) bits. This
 * function swaps bits log(N) - 1 ... log(N) / 2 with bits
 * log(N) + log(N) / 2 - 1 ... log(N).
 *
 * The end result is that 2^(N/2) locations from the same row would still be
 * consecutive in memory, while the distance between locations from the same
 * column would be only 2^(N/2)
 */
template <int logn>
int MangleBits(int x) {
  constexpr int kNB = logn / 2; //kNB represents half of log(N)
  int a95 = x & (((1 << kNB) - 1) << kNB); //Extract the bits log(N)-1 ... log(N)/2 from 'x'
  int a1410 = x & (((1 << kNB) - 1) << (2 * kNB)); // Extract the bits log(N)+log(N)/2-1 ... log(N) from 'x'
  constexpr int mask = ((1 << (2 * kNB)) - 1) << kNB; // Create a mask that covers the bits to be swapped
  a95 = a95 << kNB; // Shift a95 to the higher bit positions (log(N)+log(N)/2-1 ... log(N))
  a1410 = a1410 >> kNB; // Shift a1410 to the lower bit positions (log(N)-1 ... log(N)/2)
  return (x & ~mask) | a95 | a1410; // Combine the original unmodified part of 'x' with the swapped bits
}

/* This kernel reads the matrix data and provides "1<<log_points" data to the
 * FFT engine. （fetches multiple rows of matrix data in parallel）
 */
template <int logn, size_t log_points, typename PipeOut, typename T>
struct Fetch { //reading matrix data from memory and sending the fetched data to FFT engine using a data pipeline
  ac_complex<T> *src;
  int mangle; //The mangle flag controls whether to use an optimized memory layout

  Fetch(ac_complex<T> *src_, int mangle_) : src(src_), mangle(mangle_) {}

  [[intel::kernel_args_restrict]]  // NO-FORMAT: Attribute, an Intel FPGA-specific attribute to enable more efficient memory access by guaranteeing that kernel function arguments do not alias
  void operator()() const {
    constexpr int kN = (1 << logn); //The size of the matrix, N×N
    constexpr int kPoints = (1 << log_points); //num of points fetched in one operation.

    constexpr int kWorkGroupSize = kN; //size of the work group
    constexpr int kIterations = kN * kN / kPoints / kWorkGroupSize; //num of iterations required to process the entire matrix.
               // 2 iteration = 16x16 / 8       /16

    for (int i = 0; i < kIterations; i++) {
      // Local memory for storing 8 rows 
      ac_complex<T> buf[kPoints * kN]; //matrix data is fetched row by row and stored in the buf array.
      for (int work_item = 0; work_item < kWorkGroupSize; work_item++) {
//These are just computing the index or offset need to "read data"
        // Each read fetches 8 matrix points
        int x = (i * kN + work_item) << log_points; //matrix offset for the current work item.

        /* When using the alternative memory layout, each row consists of a set
         * of segments placed far apart in memory. Instead of reading all
         * segments from one row in order, read one segment from each row before
         * switching to the next segment. This requires swapping bits log(N) + 2
         * ... log(N) with bits log(N) / 2 + 2 ... log(N) / 2 in the offset.
         */
        int where, where_global;
        if (mangle) {
          constexpr int kNB = logn / 2;
          int a1210 = x & ((kPoints - 1) << (2 * kNB));
          int a75 = x & ((kPoints - 1) << kNB);
          int mask = ((kPoints - 1) << kNB) | ((kPoints - 1) << (2 * kNB));
          a1210 >>= kNB;
          a75 <<= kNB;
          where = (x & ~mask) | a1210 | a75; 
          where_global = MangleBits<logn>(where);
        } else {
          where = x;
          where_global = where;
        }
//Now come to real "read data" operation
#pragma unroll //Buffered Data Storage: 
//The data is then read into a local buffer(buf) using the computed memory index
        for (int k = 0; k < kPoints; k++) {
          buf[(where & ((1 << (logn + log_points)) - 1)) + k] =
              src[where_global + k];
        }
      }

      for (int work_item = 0; work_item < kWorkGroupSize; work_item++) {
        int row = work_item >> (logn - log_points);
        int col = work_item & (kN / kPoints - 1);

        // Stream fetched data over 8 channels to the FFT engine

        std::array<ac_complex<T>, kPoints> to_pipe; //sent to the FFT engine in chunks
#pragma unroll
        for (int k = 0; k < kPoints; k++) { //Each chunk contains kPoints data points
          to_pipe[k] =
              buf[row * kN + BitReversed<log_points>(k) * kN / kPoints + col];
        } //data is bit-reversed before being written to the PipeOut pipe
        PipeOut::write(to_pipe);
      }         //writing data to a pipe in an Intel FPGA kernel, part of the SYCL (DPC++) programming model.
    }   //PipeOut is likely an output pipe that sends data from the current kernel (the Fetch kernel) to another kernel
  }     //pipe buffer acts as a FIFO queue that can be consumed by another kernel 
};

/* The FFT engine
 * 'inverse' toggles between the direct and the inverse transform
 */ //Compute FFT in a streaming fashion using pipes for data input and output
template <int logn, size_t log_points, typename PipeIn, typename PipeOut,
          typename T>   //log_points: num of points processed in each step (batch size).
struct FFT {
  int inverse;

  FFT(int inverse_) : inverse(inverse_) {}

  void operator()() const {
    constexpr int kN = (1 << logn);
    constexpr int kPoints = (1 << log_points);

    /* The FFT engine requires a sliding window for data reordering; data stored
     * in this array is carried across loop iterations and shifted by 1 element
     * every iteration; all loop dependencies(derived from the uses of this
     * array)are simple transfers between adjacent array elements
     */

    ac_complex<T> fft_delay_elements[kN + kPoints * (logn - 2)]; //sliding window for data reordering
                                            //The logn - 2 portion relates to the stages of FFT pipeline that require reordering.
    // needs to run "kN / kPoints - 1" additional iterations to drain the last outputs
    //It runs for kN * (kN / kPoints) iterations to process the input data, 
    //followed by kN / kPoints - 1 additional iterations to flush the remaining output data from the pipeline. 
    for (unsigned i = 0; i < kN * (kN / kPoints) + kN / kPoints - 1; i++) { // kN/kPoints determines how many iterations are needed to process all points
      std::array<ac_complex<T>, kPoints> data; //process batch

      // Read data from channels (input pipe)
      if (i < kN * (kN / kPoints)) {
        data = PipeIn::read();  //Reading from PipeIn:This retrieves a batch of kPoints FFT points.
      } else { //Padding with zeros: Once all input data has processed, the remaining iterations output zeros, 
        data = std::array<ac_complex<T>, kPoints>{0}; //which allows the pipeline to "drain" (i.e., output the remaining computed results).
      }       //Padding with Zeros: Once all the real data has been processed, zeros are fed into the pipeline as padding. 
              //These zeros don't affect the final FFT result, but they trigger the pipeline to continue processing and push any remaining valid results out of the pipeline.
              //Flushing the Pipeline: This ensures that the entire pipeline gets emptied, and all valid results are "flushed out" to the output.

      // Perform one FFT step
      data = 
          FFTStep<logn>(data, i % (kN / kPoints), fft_delay_elements, inverse);
          //i%(kN / kPoints) provides the current index within the current FFT block, 
          //fft_delay_elements sliding window handles data storage and shifts across iterations.
      
      // Write result to channels (output pipe)
      if (i >= kN / kPoints - 1) { //write data after i reaches kN/kPoints - 1,
        PipeOut::write(data); //ensures all necessary FFT steps for each batch have been completed before the output is sent.
      }
    }
  }
};

/* This kernel receives the FFT results, buffers "1<<log_points" rows and then
 * writes the results transposed in memory. Because "1<<log_points" rows are
 * buffered, "1<<log_points" consecutive columns can be written at a time on
 * each transposed row. This provides some degree of locality. In addition, when
 * using the alternative matrix format, consecutive rows are closer in memory,
 * and this is also beneficial for higher memory access efficiency
 */
template <int logn, size_t log_points, typename PipeIn, typename T>
struct Transpose {
#if defined IS_BSP  //the #if-else-endif used for conditional compilation
  ac_complex<T> *dest;
#else
  // Specify the memory interface when in the SYCL HLS flow
  // The buffer location is set to 1 (identical as the malloc performed in
  // fft2d_demo.cpp)
  // The data width is equal to width of the DDR burst performed by the function
  // This type allows specifying hardware-specific memory properties (like buffer location, data width, latency) in the SYCL HLS flow.
  sycl::ext::oneapi::experimental::annotated_arg< 
      ac_complex<T> *, decltype(sycl::ext::oneapi::experimental::properties{
                           sycl::ext::intel::experimental::buffer_location<1>,
                           sycl::ext::intel::experimental::dwidth<
                               sizeof(ac_complex<T>) * 8 * (1 << log_points)>,
                           sycl::ext::intel::experimental::latency<0>})>
      dest;  //The output memory buffer that stores the transposed results.
#endif

  int mangle; //flag controls whether the alternative memory layout is used.

  Transpose(ac_complex<T> *dest_, int mangle_) : dest(dest_), mangle(mangle_) {}

  [[intel::kernel_args_restrict]]  // NO-FORMAT: Attribute
  void operator()() const {
    constexpr int kN = (1 << logn);
    constexpr int kWorkGroupSize = kN;
    constexpr int kPoints = (1 << log_points);
    constexpr int kIterations = kN * kN / kPoints / kWorkGroupSize;

    for (int t = 0; t < kIterations; t++) {
      //Data Buffering
      ac_complex<T> buf[kPoints * kN]; //The kernel buffers(local array) kPoints rows of FFT results at a time,before being written to the output.
      for (int work_item = 0; work_item < kWorkGroupSize; work_item++) {
        //Reading Data from Pipe
        std::array<ac_complex<T>, kPoints> from_pipe = PipeIn::read(); //reads kPoints complex numbers from the input pipe,correspond to one row of FFT output data.

#pragma unroll
        for (int k = 0; k < kPoints; k++) {
          //Buffering the Read Data
          buf[kPoints * work_item + k] = from_pipe[k]; 
        }//Each row stored in local buf, with consecutive rows stacked vertically in memory.
      }

      for (int work_item = 0; work_item < kWorkGroupSize; work_item++) {
        //Transpose Logic
        int colt = work_item; //colt (column index)
        int revcolt = BitReversed<logn>(colt); //bit-reversed column index, transpose write operation to ensure the FFT results are placed correctly
        //Calculating Memory Offsets
        int i = (t * kN + work_item) >> logn;
        int where = colt * kN + i * kPoints; //the memory location (where)the transposed data will be written
        if (mangle) where = MangleBits<logn>(where); //applies the alternative memory layout

#pragma unroll
        for (int k = 0; k < kPoints; k++) {
          //Writing the Transposed Data
          dest[where + k] = buf[k * kN + revcolt];
        }
      }
    }
  }
};

#endif /* __FFT2D_HPP__ */
