#include <math.h>

#define _USE_MATH_DEFINES //this macro enables mathematical constants like M_PI from <cmath>.
#include <cmath>

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "fft2d.hpp"
#include "fftKernel.hpp"
// using namespace sycl;

// Forward declarations
void TestFFT(bool inverse); //reordering of bits, fft/ifft
template <int n> //Templates allow for optimizations at compile-time
int Coordinates(int iteration, int i);
template <int lognr_points>
void FourierTransformGold(ac_complex<double> *data, bool inverse); //a reference implementation of FFT (gold standard)
template <int lognr_points>
void FourierStage(ac_complex<double> *data); //handle individual stages of the FFT pipeline

//**argv: This is a pointer to an array of character pointers (i.e., char*), representing the command-line arguments. 
//each element of argv points to a string (a char[]) that corresponds to an individual argument 
int main(int argc, char **argv) { //main() controls the execution of FFT tests under different configurations
  //handle command-line arguments:
  //argc (argument count): This integer represents the num of command-line arguments passed to the program. 
  //it includes the name of the program itself, so argc is always at least 1.
  //argv (argument vector): This array of strings (char pointers) contains the actual arguments. 
  //argv[0] is typically the name of the program, and argv[1] to argv[argc - 1] are the additional arguments passed by the user.
  //eg: ./fft2d_demo normal  argc=2 ("./fft2d_demo" and "normal")
  if (argc == 1) { //no arguments are passed, run all of 4 modes/configs
    std::cout << "No program argument was passed, running all fft2d variants"
              << std::endl;

    // test FFT transform with ordered memory layout
    TestFFT(false);
    // test inverse FFT transform with ordered memory layout
    TestFFT(true);
    // test FFT transform with alternative memory layout
    // TestFFT(true, false);
    // test inverse FFT transform with alternative memory layout
    // TestFFT(true, true);

  } else {
    std::string mode = argv[1];

    bool inverse{};

    if (mode == "normal") {
      inverse = false;
    } else if (mode == "inverse") {
      inverse = true;
    } else { //std::cerr:standard output stream "standard error."
      std::cerr << "Usage: fft2d <mode>" << std::endl;
      std::cerr << "Where mode can be normal|inverse|all"
                << std::endl;
      std::terminate(); //Immediate Termination without unwinding the stack or performing cleanup
    }

    TestFFT(inverse);
  }
  return 0;
}

void TestFFT(bool inverse) {
  try { 
  //SYCL Device and Queue Setup
    // Device selector selection
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  //FPGA emulator
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list queue_properties{
        sycl::property::queue::enable_profiling()};
    sycl::queue q =
        sycl::queue(selector, fpga_tools::exception_handler, queue_properties);

    sycl::device device = q.get_device();

    // Print out the device information.
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>() << std::endl;

////////////////////////////////


    // Define the log of the FFT size on each dimension and the level of parallelism to implement
#if FPGA_SIMULATOR
    // Force small sizes in simulation mode to reduce simulation time
    constexpr int kLogN = 4; //KN=16
    constexpr int kParallelism = 4;
#else
    constexpr int kLogN = LOGN;
    constexpr int kParallelism = PARALLELISM;
#endif

    static_assert(kParallelism == 4 || kParallelism == 8,
                  "The FFT kernel implementation only supports 4-parallel and "
                  "8-parallel FFTs.");

    constexpr int kN = 1 << kLogN; //size of fft matrix, eg 16*16
    constexpr int kLogParallelism = kParallelism == 8 ? 3 : 2; //log8=3,log4=2

    // Host memory
    ac_complex<float> *host_input_data =
        (ac_complex<float> *)std::malloc(sizeof(ac_complex<float>) * kN * kN); //kN * kN is num of all elements in a matrix
    ac_complex<float> *host_output_data =
        (ac_complex<float> *)std::malloc(sizeof(ac_complex<float>) * kN * kN);
    ac_complex<double> *host_verify =
        (ac_complex<double> *)std::malloc(sizeof(ac_complex<double>) * kN * kN);
    ac_complex<double> *host_verify_tmp =
        (ac_complex<double> *)std::malloc(sizeof(ac_complex<double>) * kN * kN);

    //checks whether the memory allocation for host data is successful
    if ((host_input_data == nullptr) || (host_output_data == nullptr) ||
        (host_verify == nullptr) || (host_verify_tmp == nullptr)) {
      std::cerr << "Failed to allocate host memory with malloc." << std::endl;
      std::terminate();
    }

    // Initialize input and produce verification data
    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        // int where = mangle ? MangleBits<kLogN>(Coordinates<kN>(i, j))
                          //  : Coordinates<kN>(i, j);
        int where = Coordinates<kN>(i, j);
        host_verify[Coordinates<kN>(i, j)].r() = host_input_data[where].r() =
            (float)((double)rand() / (double)RAND_MAX);
        host_verify[Coordinates<kN>(i, j)].i() = host_input_data[where].i() =
            (float)((double)rand() / (double)RAND_MAX);
      }
    }


// Device memory
    ac_complex<float> *input_data_0;
    ac_complex<float> *input_data_1;
    ac_complex<float> *output_data_0;
    ac_complex<float> *output_data_1;
    ac_complex<float> *temp_data_0;
    ac_complex<float> *temp_data_1;

    //Device memory is allocated depending on the device's support for Unified Shared Memory (USM).
    if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
      std::cout << "Using USM device allocations" << std::endl;
      // Allocate FPGA DDR memory. - BURST-NON-ALIGNED
      // input_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      // input_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      // output_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      // output_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      // temp_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      // temp_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);

    //Ensure properly aligned for the hardware. Misaligned memory can cause segmentation faults, especially on FPGA hardware
      // size_t base_addr_align = device.get_info<sycl::info::device::mem_base_addr_align>();
      // size_t max_alloc_size = device.get_info<sycl::info::device::max_mem_alloc_size>();
      // //Memory base address alignment (in bytes): 128 // Maximum memory allocation size: 66999881728 bytes
      // std::cout << "Memory base address alignment (in bytes): " << base_addr_align / 8 << std::endl;
      // std::cout << "Maximum memory allocation size: " << max_alloc_size << " bytes" << std::endl;
      
      //aligned_malloc_shared - BURST-NON-ALIGNED // aligned_alloc_device - BURST-NON-ALIGNED //aligned_alloc
      input_data_0 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);
      input_data_1 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);
      temp_data_0 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);
      temp_data_1 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);
      output_data_0 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);
      output_data_1 = sycl::aligned_alloc_shared<ac_complex<float>>(128, kN * kN / 2, q);


    } else if (q.get_device().has(sycl::aspect::usm_host_allocations)) {
      std::cout << "Using USM host allocations" << std::endl;
      // No device allocations means that we are probably in a SYCL HLS flow

#if defined IS_BSP
      auto prop_list = sycl::property_list{};
#else
      // In the SYCL HLS flow, we need to define the memory interface.
      // For that we need to assign a location to the memory being accessed.
      auto prop_list = sycl::property_list{
          sycl::ext::intel::experimental::property::usm::buffer_location(1)};
#endif
      // input_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q);
      // output_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);
      // temp_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);
      input_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      input_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q);
      output_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q, prop_list);
      output_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q, prop_list);
      temp_data_0 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q, prop_list);
      temp_data_1 = sycl::malloc_device<ac_complex<float>>(kN * kN / 2, q, prop_list);
      
    } else {
      std::cerr << "USM device allocations or USM host allocations must be "
                   "supported to run this sample."
                << std::endl;
      std::terminate();
    }

    if (input_data_0 == nullptr || input_data_1 == nullptr || 
        output_data_0 == nullptr || output_data_1 == nullptr ||
        temp_data_0 == nullptr || temp_data_1 == nullptr) {
      std::cerr << "Failed to allocate USM memory." << std::endl;
      std::terminate();
    }
    // Copy the input data from host DDR to USM memory
    // q.memcpy(input_data, host_input_data, sizeof(ac_complex<float>) * kN * kN).wait();


// Assuming input_data_0 and input_data_1 are allocated with enough memory
    int chunk_size = 4 * kN; // Size of each chunk in the range
    int num_chunks = kN / 8; // Total number of chunks to process

    int offset_in = 0;  // Offset in host_input_data
    int offset_out_0 = 0; // Offset in input_data_0
    int offset_out_1 = 0; // Offset in input_data_1

    // //Validate memory boundaries
    // if (offset_out_0 + chunk_size > kN * kN / 2 || offset_out_1 + chunk_size > kN * kN / 2) {
    //   std::cerr << "Memory copy exceeds buffer size!" << std::endl;
    //   std::terminate();
    // }
        //if do not use memcpy---
    // for (int i = 0; i < num_chunks; i++) {
    //   // Copy the current chunk to input_data_0
    //   offset_in = i * 2 * chunk_size;
    //   q.memcpy(input_data_0 + offset_out_0, host_input_data + offset_in, 
    //           sizeof(ac_complex<float>) * chunk_size).wait();
    //   offset_out_0 += chunk_size;

    //   // Copy the next chunk to input_data_1
    //   offset_in = (i * 2 + 1) * chunk_size;
    //   q.memcpy(input_data_1 + offset_out_1, host_input_data + offset_in, 
    //           sizeof(ac_complex<float>) * chunk_size).wait();
    //   offset_out_1 += chunk_size;
    // }

    for (int i = 0; i < num_chunks; i++) {
    offset_in = i * 2 * chunk_size;

    // Copy the current chunk to input_data_0
    for (int j = 0; j < chunk_size; ++j) {
        input_data_0[offset_out_0 + j] = host_input_data[offset_in + j];
    }
    offset_out_0 += chunk_size;

    // Copy the next chunk to input_data_1
    offset_in = (i * 2 + 1) * chunk_size;
    for (int j = 0; j < chunk_size; ++j) {
        input_data_1[offset_out_1 + j] = host_input_data[offset_in + j];
    }
    offset_out_1 += chunk_size;
  }


    std::cout << "Launching a " << kN * kN << " points " << kParallelism
              << "-parallel " << (inverse ? "inverse " : "")
              // << "FFT transform (" << (mangle ? "alternative" : "ordered")
              << "FFT transform (" << ("ordered")
              << " data layout)" << std::endl;

    /*
     * A 2D FFT transform requires applying a 1D FFT transform to each matrix
     * row followed by a 1D FFT transform to each column of the intermediate
     * result.
     * A single FFT engine can process rows and columns back-to-back. However,
     * as matrix data is stored in global memory, the efficiency of memory
     * accesses will impact the overall performance. Accessing consecutive
     * memory locations leads to efficient access patterns. However, this is
     * obviously not possible when accessing both rows and columns.
     *
     * The implementation is divided between three concurrent SYCL kernels, as
     * depicted below:
     *
     *  --------------------      --------------      --------------------------
     *  | read matrix rows | ---> | FFT engine | ---> | bit-reverse, transpose |
     *  |                  |      |            |      |    and write matrix    |
     *  --------------------      --------------      --------------------------
     *
     * This sequence of kernels does back-to-back row processing followed by a
     * data transposition and writes the results back to memory. The host code
     * runs these kernels twice to produce the overall 2D FFT transform
     *
     *
     * These kernels transfer data through pipes.
     * This avoids the need to read and write intermediate data using global
     * memory.
     *
     * In many cases the FFT engine is a building block in a large application.
     * In this case, the memory layout of the matrix can be altered to achieve
     * higher memory transfer efficiency. This implementation demonstrates how
     * an alternative memory layout can improve performance. The host switches
     * between the two memory layouts using a kernel argument. See the
     * 'MangleBits' function for additional details.
     */

    double start_time;
    double end_time;

    // This is a limitation of the design
    //a compile-time assertion ensures KN divided by parallelism >= the parallelism itself
    //kParallelism is how many data points are processed in parallel by FFT kernel at each stage
    //16/4-parallel or 64/8-parallel. If this condition fails, the program won’t compile;
    //If less than, there would not be enough data points to keep 
    //all the parallel units busy, leading to underutilization of resources.
    static_assert(kN / kParallelism >= kParallelism); 

    // Kernel to kernel pipes
    using FetchToFFT0 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe0,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT1 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe1, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT2 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe2, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT3 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe3, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT4 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe4,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT5 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe5, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT6 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe6, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FetchToFFT7 = 
        sycl::ext::intel::pipe<class FetchToFFTPipe7, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    ////////////////
    using FFTToTranspose0 =
        sycl::ext::intel::pipe<class FFTToTransposePipe0,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose1 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe1, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose2 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe2, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose3 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe3, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose4 =
        sycl::ext::intel::pipe<class FFTToTransposePipe4,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose5 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe5, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose6 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe6, 
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose7 = 
        sycl::ext::intel::pipe<class FFTToTransposePipe7, 
                               std::array<ac_complex<float>, kParallelism>, 0>;


// #define NO_INTERLEAVING //using cmake to get it 
/*
#define MEM_CHANNELS

 //using mem_channel 
#if defined(NO_INTERLEAVING) && defined(MEM_CHANNELS)
    sycl::range<1> num_items(kN * kN / 2);

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

    // Validate Accessor Creation
    std::cout << "Buffer buf_in0 size: " << buf_in0.get_range().size() << std::endl;
*/

      // Debug Accessor-to-Pointer Conversion
      // if (input_data_mem0.size() == 0) {
      //   std::cerr << "input_data_mem0 accessor size is zero!" << std::endl;
      //   std::terminate();
      // }
      // if (temp_data_mem0.size() == 0) {
      //   std::cerr << "temp_data_mem0 accessor size is zero!" << std::endl;
      //   std::terminate();
      // }
      // std::cout << "input_data_mem0 range: " << input_data_mem0.get_range().size() << std::endl;
      // std::cout << "temp_data_mem0 range: " << temp_data_mem0.get_range().size() << std::endl;
   
    for (int i = 0; i < 2; i++) { 
        // ac_complex<float> *to_read0 = i == 0 ? input_data : temp_data; 
        // ac_complex<float> *to_read1 = i == 0 ? input_data + 4 * kN : temp_data + 4 * kN;
        // ac_complex<float> *to_write = i == 0 ? temp_data : output_data;
    
    /* q.submit([&](sycl::handler &h) {
        sycl::accessor input_data_mem0(buf_in0, h, sycl::read_write);
        sycl::accessor input_data_mem1(buf_in1, h, sycl::read_write);
        sycl::accessor temp_data_mem0(buf_inout0, h, sycl::read_write);
        sycl::accessor temp_data_mem1(buf_inout1, h, sycl::read_write);
        sycl::accessor output_data_mem0(buf_out0, h, sycl::read_write);
        sycl::accessor output_data_mem1(buf_out1, h, sycl::read_write);

      h.single_task< >([=]() [[intel::kernel_args_restrict]] {
      
        // try to Access data directly
        // ac_complex<float> *to_read0 = i == 0 ? &input_data_mem0[0] : &temp_data_mem0[0]; 
        // ac_complex<float> *to_read1 = i == 0 ? &input_data_mem1[0] : &temp_data_mem1[0];
        // ac_complex<float> *to_write0 = i == 0 ? &temp_data_mem0[0] : &output_data_mem0[0];
        // ac_complex<float> *to_write1 = i == 0 ? &temp_data_mem1[0] : &output_data_mem1[0];
      });
      }); //q.submit([&](handler &h) {
    */ 
      //----------------------------------

      /*  //  Check Conditional Logic
        std::cout << "i=" << i << ", selecting: "
          << ((i == 0) ? "input_data_mem0" : "temp_data_mem0") << std::endl;
        //   // Print Pointer Status
        // auto ptr = input_data_mem0.get_multi_ptr<sycl::access::decorated::no>().get();
        // if (!ptr) {
        //     printf("Pointer from input_data_mem0 is null!\n");
        // } //This method seems cannot get pointer.
        if (!to_read0 || !to_read1 || !to_write0 || !to_write1) {
          std::cerr << "Error: One or more pointers are null!" << std::endl;
          std::terminate();
        }
      */
      
      /* // Implement FFT
      Start a 1D FFT on the matrix rows/columns
      auto fetch_event = q.single_task<class FetchKernel>(
          Fetch<kLogN, kLogParallelism, 
                FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
                FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
                float>{to_read0, to_read1});

      auto fft_event = q.single_task<class FFTKernel>(
          FFT<kLogN, kLogParallelism, 
              FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
              FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
              FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
              FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
              float>{inverse});

      auto transpose_event = q.single_task<class TransposeKernel>(
          Transpose<kLogN, kLogParallelism, 
                    FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
                    FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
                    float>{to_write0, to_write1});
      */
      // ac_complex<float> *to_read0;
      // ac_complex<float> *to_read1;
      // ac_complex<float> *to_write0;
      // ac_complex<float> *to_write1;
/*
      auto fetch_event = q.submit([&](sycl::handler &h) {
        sycl::accessor input_data_mem0(buf_in0, h, sycl::read_write);
        sycl::accessor input_data_mem1(buf_in1, h, sycl::read_write);
        sycl::accessor temp_data_mem0(buf_inout0, h, sycl::read_write);
        sycl::accessor temp_data_mem1(buf_inout1, h, sycl::read_write);
        h.single_task<class FetchKernel>([=]() {          
          if (i == 0) {
            Fetch<kLogN, kLogParallelism, 
                FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
                FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
                float>{&input_data_mem0[0], &input_data_mem1[0]};
          } else {
            Fetch<kLogN, kLogParallelism, 
                FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
                FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
                float>{&temp_data_mem0[0], &temp_data_mem1[0]};
          }                   
        });
      });
      
      auto fft_event = q.submit([&](sycl::handler &h) {
        h.single_task<class FFTKernel>([=]() {
          FFT<kLogN, kLogParallelism, 
              FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
              FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
              FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
              FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
              float>{inverse};
        });
      });      

      auto transpose_event = q.submit([&](sycl::handler &h) {
        sycl::accessor temp_data_mem0(buf_inout0, h, sycl::read_write);
        sycl::accessor temp_data_mem1(buf_inout1, h, sycl::read_write);
        sycl::accessor output_data_mem0(buf_out0, h, sycl::read_write);
        sycl::accessor output_data_mem1(buf_out1, h, sycl::read_write);
        // to_write0 = i == 0 ? &temp_data_mem0[0] : &output_data_mem0[0];
        // to_write1 = i == 0 ? &temp_data_mem1[0] : &output_data_mem1[0];
        h.single_task<class TransposeKernel>([=]() {          
          if (i == 0) {
            Transpose<kLogN, kLogParallelism, 
                FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
                FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
                float>{&temp_data_mem0[0], &temp_data_mem1[0]};
          } else {
            Transpose<kLogN, kLogParallelism, 
                FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
                FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
                float>{&output_data_mem0[0], &output_data_mem1[0]};
          }          
        });
      });

      fft_event.wait();
      transpose_event.wait();
*/
      sycl::event fetch_event;
      sycl::event fft_event;
      sycl::event transpose_event;

      std::cout << "fetch_event come here" << std::endl;
      fetch_event = runFetchKernel<kLogN, kLogParallelism, 
                    FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
                    FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
                    // float>(q, i, buf_in0, buf_in1, buf_inout0, buf_inout1);
                    float>(q, i, 
                    input_data_0, input_data_1, 
                    temp_data_0, temp_data_1,
                    output_data_0, output_data_1);
                    
      std::cout << "fft_event come here" << std::endl;
      fft_event = runFFTKernel<kLogN, kLogParallelism, 
                  FetchToFFT0, FetchToFFT1, FetchToFFT2, FetchToFFT3, 
                  FetchToFFT4, FetchToFFT5, FetchToFFT6, FetchToFFT7, 
                  FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
                  FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
                  float>(q, inverse);
      
      std::cout << "transpose_event come here" << std::endl;
      transpose_event = runTransposeKernel<kLogN, kLogParallelism, 
                FFTToTranspose0, FFTToTranspose1, FFTToTranspose2, FFTToTranspose3,
                FFTToTranspose4, FFTToTranspose5, FFTToTranspose6, FFTToTranspose7, 
                // float>(q, i, buf_inout0, buf_inout1, buf_out0, buf_out1);
                float>(q, i, 
                    input_data_0, input_data_1, 
                    temp_data_0, temp_data_1,
                    output_data_0, output_data_1);
      
      std::cout << "wait fft_event" << std::endl;
      fft_event.wait();
      std::cout << "wait transpose_event" << std::endl;
      transpose_event.wait();


      //Time
      if (i == 0) {  //0 to read(Fetch->FFT kernel) 1st fft start
        start_time = fetch_event.template get_profiling_info<
            sycl::info::event_profiling::command_start>();
      } else { //1 to transpose and write back(FFT-> Transpose kernel) 2nd fft end
        end_time = transpose_event.template get_profiling_info<
            sycl::info::event_profiling::command_end>();
      }
    }

    //it's the time for twice fft - 2d fft time
    double kernel_runtime = (end_time - start_time) / 1.0e9; //ns (unit)-> s

    // // Copy the output data from the USM memory to the host DDR
    // q.memcpy(host_output_data, output_data, sizeof(ac_complex<float>) * kN * kN)
    //     .wait();

    int chunk_size_out = 4 * kN; // Size of each chunk
    int num_chunks_out = kN / 8; // Total number of chunks

    int offset_host_out = 0;   // Offset in host_input_data
    int offset_out_0_ = 0;   // Offset in input_data_0
    int offset_out_1_ = 0;   // Offset in input_data_1

    for (int i = 0; i < num_chunks_out; i++) {
        // Copy from input_data_0 back to host_input_data
        offset_host_out = i * 2 * chunk_size_out;
        q.memcpy(host_output_data + offset_host_out, output_data_0 + offset_out_0_,
                sizeof(ac_complex<float>) * chunk_size_out).wait();
        offset_out_0_ += chunk_size_out;

        // Copy from input_data_1 back to host_input_data
        offset_host_out = (i * 2 + 1) * chunk_size_out;
        q.memcpy(host_output_data + offset_host_out, output_data_1 + offset_out_1_,
                sizeof(ac_complex<float>) * chunk_size_out).wait();
        offset_out_1_ += chunk_size_out;
    }


    std::cout << "Processing time = " << kernel_runtime << "s" << std::endl;
    
    //how many data points (or grid points in the 2D FFT matrix) are processed per second
    double gpoints_per_sec = ((double)kN * kN / kernel_runtime) * 1e-9; //perf. of throughput(Gpoints/sec)
    //a single point in an FFT requires roughly 10 flops (so 2×5 factor)
    double gflops = 2 * 5 * kN * kN * (log((float)kN) / log((float)2)) /
                    (kernel_runtime * 1e9); ///perf of floating-point operations per second(Gflops)
    //from results, gpoints*100=gflops, 100 derives from 2*5*log2(KN), in the test KN=1024=2^10

    std::cout << "Throughput = " << gpoints_per_sec << " Gpoints / sec ("
              << gflops << " Gflops)" << std::endl;

    // Check signal to noise ratio

    // Run reference code on the host,to verify the accuracy of the kernel's results
    //fft-transpose-fft-transpose
    for (int i = 0; i < kN; i++) {
      FourierTransformGold<kLogN>(host_verify + Coordinates<kN>(i, 0), inverse);
    }
    //transpose and tmp-store-intermidiate-result
    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        host_verify_tmp[Coordinates<kN>(j, i)] =
            host_verify[Coordinates<kN>(i, j)];
      }
    }

    for (int i = 0; i < kN; i++) {         //addr + linear offset
      FourierTransformGold<kLogN>(host_verify_tmp + Coordinates<kN>(i, 0),
                                  inverse);
    }

    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        host_verify[Coordinates<kN>(j, i)] =
            host_verify_tmp[Coordinates<kN>(i, j)];
      }
    }
    //SNR
    double magnitude_sum = 0;
    double noise_sum = 0;
    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        int where = Coordinates<kN>(i, j);
        //real^2+imaginary^2
        double magnitude = (double)host_verify[Coordinates<kN>(i, j)].r() *
                               (double)host_verify[Coordinates<kN>(i, j)].r() +
                           (double)host_verify[Coordinates<kN>(i, j)].i() *
                               (double)host_verify[Coordinates<kN>(i, j)].i();
        //diff-real^2 + diff-imaginary^2
        double noise = (host_verify[Coordinates<kN>(i, j)].r() -
                        (double)host_output_data[where].r()) *
                           (host_verify[Coordinates<kN>(i, j)].r() -
                            (double)host_output_data[where].r()) +
                       (host_verify[Coordinates<kN>(i, j)].i() -
                        (double)host_output_data[where].i()) *
                           (host_verify[Coordinates<kN>(i, j)].i() -
                            (double)host_output_data[where].i());
        //Traverse all points(rows and columns) and accumulate
        magnitude_sum += magnitude;
        noise_sum += noise;
      }
    }
    //SNR between the reference and kernel results is calculated
    double db = 10 * log(magnitude_sum / noise_sum) / log(10.0);
    //test is marked as PASSED if the SNR > a certain threshold (120 dB)
    std::cout << "Signal to noise ratio on output sample: " << db << std::endl;
    std::cout << " --> " << (db > 120 ? "PASSED" : "FAILED") << std::endl;

    // sycl::free(input_data, q);
    // free(output_data, q);
    // free(temp_data, q);

    sycl::free(input_data_0, q);
    free(input_data_1, q);
    free(output_data_0, q);
    free(output_data_1, q);
    free(temp_data_0, q);
    free(temp_data_1, q);



  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
}

/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
template <int n> //number of points, size of one dimension in the 2D array
int Coordinates(int iteration, int i) {
  return iteration * n + i; //index calcu: convert 2D coordinates to a linear index(1D array)
}

// Reference Fourier transform
template <int lognr_points>
void FourierTransformGold(ac_complex<double> *data, bool inverse) {
  constexpr int kNrPoints = 1 << lognr_points;

  // The inverse requires swapping the real and imaginary component
  if (inverse) {
    for (int i = 0; i < kNrPoints; i++) {
      double tmp = data[i].r();
      data[i].r() = data[i].i();
      data[i].i() = tmp;
    }
  }

  // Do a FT recursively 递归执行
  FourierStage<lognr_points>(data);

  // The inverse requires swapping the real and imaginary component
  if (inverse) {
    for (int i = 0; i < kNrPoints; i++) {
      double tmp = data[i].r();
      data[i].r() = data[i].i();
      data[i].i() = tmp;
    }
  }
}

//implements a recursive step of the FT
//recursively breaking down the input data into smaller arrays, processing them, and combining the results. 
template <int lognr_points>
void FourierStage(ac_complex<double> *data) {
  if constexpr (lognr_points > 0) {
    constexpr int kNrPoints = 1 << lognr_points;

    ac_complex<double> *half1 = (ac_complex<double> *)malloc(
        sizeof(ac_complex<double>) * kNrPoints / 2);
    ac_complex<double> *half2 = (ac_complex<double> *)malloc(
        sizeof(ac_complex<double>) * kNrPoints / 2);

    if (half1 == nullptr || half2 == nullptr) { //to make sure allocate mem to half1 successfully
      std::cerr << "Failed to allocate memory in validation function."
                << std::endl;
      std::terminate();
    }

    for (int i = 0; i < kNrPoints / 2; i++) {
      half1[i] = data[2 * i]; //half1[]0-1-2...=data[]0-2-4 even-index
      half2[i] = data[2 * i + 1]; //0-1-2...1-3-5 odd-index
    }

    FourierStage<lognr_points - 1>(half1); //do half-operation again - recursively
    FourierStage<lognr_points - 1>(half2); //until kNrPoints = 1 << 1 =2,smallest 2*2 matrix

    for (int i = 0; i < kNrPoints / 2; i++) {
      data[i].r() = half1[i].r() + 
                    cos(2 * M_PI * i / kNrPoints) * half2[i].r() + //+是因为是对乘积的实部进行加，已经带负号了
                    sin(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i].i() = half1[i].i() -
                    sin(2 * M_PI * i / kNrPoints) * half2[i].r() +
                    cos(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i + kNrPoints / 2].r() =
          half1[i].r() - cos(2 * M_PI * i / kNrPoints) * half2[i].r() -
          sin(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i + kNrPoints / 2].i() =
          half1[i].i() + sin(2 * M_PI * i / kNrPoints) * half2[i].r() -
          cos(2 * M_PI * i / kNrPoints) * half2[i].i();
    }

    free(half1); //The allocated memory is freed to prevent memory leaks.
    free(half2);
  }
}
