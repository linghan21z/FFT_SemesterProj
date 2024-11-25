#include <math.h>

#define _USE_MATH_DEFINES //this macro enables mathematical constants like M_PI from <cmath>.
#include <cmath>

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "fft2d.hpp"

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
    ac_complex<float> *input_data;
    ac_complex<float> *output_data;
    ac_complex<float> *temp_data;

    //Device memory is allocated depending on the device's support for Unified Shared Memory (USM).
    if (q.get_device().has(sycl::aspect::usm_host_allocations)) {
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

      input_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q);
      output_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);
      temp_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);      

    } else if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
      std::cout << "Using USM device allocations" << std::endl;
      // Allocate FPGA DDR memory.
      input_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
      output_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
      temp_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
    } else {
      std::cerr << "USM device allocations or USM host allocations must be "
                   "supported to run this sample."
                << std::endl;
      std::terminate();
    }

    if (input_data == nullptr || output_data == nullptr ||
        temp_data == nullptr) {
      std::cerr << "Failed to allocate USM memory." << std::endl;
      std::terminate();
    }

    // Copy the input data from host DDR to USM memory
    //host-to-device transfer(if Using USM device allocations)
    q.memcpy(input_data, host_input_data, sizeof(ac_complex<float>) * kN * kN)
        .wait();

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
    using FetchToFFT =
        sycl::ext::intel::pipe<class FetchToFFTPipe,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose =
        sycl::ext::intel::pipe<class FFTToTransposePipe,
                               std::array<ac_complex<float>, kParallelism>, 0>;

    for (int i = 0; i < 2; i++) { //0 to read(Fetch->FFT kernel), 1 to write(FFT->Transpo kernel)
      ac_complex<float> *to_read = i == 0 ? input_data : temp_data;
      ac_complex<float> *to_write = i == 0 ? temp_data : output_data;
      //Implement FFT
      // Start a 1D FFT on the matrix rows/columns
      auto fetch_event = q.single_task<class FetchKernel>(
          Fetch<kLogN, kLogParallelism, FetchToFFT, float>{to_read});

      auto fft_event = q.single_task<class FFTKernel>(
          FFT<kLogN, kLogParallelism, FetchToFFT, FFTToTranspose, float>{
              inverse});

      auto transpose_event = q.single_task<class TransposeKernel>(
          Transpose<kLogN, kLogParallelism, FFTToTranspose, float>{to_write});

      fft_event.wait();
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

    // Copy the output data from the USM memory to the host DDR
    q.memcpy(host_output_data, output_data, sizeof(ac_complex<float>) * kN * kN)
        .wait();

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

    sycl::free(input_data, q);
    free(output_data, q);
    free(temp_data, q);

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
