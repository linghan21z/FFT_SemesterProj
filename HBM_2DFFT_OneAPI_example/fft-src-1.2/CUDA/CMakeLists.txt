cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

cuda_add_executable(fft fft.cu)

cuda_add_cufft_to_target(fft)

