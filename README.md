# FFT2D Implementations and Variations  

This repository contains multiple implementations and variations of 2D Fast Fourier Transform (FFT) using Intel oneAPI, including optimizations with pipes, memory channels, and HBM.  

## Implementations  

### Original Implementation  
**`fft2d`**  
- Single pipe (PipeIn, PipeOut) transferring 8 points of data between kernels (FetchKernel → FFTKernel → TransposeKernel).  
- **Status:** Works.  

### Variations  

1. **`fft2d_squareM1`**  
   - Modified to use a single pipe with 32 points of data split into 4 arrays (`data0`, `data1`, `data2`, `data3`) for 4 FFT steps.  
   - **Issue:** Pipe capacity only allows 8 points, causing failure.  

2. **`fft2d_squareM2`**  
   - Uses 4 pipes to transfer 32 points of data for FFT. Each pipe processes data from different rows consecutively.  
   - **Status:** Works, reduced latency.  

3. **`fft2d_del_mangle`**  
   - Removes "mangle" logic from `fft2d`, reducing area estimates and FetchKernel cycles while keeping FFT and TransposeKernel unchanged.  
   - **Status:** Works, optimized area.  

4. **`fft2d_square_4pipe`**  
   - Based on `fft2d_squareM2`, removes "mangle".  
   - **Status:** Works, reduced latency compared to single pipe.  

5. **`fft2d_square_8pipe`**  
   - Based on `fft2d_del_mangle`, uses 8 pipes to transfer 64 points of data. Each pipe processes data consecutively from different rows.  
   - **Status:** Works, no latency improvement.  

6. **`fft2d_square_8pipe_partition`**  
   - Based on `fft2d_square_8pipe`, duplicates FFTKernel to handle 8 pipes (4 each).  
   - **Status:** Works, no latency improvement.  

7. **`fft2d_hbm` / `fft2d_hbm_usmHost`**  
   - Based on `fft2d_del_mangle`, attempts to integrate HBM using "HBM_2DFFT_OneAPI_example".  
   - **Status:** Failed due to version mismatch.  

8. **`fft2d_partition_square_8pipe`**  
   - Based on `fft2d_square_8pipe`, uses multi-memory channels with accessors and multi-pipe integration. Updates to CMake file included.  
   - **Status:** In progress.  

9. **`fft2d_nonsquare`**  
   - Based on `fft2d_del_mangle`, supports nonsquare matrices (e.g., 64x128, 64x512). Flexible dimensions (`2^x`).  
   - **Status:** Works.  

---

## Additional Files  

### Python Simulations (`fft_2D_pySim`)  
1. **`fft_2d.ipynb` / `fft_2d_shell.ipynb`**  
   - Visualizes 2D FFT results, magnitude spectrum, compressed images, and 3D grayscale images.  

2. **`n2logn.ipynb`**  
   - Computes algorithm complexity (`N^2`, `logN`, and `N^2*logN`).  

3. **`compare_DFT_FFT_time.ipynb`**  
   - Compares computation times of DFT and FFT to verify FFT's efficiency.  

### 1D FFT Implementation (`fft1d`)  
- Recursive C++ implementation of 1D FFT for understanding recursive functions.  

### References  
1. **`mem_channel`**: Usage of memory channels and buffer accessors with oneAPI.  
2. **`HBM_2DFFT_OneAPI_example`**: Example of using HBM.  

---

## Commands  

1. **`command_git.txt`**  
   - Includes basic Git commands for file management.  

2. **`command_devCloud.txt`**  
   - Steps to use Intel DevCloud.  

---

## Repository Highlights  
- Various optimizations of FFT2D to improve performance and reduce area estimates.  
- Support for nonsquare matrices and HBM integration.  
- Python simulations and references for deeper understanding of FFT concepts.  

Feel free to explore the code and contribute!  
