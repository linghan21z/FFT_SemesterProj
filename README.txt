1. fft2d is the original code. - works
The provided code is to use 1 pipe(PipeIn, PipeOut) to transfer data between kernels
(Fetchkernel to FFT kernel to Transposekernel), and every pipe transfers 8 points of data. 

2. fft2d_squareM1 - doesn't work
<Modified based on fft2d.>
1 pipe, every pipe 32 points of data, then separate in 4 data[] (arrays) (data0, data1, data2, data3)
to do 4 FFTStep. But it does not work because the the pipe is defined to be able to contain 
8 points of data, can't contain 32 points. (But it's possibe to modify the data capacity of pipe.)

3. fft2d_squareM2 - works
<Modified based on fft2d.>
4 pipes to transfer 32 points of data to do FFT, 
And every pipeIn should pipeIn 8 data points from different rows of the matrix, 
because we know that FFT of a matrix is to do FFT for each row, 
so each row's data should be in each pipeIn consecutively. 
-- Acceleration on latency.

4. fft2d_del_mangel - works
<Modified based on fft2d.>
Delete everything about mangle in the original fft2d code.
-- Less area estimates, because of deleted "mangle". 
And in the Schedule Viewer, the FetchKernel's cycles become less, while FFT and TransposeKernel do not change.


5. fft2d_square_4pipe - works ----------------------------
<Modified based on fft2d_squareM2>
Delete everything about mangle.
-- Acceleration on latency, compared with 1 pipe.

6. fft2d_square_8pipe - works -------------------------------
<Modified based on fft2d_del_mangle.>
8 pipes to transfer 64 points of data to do FFT, 
And every pipeIn should pipeIn 8 data points from different rows of the matrix, 
because we know that FFT of a matrix is to do FFT for each row, 
so each row's data should be in each pipeIn consecutively. 
-- No acceleration on latency.

7. fft2d_square_8pipe_partition - works
<Modified based on fft2d_square_8pipe.>
Duplicate to 2 FFTKernel to separate 8 pipes to be 4 + 4.
Each FFTKernel deals with 4 pipes.
-- No acceleration on latency.

8. fft2d_hbm/fft2d_hbm_usmHost
<Modified based on fft2d_del_mangle.>
Try to use hbm, referring to "HBM_2DFFT_OneAPI_example".

9. fft2d_partition_square_8pipe  -------------------------------
<Modified based on fft2d_square_8pipe.>
Multi-memory-channels with accessors, combined with multi-pipe.
Also add things to the CMake file.

10. fft2d_nonsquare - works  -------------------------------
<Modified based on fft2d_del_mangle.>
To deal with nonsquare matrix's 2dFFT with different kN_row and kN_column. (eg. 64*128, 64*512...)
The number of rows and columns could be flexible by 2^x.
When modifying the code, pay attention to the WR_index in "TransposeKernel" 
and reference result computation in main()(also the transpose part).

----------------------------------------
Other code files:
1. fft_2D_pySim
(1) fft_2d.ipynb 
- using python to do 2d-fft, to see the effect, magnitude spectum, 
compressed images and 3D grayscale image.
fft_2d_shell.ipynb 
- same as fft_2d.ipynb.

(2) n2logn.ipynb 
- Compute N^2, logN, and N^2*logN to see the algorithm complexity.

(3) compare_DFT_FFT_time.ipynb 
- Measure DFT and FFT computation time, verify that FFT is faster than DFT.

2. fft1d
Implementing 1D Fast Fourier Transform in C++, recursively.
Understanding the implementation of recursive functions

3. mem_channel
Reference of using mem_channel and buffer accessors, from oneAPI.

4. HBM_2DFFT_OneAPI_example
Reference of using HBM.

5. command_git.txt
I learned how to use git, and files management. There are some commands.

6. command_devCloud.txt
I learned how to use devCloud.



