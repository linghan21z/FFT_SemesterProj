1. fft2d is the original code. - works
The provided code is to use 1 pipe(PipeIn, PipeOut) to transfer data between kernels
(Fetchkernel to FFT kernel to transfer kernel), and every pipe transfers 8 points of data. 

2. fft2d_squareM1 - doesn't work
<Modified based on fft2d.>
1 pipe, every pipe 32 points of data, then separate in 4 data[] (data0, data1, data2, data3)
to do 4 FFTStep. But it does not work because the the pipe is defined to be able to contain 
8 points of data, can't contain 32 points.

3. fft2d_squareM2 - works
<Modified based on fft2d.>
4 pipes to transfer 32 points of data to do FFT, 
And every pipeIn should pipeIn 8 data points from different rows of the matrix, 
because we know that FFT of a matrix is to do FFT for each row, 
so each row's data should be in the same pipeIn consecutively. 
-- Acceleration on latency.

4. fft2d_del_mangel - works
<Modified based on fft2d.>
Delete everything about mangle in the original fft2d code.

The area estimates in the report reduce. 
And in the Schedule Viewer, the FetchKernel's cycles become less, while FFT and TransposeKernel do not change.

5. fft2d_square_4pipe - works
<Modified based on fft2d_squareM2>
Delete everything about mangle.
-- Acceleration on latency.

6. fft2d_square_8pipe - works
<Modified based on fft2d_del_mangle.>
8 pipes to transfer 64 points of data to do FFT, 
And every pipeIn should pipeIn 8 data points from different rows of the matrix, 
because we know that FFT of a matrix is to do FFT for each row, 
so each row's data should be in the same pipeIn consecutively. 
-- No acceleration on latency.

7. fft2d_square_8pipe_partition - works
<Modified based on fft2d_square_8pipe.>
Duplicate to 2 FFTKernel to separate 8 pipes to be 4 + 4.
Each FFTKernel deals with 4 pipes.
-- No acceleration on latency.

8. fft2d_hbm
<Modified based on fft2d_del_mangle.>


