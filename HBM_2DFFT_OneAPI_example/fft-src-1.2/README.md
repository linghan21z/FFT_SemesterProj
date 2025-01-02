*******************************************************************************
Overview
*******************************************************************************

Version 1.2

This directory contains three different implementations of a 2DFFT.  The highlight is
the OneAPI implementation which is a "port" of the earlier OpenCL implementation.
The CUDA implementation provides a comparison benchmark. The CUDA
implementation uses bundled FFT libraries, not our OpenCL/OneAPI algorithm.

The three implementations are completely independent with separate build
instructions.

	CUDA/INSTALL
	OneAPI/INSTALL
	OpenCL/INSTALL

The build scripts have been tested on Centos 8 and are not compatible with Windows.

NOTE: The OneAPI and the OpenCL build process requires a "board support package" (BSP)
for a target with an Intel FPGA featuring HBM2 memory. BittWare has only tested the
build process using a BSP for BittWare's 520N-MX card. 

NOTE: Please ensure fpga builds are performed on systems meeting the memory
requirements documented in the OneAPI documentation. 64GB is recommended for
these Stratix 10 based designs. 

*******************************************************************************
Algorithm
*******************************************************************************

This example performs a 2D floating point complex to complex FFT using 16 
parallel 1D FFTs. Each 1D FFT is fully pipelined, consuming a complex pair
and generating a complex pair each clock cycle. 

Two passes of main pipeline are performed, representing the row and column 
computation of the 2D FFT. Each pass transposes the output inline with the 
calculation. 

In order to maintain efficient HBM memory access, a burst buffer caches 32 
complex pairs, allowing the HBMs to be updated as an efficient burst of sixteen 256
bit writes before changing a row address.

For more information see...

   https://www.bittware.com/resources/hbm2-2d-fft-oneapi

In contrast, the CUDA implementation uses NVIDIA's cufft. We do not know how
cufft is implemented.


*******************************************************************************
Version History
*******************************************************************************

May 18th 2021 : Version 1.2

* Updates to documentation and scripts to target 20.4 Quartus based 520N-MX OpenCL BSP
* This BSP also is now based on a PCIe Gen3x16 interface.
* Fix to difference checks to ignore small numbers in output. 

Mar 18th 2021 : Version 1.1

* Improved host buffer logic for OneAPI implementation.
* Added -reuse flag to OneAPI makefile.
* New python script to display input and output images.

Jan 25th 2021 : Initial release


*******************************************************************************
Known Issue
*******************************************************************************

One known OneAPI issue is the limitation in number of kernel attributes supported
in earlier versions of OneAPI. Although that limit was 128 arguments, OneAPI uses four
arguments per HBM access, which limits the 2DFFT design to a single pipeline. Our
original OpenCL performance results and code placed two pipelines in the FPGA design.

Intel tells us this is fixed in the current OneAPI release.  However, we did not try to
place two pipelines into the OneAPI design in this 1.2 release of the BittWare FFT.  