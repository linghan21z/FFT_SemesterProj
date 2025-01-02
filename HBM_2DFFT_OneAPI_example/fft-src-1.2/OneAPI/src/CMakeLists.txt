set(TARGET_NAME fft2d_oneapi)

set(LIBRARY_SOURCE 
	   ${LIBRARY_SOURCE}
	   ${CMAKE_CURRENT_SOURCE_DIR}/FFT_2d_hbm.cpp
	   )

set (LIBRARY_HEADERS
	   ${LIBRARY_HEADERS}
	   ${CMAKE_CURRENT_SOURCE_DIR}/FFT_2d_hbm.h
	   )

set (HOST_SOURCE
	   ${HOST_SOURCE}
	   ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	   ${CMAKE_CURRENT_SOURCE_DIR}/CompareToFFTW.cpp
	   )

set (HOST_HEADERS
	   ${HOST_HEADERS}
	   ${CMAKE_CURRENT_SOURCE_DIR}/CompareToFFTW.h}
           )	   

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

String(STRIP "${CMAKE_EXE_LINKER_FLAGS}" CMAKE_EXE_LINKER_FLAGS)

IF (DEFINED ENV{BOARD_PACKAGE_DIR})
	MESSAGE(STATUS "Board package directory BOARD_PACKAGE_DIR environment variable set to: $ENV{BOARD_PACKAGE_DIR}")
	SET(_FPGA_BOARD "$ENV{BOARD_PACKAGE_DIR}:p520_hpc_m210h_g3x16")
ELSE()
	MESSAGE(STATUS "Board package directory BOARD_PACKAGE_DIR not set, may cause problems with package in multiple locations")
	SET(_FPGA_BOARD p520_hpc_m210h_g3x16)
ENDIF()

IF (DEFINED ENV{HOST_INCLUDE_PATHS})
	MESSAGE(STATUS "Include path HOST_INCLUDES_PATHS environment variable set to: $ENV{HOST_INCLUDE_PATHS}")
	set (HOST_INCLUDES "$ENV{HOST_INCLUDE_PATHS}")
ELSE()
	set (HOST_INCLUDES .)
ENDIF()

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c)

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga  -Xsffp-reassoc -reuse-exe=../${FPGA_TARGET} -Xsprofile -Xshardware -Xsboard=${_FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
set(REPORT_FLAGS "-fintelfpga -Xshardware -Xsffp-reassoc -Xsprofile -Xsboard=${_FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
set(FFT_HBM_LIB_LINK_FLAGS "-shared" "-fPIC")

set(EMULATOR_COMPILE_FLAGS -fintelfpga -DFPGA_EMULATOR -c)
set(EMULATOR_LINK_FLAGS -fintelfpga)

# fpga emulator
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set(HOST_EMU_OBJ "main_emu.o" "CompareToFFTW_emu.o")
set(DEVICE_EMU_OBJ "FFT_2d_hbm_emu.o")
set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

add_custom_command(OUTPUT ${HOST_EMU_OBJ} 
	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp -o main_emu.o -I${HOST_INCLUDES}
	 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/CompareToFFTW.cpp -o CompareToFFTW_emu.o -I${HOST_INCLUDES} 
         DEPENDS ${HOST_SOURCE}  ${LIBRARY_HEADERS})

add_custom_command(OUTPUT ${DEVICE_EMU_OBJ} 
         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/FFT_2d_hbm.cpp -o FFT_2d_hbm_emu.o
         DEPENDS ${LIBRARY_SOURCE} ${LIBRARY_HEADERS})

add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ} 
         COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ} 
         DEPENDS ${DEVICE_EMU_OBJ})

IF (DEFINED SHARED_LIB)
    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -shared -fPIC -fintelfpga  ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} -lfftw3 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
ELSE()
    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -fintelfpga  ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} -lfftw3 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
ENDIF()

# fpga
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set(HOST_OBJ "main.o" "CompareToFFTW.o")
set(DEVICE_OBJ "FFT_2d_hbm.o")
set(DEVICE_IMAGE_OBJ "dev_image.a")

add_custom_command(OUTPUT ${HOST_OBJ} 
	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp -o main.o -I${HOST_INCLUDES}
	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/CompareToFFTW.cpp -o CompareToFFTW.o -I${HOST_INCLUDES}
        DEPENDS ${HOST_SOURCE} ${LIBRARY_HEADERS})

add_custom_command(OUTPUT ${DEVICE_OBJ} 
	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/FFT_2d_hbm.cpp -o FFT_2d_hbm.o
        DEPENDS ${LIBRARY_SOURCE} ${LIBRARY_HEADERS})

separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ} 
        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link=image ${DEVICE_OBJ} -o ${DEVICE_IMAGE_OBJ} 
        DEPENDS ${DEVICE_OBJ})

IF (DEFINED SHARED_LIB)
    add_custom_command(OUTPUT ${FPGA_TARGET} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} ${FFT_HBM_LIB_LINK_FLAGS} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA__SHARED_TARGET} -lfftw3
                 DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
ELSE()
    add_custom_command(OUTPUT ${FPGA_TARGET} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET} -lfftw3
                 DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
ENDIF()

# report

set(DEVICE_OBJ_FILE "dev_image_report.a")
add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})
separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
add_custom_command(OUTPUT ${DEVICE_OBJ_FILE} 
        COMMENT "Building report files, may take a few minutes. dev_image_report.prj/reports/"
	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${LIBRARY_SOURCE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
        DEPENDS ${LIBRARY_SOURCE})


# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
