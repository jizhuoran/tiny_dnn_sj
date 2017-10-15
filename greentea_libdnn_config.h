#ifndef GREENTEA_LIBDNN_CONFIG_HPP_
#define GREENTEA_LIBDNN_CONFIG_HPP_

/* Version */
#define GREENTEA_VERSION ""

/* Sources directory */
#define SOURCE_FOLDER "/home/zrji/embedded_dnn/libdnn"

/* Binaries directory */
#define BINARY_FOLDER "/home/zrji/embedded_dnn/libdnn"

/* 64 bit indexing */
/* #undef USE_INDEX_64 */

/* NVIDIA Cuda */
/* #undef HAVE_CUDA */
/* #undef USE_CUDA */

/* OpenCl kernels */
#define HAVE_OPENCL
#define USE_OPENCL
#define VIENNACL_WITH_OPENCL

#define CMAKE_SOURCE_DIR "src/"
#define CMAKE_EXT ""

#endif  // GREENTEA_LIBDNN_CONFIG_HPP_
