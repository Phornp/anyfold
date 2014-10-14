#ifndef _CUDA_KERNELS_CUH_
#define _CUDA_KERNELS_CUH_
#include <limits>
#include <cassert>
#include <cmath>
#include "cuda_runtime.h"

template <typename TransferT, typename TupleT>
__device__ void x_tiled_load_of_image_segment(const TransferT* _src, 
                                              TransferT* _dest,//taking for granted that _dest can hold 
                                              const uint3& _image_dims,
                                              const TupleT& _extent,
                                              const TupleT& _origin){
  
  int num_tiles_x = (_extent.x + blockDim.x - 1)/blockDim.x;
  
  uint3 src_coords;
  uint3 dest_coords;
  TransferT src_value = 0;
  for(int x_tile_index = 0;x_tile_index<num_tiles_x;x_tile_index++){
    src_coords.x = _origin.x + threadIdx.x + x_tile_index*blockDim.x;
    src_coords.y = _origin.y + threadIdx.y;
    src_coords.z = _origin.z + threadIdx.z;

    dest_coords.x = threadIdx.x + x_tile_index*blockDim.x;
    dest_coords.y = threadIdx.y;
    dest_coords.z = threadIdx.z;
        
    if(src_coords.x < _image_dims.x && src_coords.y < _image_dims.y && src_coords.z < _image_dims.z)
      src_value = _src[src_coords.z*(_image_dims.y*_image_dims.x) + src_coords.y*(_image_dims.x) + src_coords.x];
    else
      src_value = 0;

    if(dest_coords.x < _extent.x)
      _dest[dest_coords.z*(_extent.y*_extent.x) + dest_coords.y*(_extent.x) + dest_coords.x] = src_value;

    __syncthreads();
  }
  
}

//FIXME: kernel has still some bugs, try with kernel_axis=3 image_axis=100 
//FIXME: but the code works for some combinations kernel_axis=* image_axis=512
//assumes the kernel to be of equal extent in all dimensions
template <unsigned _kernel_line_size, typename TransferT>
__global__ void static_convolve(const TransferT* _image,
				const TransferT*  _kernel,
				TransferT* _output,
				uint3 _image_dims){

  // const unsigned int kernel_size = _kernel_line_size*_kernel_line_size*_kernel_line_size;
  const unsigned int image_size = _image_dims.x * _image_dims.y * _image_dims.z;
  const unsigned int image_line_size = (blockDim.x + _kernel_line_size - 1);
  const unsigned int kernel_block_size = _kernel_line_size;
  const unsigned int half_kernel_dim = _kernel_line_size/2;

  extern __shared__ TransferT shmem_data[];
  
  TransferT* image_shm = &shmem_data[0];

  //pixel ID incl padding
  const int pixel_x = (int)blockIdx.x*blockDim.x + threadIdx.x + (int)(half_kernel_dim);
  const int pixel_y = (int)blockIdx.y*blockDim.y + threadIdx.y + (int)(half_kernel_dim);
  const int pixel_z = (int)blockIdx.z*blockDim.z + threadIdx.z + (int)(half_kernel_dim);
  const unsigned int pixel_index = pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x) + pixel_x;

  //top left pixel coordinates
  const int first_block_pixel_x = pixel_x  - (int)threadIdx.x ;
  const int first_block_pixel_y = pixel_y  - (int)threadIdx.y ;
  const int first_block_pixel_z = pixel_z  - (int)threadIdx.z ;
   

  // int image_block_start_index_z = 0;
  int kernel_block_start_index_z = 0;

  // int image_block_start_index_y = 0;
  int kernel_block_start_index_y = 0;
     
  // int image_block_start_index = 0;
  int kernel_block_start_index = 0;

  TransferT value = 0.f;
  TransferT image_pixel = 0.f;
  TransferT kernel_pixel = 0.f;
  TransferT kernel_data[_kernel_line_size];
  TransferT image_data[_kernel_line_size];

  uint3 segment_extent;
  segment_extent.x = image_line_size;//blockDim.x + 2*half_kernel_dim
  segment_extent.y = blockDim.y;
  segment_extent.z = blockDim.z;
  uint3 segment_origin;

  //loop through x-y slices
  for(int offset_z = -int(half_kernel_dim);offset_z<=int(half_kernel_dim);++offset_z){


    kernel_block_start_index_z = (offset_z + int(half_kernel_dim))*int(_kernel_line_size*_kernel_line_size) ;

    //
    //////////////////////////////// PRELOAD IMAGE LINE/BLOCK TO SHMEM FOR FIRST ITERATION /////////////////////////////////////////////
    segment_origin.x = first_block_pixel_x - int(half_kernel_dim);
    segment_origin.y = first_block_pixel_y - int(half_kernel_dim);
    segment_origin.z = first_block_pixel_z + offset_z;//

    x_tiled_load_of_image_segment(_image,
				  image_shm,
				  _image_dims,
				  segment_extent,
				  segment_origin);
    

    //loop through lines in slice
    for(int offset_y = -int(half_kernel_dim);offset_y<=int(half_kernel_dim);offset_y+=1){
	
      kernel_block_start_index_y = (offset_y + int(half_kernel_dim))*int(_kernel_line_size) ;

      kernel_block_start_index = kernel_block_start_index_z + kernel_block_start_index_y;

      // 
      //////////////////////////////// LOAD KERNEL LINE/BLOCK TO REGISTERS ////////////////////////////////////////////
      //
      #pragma unroll
      for(unsigned int k_index=kernel_block_start_index;k_index<kernel_block_start_index+kernel_block_size;k_index++){
	kernel_data[k_index-kernel_block_start_index] = _kernel[k_index];
      }

      // 
      //////////////////////////////// COPY SHMEM PIXELS TO REGISTERS ////////////////////////////////////////////
      //
      #pragma unroll
      for(int offset_x = -int(half_kernel_dim);offset_x<=int(half_kernel_dim);++offset_x){
	image_data[offset_x+int(half_kernel_dim)] = image_shm[threadIdx.x + offset_x + int(half_kernel_dim) + (threadIdx.y*image_line_size)];
      }
      	
      // 
      //////////////////////////////// PERFORM DOT PRODUCT ////////////////////////////////////////////
      //
      #pragma unroll
      for(int index_x = 0;index_x<_kernel_line_size;++index_x){
	
	image_pixel = image_data[index_x];

	kernel_pixel = kernel_data[index_x];
		
	value += image_pixel*kernel_pixel;
      }//x loop

      //
      //////////////////////////////// LOAD IMAGE LINE/BLOCK TO SHMEM /////////////////////////////////////////////
      //
      segment_origin.y = first_block_pixel_y + offset_y + 1;//

      x_tiled_load_of_image_segment(_image,
				    image_shm,
				    _image_dims,
				    segment_extent,
				    segment_origin);
      
    }//y loop
  }//z loop


  if(pixel_index<image_size){
    _output[pixel_index] = value;
  }
}

#endif
