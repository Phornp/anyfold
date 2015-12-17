#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CUDA_MULTIPLE_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include <boost/mpl/vector.hpp>
#include "test_fixtures_asym.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <algorithm>
#include "anyfold.hpp"

#include "test_algorithms.hpp"
#include "image_stack_utils.h"

static anyfold::storage local_order = boost::c_storage_order();

typedef anyfold::convolutionFixture3DAsym<5,9,13,64> fixture_3D_64_5_9_13;
typedef anyfold::convolutionFixture3DAsym<21,3,11,64> fixture_3D_64_21_3_11;

typedef boost::mpl::vector<
	  fixture_3D_64_5_9_13
	, fixture_3D_64_21_3_11
	> Fixtures;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(trivial_convolveBuffer, T, Fixtures, T)
{
  std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
  std::vector<anyfold::image_stack> output_images(10);
  anyfold::image_stack expected(T::image_);
	
  std::vector<float> kernel(T::kernel_size_,0.f);

  anyfold::cuda::convolve_3dBuffers(input_images, T::padded_image_shape_,
				    kernel,kernel_dims_,
				    output_images);

  for(int i = 0;i<output_images.size();i++){
    float sum = std::accumulate(output_images[i].data(),
				output_images[i].data() + T::padded_output_.num_elements(),0.f);
	  
    BOOST_CHECK_CLOSE(sum, 0.f, .00001);
  }

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(identity_convolveBuffer, T, Fixtures, T)
{
  
  std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
  std::vector<anyfold::image_stack> output_images(10);
	
  const float sum_original = std::accumulate(T::padded_image_.data(),
					     T::padded_image_.data() + T::padded_image_.num_elements(),
					     0.f);

  anyfold::cuda::convolve_3dBuffers(input_images, T::padded_image_shape_,
				    T::identity_kernel_.data(),kernel_dims_,
				    output_images);

  for(int i = 0;i<output_images.size();i++){
	
    float sum = std::accumulate(output_images[i].data(),
				output_images[i].data() +T::padded_output_.num_elements(),
				0.f);
    BOOST_CHECK_CLOSE(sum, sum_original, .00001);
	  
    float l2norm = anyfold::l2norm(input_images [i].data(),
				   output_images[i].data(),
				   output_images[i].num_elements());
    BOOST_CHECK_CLOSE(l2norm, 0, .00001);
	
  }
}

// BOOST_FIXTURE_TEST_CASE_TEMPLATE(horizontal_convolveBuffer, T, Fixtures, T)
// {
//   std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
//   std::vector<anyfold::image_stack> output_images(10);
	
//   const float sum_original = std::accumulate(T::padded_image_.data(),
// 					     T::padded_image_.data() + T::padded_image_.num_elements(),
// 					     0.f);

//   anyfold::cuda::convolve_3dBuffers(input_images, T::padded_image_shape_,
// 				    T::horizontal_kernel_.data(),kernel_dims_,
// 				    output_images);

//   for(int i = 0;i<output_images.size();i++){
	
//     float sum = std::accumulate(output_images[i].data(),
// 				output_images[i].data() +T::padded_output_.num_elements(),
// 				0.f);
//     BOOST_CHECK_CLOSE(sum, sum_original, .00001);
	  
//     float l2norm = anyfold::l2norm(input_images [i].data(),
// 				   output_images[i].data(),
// 				   output_images[i].num_elements());
//     BOOST_CHECK_CLOSE(l2norm, 0, .00001);
	
//   }
// }

// BOOST_FIXTURE_TEST_CASE_TEMPLATE(vertical_convolveBuffer, T, Fixtures, T)
// {
//   std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
//   std::vector<anyfold::image_stack> output_images(10);
	
//   const float sum_original = std::accumulate(T::padded_image_.data(),
// 					     T::padded_image_.data() + T::padded_image_.num_elements(),
// 					     0.f);

//   anyfold::cuda::convolve_3dBuffers(input_images, T::padded_image_shape_,
// 				    T::vertical_kernel_.data(),kernel_dims_,
// 				    output_images);

//   for(int i = 0;i<output_images.size();i++){
	
//     float sum = std::accumulate(output_images[i].data(),
// 				output_images[i].data() +T::padded_output_.num_elements(),
// 				0.f);
//     BOOST_CHECK_CLOSE(sum, sum_original, .00001);
	  
//     float l2norm = anyfold::l2norm(input_images [i].data(),
// 				   output_images[i].data(),
// 				   output_images[i].num_elements());
//     BOOST_CHECK_CLOSE(l2norm, 0, .00001);
	
//   }
// }


// BOOST_FIXTURE_TEST_CASE_TEMPLATE(depth_convolveBuffer, T, Fixtures, T)
// {
//   std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
//   std::vector<anyfold::image_stack> output_images(10);
	
//   const float sum_original = std::accumulate(T::padded_image_.data(),
// 					     T::padded_image_.data() + T::padded_image_.num_elements(),
// 					     0.f);

//   anyfold::cuda::convolve_3dBuffers(input_images, T::padded_image_shape_,
// 				    T::depth_kernel_.data(),kernel_dims_,
// 				    output_images);

//   for(int i = 0;i<output_images.size();i++){
	
//     float sum = std::accumulate(output_images[i].data(),
// 				output_images[i].data() +T::padded_output_.num_elements(),
// 				0.f);
//     BOOST_CHECK_CLOSE(sum, sum_original, .00001);
	  
//     float l2norm = anyfold::l2norm(input_images [i].data(),
// 				   output_images[i].data(),
// 				   output_images[i].num_elements());
//     BOOST_CHECK_CLOSE(l2norm, 0, .00001);
	
//   }
// }

