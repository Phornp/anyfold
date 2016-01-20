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

BOOST_FIXTURE_TEST_CASE_TEMPLATE(forward_backward, T, Fixtures, T)
{
  std::vector<anyfold::image_stack> input_images(10, T::padded_image_);
  std::vector<anyfold::image_stack> output_images(10, T::padded_image_);

  const float sum = std::accumulate(T::padded_image_.data(),
	                                     T::padded_image_.data() + T::padded_image_.num_elements(),
	                                     0.f);

  //std::vector</*TODO: please implement*/anyfold::complex_image_stack> intermediate_images(10,  
	//										  /*TODO: please implement*/T::padded_complex_image_
	//										  );
 
  
  anyfold::cuda::convolve_forward_fft(input_images, T::padded_image_shape_,
				      intermediate_images);

  
  anyfold::cuda::convolve_backward_fft(intermediate_images, T::padded_image_shape_,
				       output_images);


  for(int i = 0;i<output_images.size();i++){
    float sum_out = std::accumulate(output_images[i].data(),
				    output_images[i].data() + + T::padded_image_shape_.num_elements(),
				    0);
    
    BOOST_CHECK_CLOSE(sum,sum_out,.0001);

  }

}
