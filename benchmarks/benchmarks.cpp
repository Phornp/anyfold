#include <iostream>
#include <string>
#include <vector>

#include "measureTime.hpp"

#include "anyfold.hpp"

template<typename T>
T uniformRandom(T const& min = T(0), T const& max = T(1))
{
	static std::default_random_engine generator;
	std::uniform_real_distribution< T > distribution(min, max);
	return distribution(generator);
}

template<typename T>
T* genRandomArray(size_t size, T min, T max)
{
	T* array = new T[size];
	for (size_t i = 0; i < size; ++i)
	{
		array[i] = uniformRandom(min, max);
	}

	return array;
}

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cerr << "Please provide a convolution method!\n"
		          << "<0> : Buffer\n"
		          << "<1> : Buffer and local memory\n"
		          << "<2> : Images\n"
		          << "<3> : Images and local memory" << std::endl;
		exit(-1);
	}
	int method = std::atoi(argv[1]);

	SimpleTimer timer;
	int rep = 32;
	std::vector<std::vector<int>> k = {{3,9,15}, {9,21,27}, {15,27,33}, {27,39,51}, {39, 51, 21}};
	for(auto ks : k)
	for(int i = 64; i <= 256; i *= 2)
	{
		std::cout << "\nImage size:  " << i
		          << "\nKernel size: " << ks[0] << "x" << ks[1] << "x" << ks[2] << std::endl;
		
		float* image = genRandomArray(i*i*i, 0.0f, 1.0f);
		int imageShape[] = {i, i, i};
		float* kernel = genRandomArray(ks[0]*ks[1]*ks[2], 0.0f, 1.0f);
		int kernelShape[] = {ks[0], ks[1], ks[2]};
		float* output = new float[i*i*i];

		uint64_t time = 0;
		// OpenCL
		for(int r = 0; r < rep; ++r)
		{
			timer.start();
			switch(method)
			{
			case 0: anyfold::opencl::convolve_3dBuffer(image, imageShape,
			                                           kernel, kernelShape, output);
				break;
			case 1: anyfold::opencl::convolve_3dBufferLocalMem(image, imageShape,
			                                                   kernel, kernelShape,
			                                                   output);
				break;
			case 2: anyfold::opencl::convolve_3dImage(image, imageShape,
			                                          kernel, kernelShape, output);
				break;
			case 3: anyfold::opencl::convolve_3dImageLocalMem(image, imageShape,
			                                                  kernel, kernelShape,
			                                                  output);
				break;
			default:
				std::cerr << "Invalid method!" << std::endl;
				exit(-1);
			}
				timer.end();
			time += timer.getNS();
		}
		std::cout << "OpenCL (average): \n" << (float)time / 1000000000.0f / (float)rep << std::endl;

		// CPU
		float* outputCPU = new float[i*i*i];
		timer.start();
		anyfold::cpu::convolve_3d(image, imageShape, kernel, kernelShape, outputCPU);
		timer.end();
		std::cout << "CPU (single core): \n";
		timer.print(true);
		delete[] outputCPU;


		delete[] image;
		delete[] kernel;
		delete[] output;
	}	
	return 0;
}
