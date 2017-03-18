#ifndef Utils_cuh
#define Utils_cuh

//#include "helper_cuda.h"

#include <cuda_runtime.h>
#include <math_functions.h>
#include <iostream>

// add
__host__ __device__ inline float3 operator + (const float3 & v, float s)
{
	return make_float3(v.x + s, v.y + s, v.z + s);
}

__host__ __device__ inline float3 operator + (float s, const float3 & v)
{
	return v + s;
}

__host__ __device__ inline float3 operator + (const float3 & v0, const float3 & v1)
{
	return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}

// negate
__host__ __device__ inline float3 operator - (const float3 & v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

// multiple
__host__ __device__ inline float3 operator * (const float3 & v, float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline float3 operator * (float s, const float3 & v)
{
	return v * s;
}

__host__ __device__ inline float3 operator * (const float3 & v0, const float3 & v1)
{
	return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}

__host__ __device__ inline float squared_length(const float3 & v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline float length(const float3 & v)
{
	return sqrtf(squared_length(v));
}

__host__ __device__ inline unsigned int range_hash(unsigned int seed, unsigned int range)
{
	return (seed + 68857) % range;
	//unsigned int hashCode = seed ^ (seed >> 20) ^ (seed >> 12);
	//return (seed ^ (seed >> 7) ^ (seed >> 4)) % range;
}


//#ifdef __DRIVER_TYPES_H__
//#ifndef DEVICE_RESET
//#define DEVICE_RESET cudaDeviceReset();
//#endif
//
//template< typename T >
//void check(T result, char const *const func, const char *const file, int const line)
//{
//	if (result)
//	{
//		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
//			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
//		DEVICE_RESET
//			// Make sure we call CUDA Device Reset before exiting
//			exit(EXIT_FAILURE);
//	}
//}
//
//#ifdef __DRIVER_TYPES_H__
//// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
//#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
//
//// This will output the proper error string when calling cudaGetLastError
//#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
//
//inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
//{
//	cudaError_t err = cudaGetLastError();
//
//	if (cudaSuccess != err)
//	{
//		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
//			file, line, errorMessage, (int)err, cudaGetErrorString(err));
//		DEVICE_RESET
//			exit(EXIT_FAILURE);
//	}
//}
//#endif
//#endif

#endif
