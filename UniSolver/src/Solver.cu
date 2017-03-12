#include "Solver.h"
#include "SolveImpl.h"
#include "Utils.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace uni
{
	void initial_device()
	{
		int count = 0;
		cudaDeviceProp prop;
		cudaGetDeviceCount(&count);
		std::cout << "Device count " << count << std::endl;
		for (int i = 0; i < count; ++i)
		{
			cudaGetDeviceProperties(&prop, i);
			std::cout << "max threads per block " << prop.maxThreadsPerBlock << std::endl;
			std::cout << "max threads dim " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
			std::cout << "max grid size " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
		}

		checkCudaErrors(cudaSetDevice(0));
	}

	void alloc_cuda_memory(SolverData * data, unsigned int p_size, unsigned int constraint_size)
	{
		checkCudaErrors(cudaMalloc((void**)&data->p, p_size * sizeof(float3)));
		checkCudaErrors(cudaMalloc((void**)&data->x, p_size * sizeof(float3)));
		checkCudaErrors(cudaMalloc((void**)&data->v, p_size * sizeof(float3)));
		checkCudaErrors(cudaMalloc((void**)&data->inv_m, p_size * sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&data->cons, constraint_size * sizeof(DistCons)));
	}

	void set_positions(SolverData * data, float3 * host_positions, unsigned int p_size)
	{
		checkCudaErrors(cudaMemcpy(data->p, host_positions, p_size * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data->x, host_positions, p_size * sizeof(float3), cudaMemcpyHostToDevice));
	}

	void set_velocities(SolverData * data, float3 * host_velocities, unsigned int p_size)
	{
		checkCudaErrors(cudaMemcpy(data->v, host_velocities, p_size * sizeof(float3), cudaMemcpyHostToDevice));
	}

	void set_inv_masses(SolverData * data, float * host_inv_m, unsigned int p_size)
	{
		checkCudaErrors(cudaMemcpy(data->inv_m, host_inv_m, p_size * sizeof(float), cudaMemcpyHostToDevice));
	}

	void set_constraints(SolverData * data, DistCons * host_constraints, unsigned int constraint_size)
	{
		checkCudaErrors(cudaMemcpy(data->cons, host_constraints, constraint_size * sizeof(DistCons), cudaMemcpyHostToDevice));
	}

	void solve(SolverData * data, unsigned int p_size, unsigned int cons_size, float time_step, int iter_cnt)
	{
		solve_Gauss(data, p_size, cons_size, time_step, iter_cnt);
	}

	void get_positions(SolverData * data, float3 * host_positions, unsigned int p_size)
	{
		checkCudaErrors(cudaMemcpy(host_positions, data->x, p_size * sizeof(float3), cudaMemcpyDeviceToHost));
	}


	void free_cuda_memory(SolverData * data)
	{
		if (data == nullptr) return;

		checkCudaErrors(cudaFree(data->p));
		checkCudaErrors(cudaFree(data->x));
		checkCudaErrors(cudaFree(data->v));
		checkCudaErrors(cudaFree(data->inv_m));
		checkCudaErrors(cudaFree(data->cons));
	}


	void reset_device()
	{
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		checkCudaErrors(cudaDeviceReset());
	}

}
