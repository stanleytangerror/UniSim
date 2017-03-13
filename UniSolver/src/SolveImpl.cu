#include "SolveImpl.h"
#include "Solver.h"
#include "Utils.cuh"
#include "GraphColoring.cuh"

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <device_functions.h>

namespace uni
{
	__device__ float eps = 1e-20f;

	int threadsPerBlock = 512;


	/***********************************************************************************/

	template <typename AdjItemType>
	__global__ void adjTable_k(DistCons * cons, AdjItemType * adj_cons, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;

		//is_colored[cid] = 0;

		//palettes[cid].valid_size = palettes[cid].max_size * 0.94f;
		//for (int i = 0; i < palettes[cid].valid_size; ++i)
		//	palettes[cid].colors[i] = 1;

		int pid0 = cons[cid].pid.x;
		int pid1 = cons[cid].pid.y;

		adj_cons[cid].valid_size = 0;
		for (int i = 0; i < con_size; ++i)
		{
			if (cid != i &&
				(cons[i].pid.x == pid0 || cons[i].pid.x == pid1 || cons[i].pid.y == pid0 || cons[i].pid.y == pid1))
			{
				adj_cons[cid].adjs[adj_cons[cid].valid_size] = i;
				adj_cons[cid].valid_size += 1;
			}
		}
	}

	__global__ void freeRun_Gauss_k(float3 * x, float3 * p, float3 * v, float * inv_m, float time_step, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;

		float3 force = { 0.0f, -0.010f, 0.0f };

		float3 offset = 0.5f * inv_m[pid] * time_step * force;
		v[pid] = v[pid] + offset;
		p[pid] = x[pid] + v[pid] * time_step;
	}

	__global__ void projectConstraint_Gauss_k(float3 * x, float3 * p, float3 * v, float * inv_m, DistCons * cons, int * colors, unsigned int con_size, int gid)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (colors[cid] != gid) return;

		int pid0 = cons[cid].pid.x;
		int pid1 = cons[cid].pid.y;
		float d = cons[cid].d;

		float inv_m0 = inv_m[pid0];
		float inv_m1 = inv_m[pid1];

		if (inv_m0 + inv_m1 < eps) return;

		float3 p0 = p[pid0];
		float3 p1 = p[pid1];

		float dist = length(p0 + (-p1));
		if (-eps < dist && dist < eps) return;
		float delta_d = dist - d;

		//float3 tmp = (delta_d / (inv_m0 + inv_m1)) * (p0 + (-p1));
		float3 tmp = delta_d / ((inv_m0 + inv_m1) * dist) * (p0 + (-p1));
		//float3 tmp = (1.0f / (inv_m0 + inv_m1) * delta_d) * (p0 + (-p1));
		float3 d_p0 = -inv_m0 * tmp;
		float3 d_p1 = inv_m1 * tmp;

		p[pid0] = p[pid0] + d_p0;
		p[pid1] = p[pid1] + d_p1;
	}

	__global__ void updateState_Gauss_k(float3 * x, float3 * p, float3 * v, float * inv_m, float time_step, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;

		v[pid] = (p[pid] + (-x[pid])) * (1.0f / time_step);
		x[pid] = p[pid];
	}

	template <int MaxDegree>
	void callGraphColoring_Gauss(SolverData * data, int * cons_colors, unsigned int cons_size)
	{
		static AdjItem<MaxDegree> * adj_table = nullptr;
		if (adj_table == nullptr)
			cudaMalloc((void **)&adj_table, cons_size * sizeof(AdjItem<MaxDegree>));

		int threadsPerBlock = 1024;

		dim3 con_blocks((cons_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 con_threads(threadsPerBlock);

		adjTable_k<AdjItem<MaxDegree>> << <con_blocks, con_threads >> > (data->cons, adj_table, cons_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		graph_coloring<MaxDegree>(adj_table, cons_colors, cons_size);
	}

	void solve_Gauss(SolverData * data, unsigned int p_size, unsigned int cons_size, float time_step, int iter_cnt)
	{
		static int * colors = nullptr;
		if (colors == nullptr)
			cudaMalloc((void **)&colors, cons_size * sizeof(int));

#ifdef PROFILE_CUDA
		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		checkCudaErrors(cudaEventRecord(start, 0));
#endif

		int threadsPerBlock = 1024;

		dim3 p_blocks((p_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 p_threads(threadsPerBlock);
		
		dim3 con_blocks((cons_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 con_threads(threadsPerBlock);

		freeRun_Gauss_k <<<p_blocks, p_threads>>> (data->x, data->p, data->v, data->inv_m, time_step, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		callGraphColoring_Gauss<16>(data, colors, cons_size);

		for (int i = 0; i < iter_cnt; ++i)
		{
			for (int gid = 0; gid < 16; ++gid)
			{
				projectConstraint_Gauss_k << <con_blocks, con_threads >> >(data->x, data->p, data->v, data->inv_m, data->cons, colors, cons_size, gid);
				getLastCudaError("Kernel execution failed");
				checkCudaErrors(cudaDeviceSynchronize());
			}
		}

		updateState_Gauss_k <<<p_blocks, p_threads>>>(data->x, data->p, data->v, data->inv_m, time_step, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

#ifdef PROFILE_CUDA
		checkCudaErrors(cudaEventRecord(stop, 0));
		
		checkCudaErrors(cudaEventSynchronize(stop));
		float elapse_time;
		checkCudaErrors(cudaEventElapsedTime(&elapse_time, start, stop));
		std::cout << "solve on GPU time " << elapse_time << " ms" << std::endl;
		
		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
#endif
	}


	/***********************************************************************************/
	
	__global__ void freeRun_Jacobi_k(float3 * x, float3 * p, float3 * v, float * inv_m, float time_step, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockDim.y * blockIdx.x;
		if (pid >= p_size) return;

		float3 force = { 0.0f, -0.10f, 0.0f };

		float3 offset = inv_m[pid] * time_step * force;
		v[pid] = v[pid] + offset;
		p[pid] = x[pid] + v[pid] * time_step;
	}

	__global__ void projectConstraint_Jacobi_k(float3 * x, float3 * p, float3 * delta_p, int * p_con_cnt, float3 * v, float * inv_m, DistCons * cons, unsigned int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;

		int pid0 = cons[cid].pid.x;
		int pid1 = cons[cid].pid.y;
		float d = cons[cid].d;

		float inv_m0 = inv_m[pid0];
		float inv_m1 = inv_m[pid1];

		if (inv_m0 < eps && inv_m1 < eps) return;

		float3 p0 = p[pid0];
		float3 p1 = p[pid1];

		float dist = length(p0 + (-p1));
		if (-eps < dist && dist < eps) return;
		float delta_d = dist - d;

		float3 tmp = 1.0f / (inv_m0 + inv_m1) * delta_d / dist * (p0 + (-p1));
		float3 d_p0 = -inv_m0 * tmp;
		float3 d_p1 = inv_m1 * tmp;

		delta_p[pid0] = delta_p[pid0] + d_p0;
		delta_p[pid1] = delta_p[pid1] + d_p1;

		p_con_cnt[pid0] += 1;
		p_con_cnt[pid1] += 1;
	}

	__global__ void updateState_Jacobi_k(float3 * x, float3 * p, float3 * delta_p, int * p_con_cnt, float3 * v, float * inv_m, float time_step, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;
		
		float w = 1.3f;
		int cnt = p_con_cnt[pid];
		if (cnt <= 0) return;
		p[pid] = p[pid] + delta_p[pid] * (1.0f / cnt) * w;
		v[pid] = (p[pid] + (-x[pid])) * (1.0f / time_step);
		x[pid] = p[pid];
	}

	void solve_Jacobi(SolverData * data, unsigned int p_size, unsigned int cons_size, float time_step, int iter_cnt)
	{
		static float3 * delta_p = nullptr;
		static int * p_con_cnt = nullptr;

		if (delta_p == nullptr)
			checkCudaErrors(cudaMalloc((void**)&delta_p, p_size * sizeof(float3)));
		if (p_con_cnt == nullptr)
			checkCudaErrors(cudaMalloc((void**)&p_con_cnt, p_size * sizeof(int)));

		checkCudaErrors(cudaMemset(delta_p, 0, p_size * sizeof(float3)));
		checkCudaErrors(cudaMemset(p_con_cnt, 0, p_size * sizeof(int)));

		dim3 p_blocks((p_size + 1023) / 1024);
		dim3 p_threads(1024);

		dim3 con_blocks((cons_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 con_threads(threadsPerBlock);

		freeRun_Jacobi_k << <con_blocks, p_threads >> >(data->x, data->p, data->v, data->inv_m, time_step, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		for (int i = 0; i < iter_cnt; ++i)
		{
			projectConstraint_Jacobi_k << <con_blocks, con_threads >> >(data->x, data->p, delta_p, p_con_cnt, data->v, data->inv_m, data->cons, cons_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());
		}

		updateState_Jacobi_k << <con_blocks, p_threads >> >(data->x, data->p, delta_p, p_con_cnt, data->v, data->inv_m, time_step, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());
	}


}

