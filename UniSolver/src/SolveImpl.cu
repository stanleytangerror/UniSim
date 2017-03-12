#include "SolveImpl.h"
#include "Solver.h"
#include "Utils.cuh"

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <device_functions.h>

//#define DEBUG_CUDA
//#define DEBUG_GRAPH_COLORING
//#define DEBUG_GRAPH_COLORING_REDUCE
//#define PROFILE_CUDA

namespace uni
{
	__device__ float eps = 1e-20f;

	int threadsPerBlock = 512;

	__global__ void reduce_k(int * in_idata, int * out_data, int size)
	{
		extern __shared__ int sdata[];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < size) sdata[tid] = in_idata[i];
		else sdata[tid] = 0;
		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
		{
			if (tid < s) sdata[tid] += sdata[tid + s];
			__syncthreads();
		}

		if (tid < 32)
		{
			sdata[tid] += sdata[tid + 32];
			sdata[tid] += sdata[tid + 16];
			sdata[tid] += sdata[tid + 8];
			sdata[tid] += sdata[tid + 4];
			sdata[tid] += sdata[tid + 2];
			sdata[tid] += sdata[tid + 1];
		}

		if (tid == 0) out_data[blockIdx.x] = sdata[0];
	}

	/***********************************************************************************/

	template <int Size>
	struct Palette
	{
		static const int max_size = Size;
		int valid_size;
		int colors[max_size];
	};

	template <int Size>
	struct AdjItem
	{
		static const int max_size = Size;
		int valid_size;
		int adjs[max_size];
	};

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

	template <typename AdjItemType, typename PaletteType>
	__global__ void preColoring_Gauss_k(DistCons * cons, int * is_colored, PaletteType * palettes, AdjItemType * adj_cons, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;

		is_colored[cid] = 0;

		palettes[cid].valid_size = palettes[cid].max_size * 0.94f;
		for (int i = 0; i < palettes[cid].valid_size; ++i)
			palettes[cid].colors[i] = 1;

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

	template <typename AdjItemType, typename PaletteType>
	__global__ void tentativeColoring_Gauss_k_debug(DistCons * cons, int * is_colored, PaletteType * palettes, int * colors, int iter_no, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (is_colored[cid] == 1) return;
		unsigned int no = cid % int(palettes[cid].max_size * 0.94f);
		int cnt = 0;
		for (int i = 0; ; i = (i + 1) % palettes[cid].valid_size)
		{
			if (palettes[cid].colors[i] == 1)
			{
				cnt += 1;
				if (cnt >= no)
				{
					colors[cid] = i;
					break;
				}
			}
		}
	}

	template <typename AdjItemType, typename PaletteType>
	__global__ void tentativeColoring_Gauss_k(DistCons * cons, int * is_colored, PaletteType * palettes, int * colors, int iter_no, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (is_colored[cid] == 1) return;
		
		//unsigned int no = (cid + iter_no + 68857) % int(palettes[cid].max_size * 0.94f);
		unsigned int seed = range_rand(97, cid + iter_no);
		unsigned int no = range_rand(palettes[cid].max_size * 0.94f, seed);
		int cnt = 0;
		for (int i = 0; ; i = (i + 1) % palettes[cid].valid_size)
		{
			if (palettes[cid].colors[i] == 1)
			{
				cnt += 1;
				if (cnt >= no)
				{
					colors[cid] = i;
					break;
				}
			}
		}
	}

	template <typename AdjItemType, typename PaletteType>
	__global__ void conflictResolution_Gauss_k(DistCons * cons, int * is_colored, PaletteType * palettes, AdjItemType * adj_cons, int * colors, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (is_colored[cid] == 1) return;

		bool is_conflict = false;
		for (int i = 0; i < adj_cons[cid].valid_size; ++i)
		{
			if (colors[adj_cons[cid].adjs[i]] == colors[cid])
			{
				is_conflict = true;
				break;
			}
		}
		if (is_conflict) return;

		for (int i = 0; i < adj_cons[cid].valid_size; ++i)
		{
			palettes[adj_cons[cid].adjs[i]].colors[colors[cid]] = 0;
		}
		is_colored[cid] = 1;
	}

	template <typename AdjItemType, typename PaletteType>
	__global__ void feedTheHungury_Gauss_k(DistCons * cons, int * is_colored, PaletteType * palettes, int con_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (is_colored[cid] == 1) return;

		int count = 0;
		for (int i = 0; i < palettes[cid].valid_size; ++i)
		{
			if (palettes[cid].colors[i] == 1) count += 1;
		}
		if (count == 0)
		{
			palettes[cid].colors[palettes[cid].valid_size] == 1;
			palettes[cid].valid_size += 1;
		}
	}

	__global__ void checkMemory_Gauss_k(int * is_colored, int con_size, int * all_colored)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= con_size) return;
		if (is_colored[cid] == 0) *all_colored = 0;
	}

	template <typename AdjItemType, typename PaletteType>
	void cons_graph_coloring(SolverData * data, int * colors, unsigned int cons_size)
	{
		static AdjItemType * adj_cons = nullptr;
		static PaletteType * palettes = nullptr;
		static int * is_colored = nullptr;
		static int * is_colored_cache = nullptr;
		if (adj_cons == nullptr)
			cudaMalloc((void **)&adj_cons, cons_size * sizeof(AdjItemType));
		if (palettes == nullptr)
			cudaMalloc((void **)&palettes, cons_size * sizeof(PaletteType));
		if (is_colored == nullptr)
			cudaMalloc((void **)&is_colored, cons_size * sizeof(int));
		if (is_colored_cache == nullptr)
			cudaMalloc((void **)&is_colored_cache, cons_size * sizeof(int));

		int threadsPerBlock = 1024;
		dim3 con_blocks((cons_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 con_threads(threadsPerBlock);

		checkCudaErrors(cudaMemset(is_colored, 0, cons_size * sizeof(int)));
		checkCudaErrors(cudaMemset(is_colored_cache, 0, con_blocks.x * sizeof(int)));
		std::vector<int> host_is_colored(cons_size, 0);

#ifdef DEBUG_CUDA		
		std::cout << "pre coloring process" << std::endl;
#endif
		checkCudaErrors(cudaDeviceSynchronize());
		preColoring_Gauss_k<AdjItemType, PaletteType> << <con_blocks, con_threads >> >(data->cons, is_colored, palettes, adj_cons, cons_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		int iter_no = 0;
		bool all_colored = false;
		while (all_colored == false)
		{
#ifdef DEBUG_CUDA		
			std::cout << "iterate coloring process" << std::endl;
#endif
			tentativeColoring_Gauss_k<AdjItemType, PaletteType> << <con_blocks, con_threads >> >(data->cons, is_colored, palettes, colors, iter_no, cons_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			conflictResolution_Gauss_k<AdjItemType, PaletteType> << <con_blocks, con_threads >> >(data->cons, is_colored, palettes, adj_cons, colors, cons_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			feedTheHungury_Gauss_k<AdjItemType, PaletteType> << <con_blocks, con_threads >> >(data->cons, is_colored, palettes, cons_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaMemcpy(host_is_colored.data(), is_colored, cons_size * sizeof(int), cudaMemcpyDeviceToHost));
			all_colored = true;
			for (int i = 0; i < cons_size; ++i)
			{
				if (host_is_colored[i] == 0)
				{
					all_colored = false;
					break;
				}
			}

#ifdef DEBUG_GRAPH_COLORING_REDUCE
			int uncolored_count = cons_size;
			std::vector<int> host_is_colored(con_blocks.x, 0);
			reduce_k <<< con_blocks, con_threads, con_threads.x * sizeof(int) >> > (is_colored, is_colored_cache, cons_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaMemcpy(host_is_colored.data(), is_colored_cache, con_blocks.x * sizeof(int), cudaMemcpyDeviceToHost));
			uncolored_count = cons_size;
			for (int i = 0; i < con_blocks.x; ++i)
				uncolored_count -= host_is_colored[i];
#endif

#ifdef DEBUG_GRAPH_COLORING
			int uncolored_count = cons_size;
			std::vector<int> host_is_colored(cons_size, 0);
			checkCudaErrors(cudaMemcpy(host_is_colored.data(), is_colored, cons_size * sizeof(int), cudaMemcpyDeviceToHost));
			uncolored_count = 0;
			for (int i = 0; i < cons_size; ++i)
				if (host_is_colored[i] == 0) uncolored_count += 1;

#ifdef DEBUG_CUDA		
			std::cout << "uncolored " << uncolored_count << std::endl;
#endif

#endif

			iter_no += 1;
		}

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

		using AdjItem8 = AdjItem<16>;
		using Palette8 = Palette<16>;
		cons_graph_coloring<AdjItem8, Palette8>(data, colors, cons_size);

		for (int i = 0; i < iter_cnt; ++i)
		{
			//for (int j = 0; j < cons_size; ++j)
			//	projectConstraint_Gauss_k << <1, 1 >> >(data->x, data->p, data->v, data->inv_m, data->cons, j);
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

