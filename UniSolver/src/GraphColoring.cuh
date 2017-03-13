/*******************************************
 According to "Vivace: a Practical Gauss-Seidel Method for Stable Soft Body Dynamics"
********************************************/

#ifndef GraphColoring_h
#define GraphColoring_h

#include "Solver.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace uni
{
	//#define DEBUG_CUDA
	//#define DEBUG_GRAPH_COLORING
	//#define DEBUG_GRAPH_COLORING_REDUCE
	//#define PROFILE_CUDA


	template <int Size>
	struct AdjItem
	{
		static const int max_size = Size;
		int valid_size;
		int adjs[max_size];
	};

	template <int Size>
	struct Palette
	{
		static const int max_size = Size;
		int valid_size;
		int colors[max_size];
	};


	template <typename PaletteType>
	__global__ void preColoring_k(int * is_colored, PaletteType * palettes, int node_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= node_size) return;

		is_colored[cid] = 0;

		palettes[cid].valid_size = palettes[cid].max_size * 0.94f;
		for (int i = 0; i < palettes[cid].valid_size; ++i)
			palettes[cid].colors[i] = 1;
	}

	template <typename PaletteType>
	__global__ void tentativeColoring_k(int * is_colored, PaletteType * palettes, int * colors, int iter_no, int node_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= node_size) return;
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
	__global__ void conflictResolution_k(int * is_colored, PaletteType * palettes, AdjItemType * adj_table, int * node_colors, int node_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= node_size) return;
		if (is_colored[cid] == 1) return;

		bool is_conflict = false;
		for (int i = 0; i < adj_table[cid].valid_size; ++i)
		{
			if (node_colors[adj_table[cid].adjs[i]] == node_colors[cid])
			{
				is_conflict = true;
				break;
			}
		}
		if (is_conflict) return;

		for (int i = 0; i < adj_table[cid].valid_size; ++i)
		{
			palettes[adj_table[cid].adjs[i]].colors[node_colors[cid]] = 0;
		}
		is_colored[cid] = 1;
	}

	template <typename PaletteType>
	__global__ void feedTheHungury_k(int * is_colored, PaletteType * palettes, int node_size)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= node_size) return;
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

	__global__ void checkMemory_k(int * is_colored, int node_size, int * all_colored)
	{
		unsigned int cid = threadIdx.x + blockDim.x * blockIdx.x;
		if (cid >= node_size) return;
		if (is_colored[cid] == 0) *all_colored = 0;
	}

	template <int MaxDegree>
	void graph_coloring(AdjItem<MaxDegree> * adj_table, int * node_colors, unsigned int node_size)
	{
		static Palette<MaxDegree> * palettes = nullptr;
		static int * is_colored = nullptr;
		static int * is_colored_cache = nullptr;
		if (palettes == nullptr)
			cudaMalloc((void **)&palettes, node_size * sizeof(Palette<MaxDegree>));
		if (is_colored == nullptr)
			cudaMalloc((void **)&is_colored, node_size * sizeof(int));
		if (is_colored_cache == nullptr)
			cudaMalloc((void **)&is_colored_cache, node_size * sizeof(int));

		int threadsPerBlock = 1024;
		dim3 con_blocks((node_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 con_threads(threadsPerBlock);

		checkCudaErrors(cudaMemset(is_colored, 0, node_size * sizeof(int)));
		checkCudaErrors(cudaMemset(is_colored_cache, 0, con_blocks.x * sizeof(int)));
		std::vector<int> host_is_colored(node_size, 0);

#ifdef DEBUG_CUDA		
		std::cout << "pre coloring process" << std::endl;
#endif
		checkCudaErrors(cudaDeviceSynchronize());
		preColoring_k<Palette<MaxDegree>> << <con_blocks, con_threads >> >(is_colored, palettes, node_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		int iter_no = 0;
		bool all_colored = false;
		while (all_colored == false)
		{
#ifdef DEBUG_CUDA		
			std::cout << "iterate coloring process" << std::endl;
#endif
			tentativeColoring_k<Palette<MaxDegree>> << <con_blocks, con_threads >> >(is_colored, palettes, node_colors, iter_no, node_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			conflictResolution_k<AdjItem<MaxDegree>, Palette<MaxDegree>> << <con_blocks, con_threads >> >(is_colored, palettes, adj_table, node_colors, node_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			feedTheHungury_k<Palette<MaxDegree>> << <con_blocks, con_threads >> >(is_colored, palettes, node_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaMemcpy(host_is_colored.data(), is_colored, node_size * sizeof(int), cudaMemcpyDeviceToHost));
			all_colored = true;
			for (int i = 0; i < node_size; ++i)
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
			reduce_k << < con_blocks, con_threads, con_threads.x * sizeof(int) >> > (is_colored, is_colored_cache, cons_size);
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

}

#endif