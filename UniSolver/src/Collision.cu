#include "Collision.h"
#include "Utils.cuh"
#include "helper_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\execution_policy.h>

namespace uni
{
	__device__ float eps = 1e-20f;

	namespace Slot
	{
		__device__ unsigned int ceil2Slot(int range, int3 const & id)
		{
			unsigned int res = 3;
			res = id.x + 0x9e3779b9 + (res << 6) + (res >> 2);
			res = id.y + 0x9e3779b9 + (res << 6) + (res >> 2);
			res = id.z + 0x9e3779b9 + (res << 6) + (res >> 2);
			return res % range;
		}

		__global__ void getPSlot_k(CollideGridSpace const & space, int slot_count, float3 * x, int * p_slots, int * p_ids, unsigned int p_size)
		{
			unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
			if (pid >= p_size) return;
			p_ids[pid] = pid;

			float3 offset = x[pid] + (-space.min_pos);
			int3 ceil_id = { int(offset.x / space.ceil_size),int(offset.y / space.ceil_size),int(offset.z / space.ceil_size) };

			unsigned int slot_id = ceil2Slot(slot_count, ceil_id);
			p_slots[pid] = slot_id;
		}
	}

	__device__ unsigned int offsetToCeil(CollideGridSpace const & space, int3 const & ceil_id)
	{
		return ceil_id.x * space.grid_size.y * space.grid_size.z + ceil_id.y * space.grid_size.z + ceil_id.z;
	}

	__device__ bool solvePWiseCollideConstraint(float3 & res, float3 const & p, float3 const & other_p, float inv_m, float other_inv_m, float d)
	{
		if (inv_m + other_inv_m < eps) return false;

		float sq_dist = squared_length(p + (-other_p));
		if (sq_dist >= d * d) return false;

		float dist = sqrtf(sq_dist);
		if (-eps < dist && dist < eps) return false;
		float delta_d = dist - d;

		float3 tmp = delta_d / ((inv_m + other_inv_m) * dist) * (p + (-other_p));
		res = -inv_m * tmp;

		return true;
	}

	__global__ void getParticleCeil_k(CollideGridSpace space, int ceil_count, float3 * x, int * p_ceils, int * p_ids, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;
		p_ids[pid] = pid;

		float3 offset = x[pid] + (-space.min_pos);
		int3 ceil_id = { int(offset.x / space.ceil_size),int(offset.y / space.ceil_size),int(offset.z / space.ceil_size) };
		if (ceil_id.x >= space.grid_size.x || ceil_id.y >= space.grid_size.y || ceil_id.z >= space.grid_size.z)
		{
			p_ceils[pid] = -1;
			return;
		}
		int ceil_hash = offsetToCeil(space, ceil_id);
		p_ceils[pid] = ceil_hash;
	}

	__global__ void getCeilOffset_k(int2 * ceil_ranges, int * p_ceils, int ceil_count, unsigned int p_size)
	{
		unsigned int sorted_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (sorted_idx >= p_size) return;
		
		if (sorted_idx == 0 || p_ceils[sorted_idx - 1] != p_ceils[sorted_idx])
			if (p_ceils[sorted_idx] != -1)
			ceil_ranges[p_ceils[sorted_idx]].x = sorted_idx;
		
		if (sorted_idx == p_size - 1 || p_ceils[sorted_idx] != p_ceils[sorted_idx + 1])
			if (p_ceils[sorted_idx] != -1)
			ceil_ranges[p_ceils[sorted_idx]].y = sorted_idx;
	}

	__global__ void projectCollision_Jacobi_k(CollideGridSpace space, int2 * ceil_ranges, int ceil_count, int * par_ids,
		float3 * pos, float3 * delta_pos, float * inv_m, float min_dist, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;
		delta_pos[pid] = { 0.0f, 0.0f, 0.0f };

		float3 offset = pos[pid] + (-space.min_pos);
		int3 ceil_id = { int(offset.x / space.ceil_size),int(offset.y / space.ceil_size),int(offset.z / space.ceil_size) };
		int ceil_hash = offsetToCeil(space, ceil_id);
		int count = 0;
		for (int i = -1; i <= 1; ++i) for (int j = -1; j <= 1; ++j) for (int k = -1; k <= 1; ++k)
		{
			int h = ceil_hash + i * space.grid_size.y * space.grid_size.z + j * space.grid_size.z + k;
			if (!(0 <= h && h < ceil_count)) continue;
			int par_start = ceil_ranges[h].x;
			int par_end = ceil_ranges[h].y;
			if (!(0 <= par_start && par_start < p_size)) continue;
			if (!(0 <= par_end && par_end < p_size)) continue;
			for (int i = par_start; i <= par_end; ++i)
			{
				int other_pid = par_ids[i];
				if (pid == other_pid) continue;
				bool collide = solvePWiseCollideConstraint(delta_pos[pid], pos[pid], pos[other_pid], inv_m[pid], inv_m[other_pid], min_dist);
				if (collide) count += 1;
			}
		}
		if (count > 0)
		{
			delta_pos[pid] = delta_pos[pid] * (1.0f / count);
		}
	}

	__global__ void udpate_Jacobi_k(float3 * pos, float3 * delta_pos, unsigned int p_size)
	{
		unsigned int pid = threadIdx.x + blockDim.x * blockIdx.x;
		if (pid >= p_size) return;
	
		pos[pid] = pos[pid] + delta_pos[pid];
	}

	void solveCollision(CollideGridSpace const & space, float3 * x, float * inv_m, unsigned int p_size, float min_dist, int iter_count)
	{
		static int * p_ceils = nullptr;
		static int * par_ids = nullptr;
		static int2 * ceil_ranges = nullptr;
		static int ceil_count = space.grid_size.x * space.grid_size.y * space.grid_size.z;
		static float3 * delta_x = nullptr;
		static std::vector<int2> const arrn1(ceil_count, { -1, -1 });
		
		if (p_ceils == nullptr)
			checkCudaErrors(cudaMalloc((void **)& p_ceils, p_size * sizeof(int)));
		if (par_ids == nullptr)
			checkCudaErrors(cudaMalloc((void **)& par_ids, p_size * sizeof(int)));
		if (delta_x == nullptr)
			checkCudaErrors(cudaMalloc((void **)& delta_x, p_size * sizeof(float3)));
		if (ceil_ranges == nullptr)
			checkCudaErrors(cudaMalloc((void **)& ceil_ranges, ceil_count * sizeof(int2)));
		//checkCudaErrors(cudaMalloc((void **)& space, sizeof(CollideGridSpace)));

		checkCudaErrors(cudaMemset(delta_x, 0, p_size * sizeof(float3)));
		checkCudaErrors(cudaMemcpy(ceil_ranges, arrn1.data(), ceil_count * sizeof(int2), cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(space, &host_space, sizeof(int2), cudaMemcpyHostToDevice));

		int threadsPerBlock = 1024;

		dim3 p_blocks((p_size + threadsPerBlock - 1) / threadsPerBlock);
		dim3 p_threads(threadsPerBlock);

		/* buildGridStructure */
		getParticleCeil_k << <p_blocks, p_threads >> >(space, ceil_count, x, p_ceils, par_ids, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

		thrust::sort_by_key(thrust::device, p_ceils, p_ceils + p_size, par_ids);

		getCeilOffset_k << <p_blocks, p_threads >> >(ceil_ranges, p_ceils, ceil_count, p_size);
		getLastCudaError("Kernel execution failed");
		checkCudaErrors(cudaDeviceSynchronize());
		/* buildGridStructure */

	//}

	//void solvePWiseCollision_Jacobi(CollideGridSpace const & space, unsigned int p_size, int iter_count)
	//{

	//	int threadsPerBlock = 1024;

	//	dim3 p_blocks((p_size + threadsPerBlock - 1) / threadsPerBlock);
	//	dim3 p_threads(threadsPerBlock);

		for (int i = 0; i < iter_count; ++i)
		{
			projectCollision_Jacobi_k << <p_blocks, p_threads >> >(space, ceil_ranges, ceil_count, par_ids, x, delta_x, inv_m, min_dist, p_size);
			getLastCudaError("Kernel execution failed");
		
			udpate_Jacobi_k << <p_blocks, p_threads >> >(x, delta_x, p_size);
			getLastCudaError("Kernel execution failed");
		}
		checkCudaErrors(cudaDeviceSynchronize());

	}

}