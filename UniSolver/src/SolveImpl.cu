#include "SolveImpl.h"
#include "Solver.h"
#include "Utils.cuh"
#include "GraphColoring.cuh"
#include "Collision.h"
#include "helper_cuda.h"

#include "Profile.h"

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <device_functions.h>

//#define DEBUG_SOLVER_IMPLEMENTATION
//#define DO_GRAPH_COLORING_EACH_FRAME

namespace
{
	static const std::string profileFile = "E:/Microsoft Visual Studio 2015/Workspace/UniSim/UniSolver/log/LogFile.txt";

	void ProfileNewFrame()
	{
		std::ofstream outfile;
		outfile.open(profileFile, std::ios_base::app);
		outfile << std::endl;
	}

	void ProfileResult(const ScopedProfiler::TimeCount start, const ScopedProfiler::TimeCount end)
	{
		std::ofstream outfile;
		outfile.open(profileFile, std::ios_base::app);
		outfile << std::to_string(end - start) << ", ";
	}
}

#define PROFILE_SCOPE							\
	ScopedProfiler profiler(ProfileResult);		\
	(profiler);

namespace uni
{
	__device__ float eps = 1e-20f;

	int threadsPerBlock = 512;

	template <typename AdjItemType>
	__global__ void adjTable_k(DistanceConstraint * cons, AdjItemType * adj_cons, int con_size)
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

	__global__ void projectConstraint_Gauss_k(float3 * x, float3 * p, float3 * v, float * inv_m, DistanceConstraint * cons, int * colors, unsigned int con_size, int gid)
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
		static bool	graphColored = false;
		ProfileNewFrame();

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

#ifdef DEBUG_SOLVER_IMPLEMENTATION
		std::cout << "start free run" << std::endl;
#endif

		{
			PROFILE_SCOPE;
			freeRun_Gauss_k << <p_blocks, p_threads >> > (data->x, data->p, data->v, data->inv_m, time_step, p_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());
		}

#ifdef DEBUG_SOLVER_IMPLEMENTATION
		std::cout << "start solve collision" << std::endl;
#endif

		{
			PROFILE_SCOPE;
			CollideGridSpace collide_space{ { -30.0f, -30.0f, -30.0f },{ 30.0f, 30.0f, 30.0f }, 2 * data->max_radius };
			solveCollision(collide_space, data->p, data->inv_m, data->phase, p_size, 2.0f * 2.0f * data->max_radius, 2 * iter_cnt);
		}

		{
			PROFILE_SCOPE;

			if (colors == nullptr)
			{
				cudaMalloc((void **)&colors, cons_size * sizeof(int));
			}

#ifdef DO_GRAPH_COLORING_EACH_FRAME

#ifdef DEBUG_SOLVER_IMPLEMENTATION
			std::cout << "start graph coloring" << std::endl;
#endif

			callGraphColoring_Gauss<32>(data, colors, cons_size);
#else
			if (!graphColored)
			{

#ifdef DEBUG_SOLVER_IMPLEMENTATION
				std::cout << "start graph coloring" << std::endl;
#endif

				callGraphColoring_Gauss<32>(data, colors, cons_size);
				graphColored = true;
			}
#endif
		}

		{
			PROFILE_SCOPE;
			for (int i = 0; i < iter_cnt; ++i)
			{
#ifdef DEBUG_SOLVER_IMPLEMENTATION
				std::cout << "start solve iterate no." << i << std::endl;
#endif
				for (int gid = 0; gid < 32; ++gid)
				{

#ifdef DEBUG_SOLVER_IMPLEMENTATION
					std::cout << "start solve iterate no." << i << " group no." << gid << std::endl;
#endif

					projectConstraint_Gauss_k << <con_blocks, con_threads >> >(data->x, data->p, data->v, data->inv_m, data->cons, colors, cons_size, gid);
					getLastCudaError("Kernel execution failed");
					checkCudaErrors(cudaDeviceSynchronize());
				}
			}
		}

#ifdef DEBUG_SOLVER_IMPLEMENTATION
		std::cout << "start update states" << std::endl;
#endif

		{
			PROFILE_SCOPE;
			updateState_Gauss_k << <p_blocks, p_threads >> >(data->x, data->p, data->v, data->inv_m, time_step, p_size);
			getLastCudaError("Kernel execution failed");
			checkCudaErrors(cudaDeviceSynchronize());
		}

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

}

