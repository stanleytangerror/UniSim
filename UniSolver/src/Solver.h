#ifndef UniSolver_h
#define UniSolver_h

#include <cuda_runtime.h>

namespace uni
{
	struct DistCons
	{
		int2 pid;
		float d;
	};

	struct SolverData
	{
		//int p_size = 0;

		float3 * p = nullptr;
		float3 * x = nullptr;
		float3 * v = nullptr;
		float * inv_m = nullptr;
		//int * act_p = nullptr;

		//int con_size = 0;
		DistCons * cons = nullptr;
		//int * act_con = nullptr;
	};

	void initial_device();

	void alloc_cuda_memory(SolverData * data, unsigned int p_size, unsigned int constraint_size);

	void set_positions(SolverData * data, float3 * host_positions, unsigned int p_size);

	void set_velocities(SolverData * data, float3 * host_velocities, unsigned int p_size);

	void set_inv_masses(SolverData * data, float * host_inv_m, unsigned int p_size);

	void set_constraints(SolverData * data, DistCons * host_constraints, unsigned int constraint_size);

	void solve(SolverData * data, unsigned int p_size, unsigned int constraint_size, float time_step, int iter_cnt);

	void get_positions(SolverData * data, float3 * host_positions, unsigned int p_size);

	void free_cuda_memory(SolverData * data);

	void reset_device();

}

#endif