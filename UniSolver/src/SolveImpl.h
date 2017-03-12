#ifndef SolveImpl_h
#define SolveImpl_h

#include "Solver.h"
#include <cuda_runtime.h>

namespace uni
{
	void solve_Gauss(SolverData * data, unsigned int p_size, unsigned int cons_size, float time_step, int iter_cnt);

	void solve_Jacobi(SolverData * data, unsigned int p_size, unsigned int cons_size, float time_step, int iter_cnt);
}


#endif
