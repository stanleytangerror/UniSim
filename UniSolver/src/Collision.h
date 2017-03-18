#ifndef UNI_Collision_h
#define UNI_Collision_h

#include "Solver.h"
#include "Utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace uni
{

	struct CollideGridSpace
	{
		float3 min_pos;
		float3 max_pos;
		float ceil_size;
		int3 grid_size;

		CollideGridSpace(float3 minp, float3 maxp, float c) :
			min_pos(minp), max_pos(maxp), ceil_size(c)
		{
			float3 s = (maxp + (-minp)) * (1.0f / c);
			grid_size.x = int(s.x) + 1;
			grid_size.y = int(s.y) + 1;
			grid_size.z = int(s.z) + 1;
		}
	};

	void solveCollision(CollideGridSpace const & space, float3 * x, float * inv_m, int * phases, unsigned int p_size, float min_dist, int iter_count);

}

#endif