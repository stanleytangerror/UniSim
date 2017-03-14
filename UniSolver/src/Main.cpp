#include "Solver.h"

#include <vector>
#include <iostream>
#include <string>

int main()
{
	const int particle_size = 32;
	std::vector<float3> positions;
	std::vector<float3> velocities(particle_size, float3{});
	std::vector<float> inv_mass(particle_size, 1.0f);
	for (int i = 0; i < particle_size; ++i)
	{
		positions.push_back({ i * 10.0f, 0.0f, 0.0f });
	}

	std::vector<uni::DistanceConstraint> constraints;
	
	for (int i = 0; i < particle_size - 1; ++i)
		constraints.push_back({ { i, i + 1 }, 9.9f });
	for (int i = 0; i < particle_size - 2; ++i)
		constraints.push_back({ { i, i + 2 }, 19.8f });

	int constraint_size = constraints.size();
	//std::vector<int2> constraints = { { 0, 1 },{ 1, 2 },{ 2, 3 }, {0, 2}, {1, 3} };
	//std::vector<float> distance = { 9.0f, 9.0f, 9.0f, 18.0f, 18.0f };

	float time_step = 1.0f;
	
	uni::SolverData data;

	uni::initial_device();
	uni::alloc_cuda_memory(&data, particle_size, constraint_size);

	uni::set_positions(&data, positions.data(), particle_size);
	uni::set_velocities(&data, velocities.data(), particle_size);
	uni::set_inv_masses(&data, inv_mass.data(), particle_size);
	uni::set_constraints(&data, constraints.data(), constraint_size);

	for (int i = 0; i < 10; ++i)
	{
		float e = 0.0f;
		for (int i = 0; i < constraints.size(); ++i)
		{
			auto c = constraints[i];
			auto delta_l = std::abs(positions[c.pid.y].x - positions[c.pid.x].x) - c.d;
			std::cout << delta_l << " ";
			e += delta_l * delta_l;
		}
		std::cout << std::endl << "                       " << e << std::endl;;

		uni::solve(&data, particle_size, constraint_size, time_step, 4);
		uni::get_positions(&data, positions.data(), particle_size);

	}

	uni::free_cuda_memory(&data);
	uni::reset_device();

	return 0;
}

