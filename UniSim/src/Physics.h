#ifndef Physics_h
#define Physics_h

#include <Solver.h>
#include <vector>
#include <functional>

class PhysicsSolver
{
private:
	std::unique_ptr<uni::SolverData, std::function<void(uni::SolverData *)>> m_data;
	int m_particle_size;
	int m_constraint_size;
	int m_iter_cnt;

public:
	PhysicsSolver(int p_size, int con_size, float max_radius, int iter_cnt) :
		m_data(new uni::SolverData, [](uni::SolverData * ptr) { uni::free_cuda_memory(ptr); }),
		m_particle_size(p_size), m_constraint_size(con_size), m_iter_cnt(iter_cnt)
	{
		m_data->max_radius = max_radius;
		uni::alloc_cuda_memory(m_data.get(), m_particle_size, m_constraint_size);
	}

	void set_positions(std::vector<float3> & positions)
	{
		uni::set_positions(m_data.get(), positions.data(), m_particle_size);
	}

	void set_velocities(std::vector<float3> & velocities)
	{
		uni::set_velocities(m_data.get(), velocities.data(), m_particle_size);
	}
	
	void set_inv_masses(std::vector<float> & inv_mass)
	{
		uni::set_inv_masses(m_data.get(), inv_mass.data(), m_particle_size);
	}

	void set_phases(std::vector<int> & phases)
	{
		uni::set_phases(m_data.get(), phases.data(), m_particle_size);
	}

	void set_constraints(std::vector<uni::DistanceConstraint> & constraints)
	{
		uni::set_constraints(m_data.get(), constraints.data(), m_constraint_size);
	}

	void tick(float time_step)
	{
		uni::solve(m_data.get(), m_particle_size, m_constraint_size, time_step, m_iter_cnt);
	}

	void get_positions(std::vector<float3> & positions)
	{
		uni::get_positions(m_data.get(), positions.data(), m_particle_size);
	}

};

#endif
