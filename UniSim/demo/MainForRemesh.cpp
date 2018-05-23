#define USE_MAIN_FOR_REMESH

#ifdef USE_MAIN_FOR_REMESH

#include "../src/Mesh.h"
#include "../src/Renderer.h"
#include "../src/ResourceManager.h"
#include "../src/Commands.h"
#include "../src/EventManager.h"
#include "../src/Actor.h"
#include "../src/Physics.h"

#include <unordered_map>
#include <memory>
#include <chrono>

class ClothActor : public Actor
{
private:
	SurfaceMeshObject * meshobj;
public:
	ClothActor(SurfaceMeshObject * meshobj) : meshobj(meshobj) {}

	virtual void tick(float deltaTime) override 
	{

	}

};

class RenderActor : public Actor
{
private:
	FreeCamera * camera;
	std::vector<std::pair<std::unique_ptr<SurMeshObjRenderer>, Vec3f>> renders;

public:
	RenderActor(FreeCamera * camera, std::vector<std::pair<SurfaceMeshObject*, Vec3f>> meshes, Shader * shader) :
		camera(camera)
	{
		for (auto & p : meshes)
		{
			auto * meshptr = p.first;
			auto & meshcolor = p.second;
			renders.push_back({ std::make_unique<SurMeshObjRenderer>(meshptr, shader, camera), meshcolor });
		}
	}

	virtual void tick(float deltaTime) override 
	{
		for (auto & p : renders)
		{
			auto & render = p.first;
			auto color = p.second;
			render->update();
			render->draw(color.x(), color.y(), color.z());
		}
	}

};

class PhysicsActor : public Actor
{
private:
	struct MeshPhyInfo
	{
		int phase;
		std::unordered_map<Veridx, int> vid2index;
		std::pair<int, int> particle_index_range;
		std::pair<int, int> constraint_index_range;
		bool need_write_position;
		bool need_read_position;
	};

	std::unique_ptr<PhysicsSolver> m_runner;
	std::vector<std::pair<SurfaceMeshObject *, MeshPhyInfo>> m_meshobjs;

	int m_particle_size;
	std::vector<float3> m_positions;
	std::vector<float3> m_velocities;
	std::vector<float> m_inv_masses;
	std::vector<int> m_phases;

	int m_constraint_size;
	std::vector<uni::DistanceConstraint> m_constraints;

	float m_max_radius;
	int m_iter_count;

public:
	PhysicsActor(float max_radius, int iter_cnt) :
		m_runner(nullptr), m_particle_size(0), m_constraint_size(0),
		m_max_radius(max_radius), m_iter_count(iter_cnt) {}

	void addDynamicMesh(SurfaceMeshObject * meshobj, int phase, float3 velocity, float inv_mass)
	{
		auto & mesh = meshobj->getMesh();
		int p_size = mesh.number_of_vertices();
		int con_size = mesh.number_of_edges();

		m_meshobjs.push_back({ meshobj,
			MeshPhyInfo{ phase, {},
				{ m_particle_size, m_particle_size + p_size },
				{ m_constraint_size, m_constraint_size + con_size },
				false, true } 
		});

		auto & mesh_info = m_meshobjs.back().second;
		auto & vid2index = mesh_info.vid2index;

		m_particle_size += p_size;
		m_positions.reserve(m_particle_size);
		m_velocities.reserve(m_particle_size);
		m_inv_masses.reserve(m_particle_size);
		m_phases.reserve(m_particle_size);

		m_constraint_size += con_size;
		m_constraints.reserve(m_constraint_size);

		for (auto const & vid : mesh.vertices())
		{
			vid2index[vid] = m_positions.size();
			m_positions.push_back({ mesh.point(vid).x(),mesh.point(vid).y(),mesh.point(vid).z() });
			m_velocities.push_back(velocity);
			m_inv_masses.push_back(inv_mass);
			m_phases.push_back(phase);
		}

		for (auto const & eid : mesh.edges())
		{
			auto vid0 = mesh.vertex(eid, 0);
			auto vid1 = mesh.vertex(eid, 1);
			float d = std::sqrt((mesh.point(vid0) - mesh.point(vid1)).squared_length());
			m_constraints.push_back({ { vid2index[vid0], vid2index[vid1] }, d });
		}
	}

	void addKinematicMesh(SurfaceMeshObject * meshobj, int phase)
	{
		auto & mesh = meshobj->getMesh();
		int p_size = mesh.number_of_vertices();

		m_meshobjs.push_back({ meshobj,
			MeshPhyInfo{ phase,{},
			{ m_particle_size, m_particle_size + p_size },
			{ m_constraint_size, m_constraint_size },
			true, false }
		});

		auto & mesh_info = m_meshobjs.back().second;
		auto & vid2index = mesh_info.vid2index;

		m_particle_size += p_size;
		m_positions.reserve(m_particle_size);
		m_velocities.reserve(m_particle_size);
		m_inv_masses.reserve(m_particle_size);
		m_phases.reserve(m_particle_size);

		for (auto const & vid : mesh.vertices())
		{
			vid2index[vid] = m_positions.size();
			m_positions.push_back({ mesh.point(vid).x(),mesh.point(vid).y(),mesh.point(vid).z() });
			m_velocities.push_back({ 0.0f, 0.0f, 0.0f });
			m_inv_masses.push_back(0.0f);
			m_phases.push_back(phase);
		}
	}

	void getReady()
	{
		m_runner.reset(new PhysicsSolver(m_particle_size, m_constraint_size, m_max_radius, m_iter_count));

		m_runner->set_velocities(m_velocities);
		m_runner->set_inv_masses(m_inv_masses);
		m_runner->set_phases(m_phases);
		m_runner->set_constraints(m_constraints);
	}

	virtual void tick(float deltaTime) override
	{
		for (auto & p : m_meshobjs)
		{
			auto & mesh = p.first->getMesh();
			auto & mesh_info = p.second;
			if (mesh_info.need_write_position)
			{
				for (auto const & p : mesh_info.vid2index)
				{
					auto vid = p.first;
					auto idx = p.second;
					m_positions[idx] = { mesh.point(vid).x(), mesh.point(vid).y(), mesh.point(vid).z() };
				}
			}
		}
	
		m_runner->set_positions(m_positions);

		m_runner->tick(deltaTime);

		m_runner->get_positions(m_positions);
	
		for (auto & p : m_meshobjs)
		{
			auto & mesh = p.first->getMesh();
			auto & mesh_info = p.second;
			if (mesh_info.need_read_position)
			{
				for (auto const & p : mesh_info.vid2index)
				{
					auto vid = p.first;
					auto idx = p.second;
					mesh.point(vid) = { m_positions[idx].x, m_positions[idx].y, m_positions[idx].z };
				}
			}

			p.first->computeNormals();

		}
	}


};

int main()
{
	int width = 960, height = 600;
	Screen screen(width, height);
	uni::initial_device();
	  
	FreeCamera camera({ 0.0f, 0.0f, 5.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, float(width) / float(height));

	//ResourceManager::LoadMeshes("cloth", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ColumnsClothesCouple/Cloth06.obj");
	//ResourceManager::LoadMeshes("cloth", "E:/Computer Graphics/Materials/Models/Basic Geometries/SquareCloth_50m50/Clothes.obj");
	//ResourceManager::LoadMeshes("shirt", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan02/Cloth.obj");
	//ResourceManager::LoadMeshes("shirt", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan02/Shirt02.obj");
	//ResourceManager::LoadMeshes("shirt", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan02/Shirt_8k.obj");
	ResourceManager::LoadMeshes("shirt", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan01/Cloth.obj");

	ResourceManager::LoadMeshes("trousers", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan02/Trousers.obj");

	//ResourceManager::LoadMeshes("human", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ColumnsClothesCouple/Columns.obj");
	ResourceManager::LoadMeshes("human", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan02/Man.obj");
	
	ResourceManager::LoadShader("rigid_body", "src/GLSL/rigid_body_vs.glsl", "src/GLSL/rigid_body_frag.glsl", "");
	ResourceManager::LoadShader("cloth", "src/GLSL/cloth_vs.glsl", "src/GLSL/cloth_frag.glsl", "");

	auto * shirtmesh = ResourceManager::GetMesh("shirt")[0];
	auto * trousersmesh = ResourceManager::GetMesh("trousers")[0];
	auto * humanmesh = ResourceManager::GetMesh("human")[0];
	auto * rigidBodyShader = ResourceManager::GetShader("rigid_body");
	auto * clothShader = ResourceManager::GetShader("cloth");

	//shirtmesh->affineTransform({
	//	0.32f, 0.0f, 0.0f, 0.0f,
	//	0.0f, 0.32f, 0.0f, 0.0f,
	//	0.0f, 0.0f, 0.32f, 0.0f
	//});
	shirtmesh->remesh(0.54f, 3);
	shirtmesh->computeNormals();
	std::cout << shirtmesh->getMesh().number_of_vertices() << std::endl;

	//trousersmesh->remesh(0.5f, 3);
	trousersmesh->computeNormals();

	//humanmesh->affineTransform({
	//	0.4f, 0.0f, 0.0f, 0.0f,
	//	0.0f, 0.4f, 0.0f, 0.0f,
	//	0.0f, 0.0f, 0.4f, 0.0f
	//});
	//humanmesh->remesh(0.5f, 2);
	humanmesh->computeNormals();

	FreeCameraActor camera_actor{ &camera };
	ClothActor shirt_actor{ shirtmesh };
	ClothActor trousers_actor{ trousersmesh };
	ClothActor human_actor{ humanmesh };
	RenderActor render_actor{ &camera, 
		{ 
			{ shirtmesh,{0.165f, 0.67f, 0.97f} }//,
			//{ trousersmesh,{ 0.6f, 0.9f, 0.6f } },
			//{ humanmesh,{0.5f, 0.5f, 0.5f} }
		}, 
		//rigidBodyShader
		clothShader
	};
	PhysicsActor physics_actor{ 0.15f, 6 };
	physics_actor.addDynamicMesh(shirtmesh, 1, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//physics_actor.addDynamicMesh(trousersmesh, 2, { 0.0f, 0.0f, 0.0f }, 1.0f);
	physics_actor.addKinematicMesh(humanmesh, 3);
	physics_actor.getReady();

	auto inputHandler = std::make_unique<InputHandler>();
	auto commands = std::make_unique<CommandQueue>();
	EventDispatcher::setEvent(screen.getWindow(), inputHandler.get(), commands.get());

	int cnt = 0;
	long long last_million_seconds = 0;

	while (!screen.closed())
	{
		Clock::Instance()->Tick(1.0f);

		screen.clear();
		screen.pullEvents();
		
		while (!commands->empty())
		{
			auto * command = commands->pop();
			//if (command->m_type == Command::Type::PhysicsCommmand) 
			//	((PhysicsCommmand*)command)->execute(physicsActor.get());
			//else 
				command->execute(&camera_actor);
		}

		float theta = std::sin(cnt / 100.0f) * 0.02f;
		humanmesh->affineTransform(
		{
			std::cos(theta), 0.0f, -std::sin(theta), 0.0f,
			0.0f,			 1.0f, 0.0f,			 0.0f,
			std::sin(theta), 0.0f, std::cos(theta),  0.0f
		});

		camera_actor.tick(1.0f);
		shirt_actor.tick(0.5f);
		trousers_actor.tick(0.5f);
		human_actor.tick(0.5f);

		if (!Clock::Instance()->paused())
		{
			physics_actor.tick(0.5f);
		}

		render_actor.tick(0.5f);

		screen.swapBuffers();

#if SHOW_FPS
		cnt += 1;
		if (cnt % 20 == 0)
		{
			auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count();
			std::cout << "fps " << int(20 * 1000 / float(tmp - last_million_seconds)) << std::endl;
			last_million_seconds = tmp;
		}
#endif
	}
	uni::reset_device();

	return 0;
}

#endif