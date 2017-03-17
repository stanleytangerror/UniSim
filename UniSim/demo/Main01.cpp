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

class ThingActor : public Actor
{
private:
	SurfaceMeshObject * meshobj;
public:
	ThingActor(SurfaceMeshObject * meshobj) : meshobj(meshobj) {}

	virtual void tick(float deltaTime) override 
	{
		meshobj->computeNormals();
	}
};

class RenderActor : public Actor
{
private:
	FreeCamera * camera;
	std::vector<std::unique_ptr<SurMeshObjRenderer>> renders;

public:
	RenderActor(FreeCamera * camera, std::vector<SurfaceMeshObject*> meshes, Shader * shader) :
		camera(camera)
	{
		for (auto * m : meshes)
		{
			renders.push_back(std::make_unique<SurMeshObjRenderer>(m, shader, camera));
		}
	}

	virtual void tick(float deltaTime) override 
	{
		for (auto & r : renders)
		{
			r->update();
			r->draw(0.5f, 0.3f, 0.6f);
		}
	}

};

class PhysicsActor : public Actor
{
private:
	std::unique_ptr<PhysicsSolver> m_runner;
	SurfaceMeshObject * m_meshobj;

	std::unordered_map<Veridx, int> vid2index;

	std::vector<float3> positions;
	std::vector<float3> velocities;
	std::vector<float> inv_masses;
	std::vector<int> active_particles;
	std::vector<uni::DistanceConstraint> constraints;

public:
	PhysicsActor(SurfaceMeshObject * meshobj, int iter_cnt) :
		m_meshobj(meshobj), m_runner(nullptr)
	{
		auto & mesh = m_meshobj->getMesh();
		int p_size = mesh.number_of_vertices();
		int con_size = mesh.number_of_edges();

		m_runner.reset(new PhysicsSolver(p_size, con_size, iter_cnt));

		positions.reserve(p_size);
		velocities.assign(p_size, { 0.0f, 0.0f, 0.0f });
		active_particles.assign(p_size, 1);
		inv_masses.assign(p_size, 1.0f);
		constraints.reserve(mesh.number_of_edges());

		int p_cnt = 0;
		for (auto const & vid : mesh.vertices())
		{
			vid2index[vid] = p_cnt++;
			positions.push_back({ mesh.point(vid).x(),mesh.point(vid).y(),mesh.point(vid).z() });
		}

		for (auto const & eid : mesh.edges())
		{
			auto vid0 = mesh.vertex(eid, 0);
			auto vid1 = mesh.vertex(eid, 1);
			float d = std::sqrt((mesh.point(vid0) - mesh.point(vid1)).squared_length());
			constraints.push_back({ { vid2index[vid0], vid2index[vid1] }, d });
		}

		//positions[10] = { positions[10].x + 0.01f, 0.0f, 0.0f };
		inv_masses[100] = 0.0f;

		m_runner->set_velocities(velocities);
		m_runner->set_inv_masses(inv_masses);
		m_runner->set_constraints(constraints);
	}

	void write_back()
	{
		auto & mesh = m_meshobj->getMesh();
		for (auto const & p : vid2index)
		{
			auto vid = p.first;
			auto idx = p.second;
			mesh.point(vid) = { positions[idx].x, positions[idx].y, positions[idx].z };
		}
	}

	virtual void tick(float deltaTime) override
	{
		m_runner->set_positions(positions);

		m_runner->tick(deltaTime);
		m_runner->get_states(positions);
		write_back();
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
	ResourceManager::LoadMeshes("cloth", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan01/Cloth.obj");
	
	//ResourceManager::LoadMeshes("human", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ColumnsClothesCouple/Columns.obj");
	//ResourceManager::LoadMeshes("human", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan01/Man.obj");
	//ResourceManager::LoadMeshes("cloth", "E:/Computer Graphics/Materials/Models/ComplexScenes/Scene_ClothMan01/Man.obj");
	
	ResourceManager::LoadShader("rigid_body", "src/GLSL/rigid_body_vs.glsl", "src/GLSL/rigid_body_frag.glsl", "");

	auto * clothmesh = ResourceManager::GetMesh("cloth")[0];
	//auto * humanmesh = ResourceManager::GetMesh("human")[0];
	auto * shader = ResourceManager::GetShader("rigid_body");

	//clothmesh->affineTransform({
	//	0.64f, 0.0f, 0.0f, 0.0f,
	//	0.0f, 0.64f, 0.0f, 0.0f,
	//	0.0f, 0.0f, 0.64f, 0.0f
	//});
	clothmesh->remesh(0.5f, 4);
	clothmesh->computeNormals();
	//humanmesh->remesh(0.1f, 3);
	//humanmesh->computeNormals();

	FreeCameraActor camera_actor{ &camera };
	ThingActor cloth_actor{ clothmesh };
	//ThingActor human_actor{ clothmesh };
	RenderActor render_actor{ &camera, { clothmesh }, shader };
	PhysicsActor physics_actor{ clothmesh, 5 };

	auto inputHandler = std::make_unique<InputHandler>();
	auto commands = std::make_unique<CommandQueue>();
	EventDispatcher::setEvent(screen.getWindow(), inputHandler.get(), commands.get());

	int cnt = 0;
	long long last_million_seconds = 0;

	while (!screen.closed())
	{
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

		camera_actor.tick(1.0f);
		physics_actor.tick(0.5f);
		cloth_actor.tick(0.5f);
		//human_actor.tick(0.5f);
		render_actor.tick(0.5f);

		screen.swapBuffers();

		cnt += 1;
		if (cnt % 20 == 0)
		{
			auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::system_clock::now().time_since_epoch()).count();
			std::cout << "fps " << int(20 * 1000 / float(tmp - last_million_seconds)) << std::endl;
			last_million_seconds = tmp;
		}
	}
	uni::reset_device();

	return 0;
}