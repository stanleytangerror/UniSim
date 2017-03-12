#ifndef ACTOR_H
#define ACTOR_H

#include "Camera.h"

class Actor
{
public:
	virtual ~Actor() = default;

	virtual void tick(float deltaTime) = 0;

};

class FreeCameraActor : public Actor
{
public:
	FreeCameraActor(FreeCamera * camera) :
		m_camera(camera) {}

	void setMove(Camera_Movement direction, GLfloat offset, bool moving)
	{
		if (moving)
		{
			this->direction = direction;
			this->moveStep = offset;
		}
		else
		{
			this->direction = direction;
			this->moveStep = 0.0f;
		}
	}

	void setGlance(GLfloat scrollX, GLfloat scrollY)
	{
		this->scrollX += scrollX;
		this->scrollY += scrollY;
	}

	void setFocus(GLfloat offset)
	{
		focusOffset += offset;
	}

	virtual void tick(float deltaTime) override
	{
		//std::cout << "update camera " << direction << " " << moveStep << std::endl;
		m_camera->move(direction, moveStep * deltaTime);

		m_camera->focus(focusOffset * deltaTime);
		focusOffset = 0.0f;

		m_camera->glance(scrollX * deltaTime, scrollY * deltaTime, true);
		scrollX = 0.0f;
		scrollY = 0.0f;
	}

protected:
	FreeCamera * m_camera;

	Camera_Movement direction;
	GLfloat moveStep;
	GLfloat scrollX, scrollY;
	GLfloat focusOffset;
};

//class MeshClothActor : public Actor
//{
//public:
//	MeshClothActor(SurfaceMeshObject * mesh) :
//		m_mesh(mesh) {}
//
//	virtual void tick(float deltaTime) override {}
//	
//	void affineTransform(AffTransformf const & aff)
//	{
//		m_mesh->affineTransform(aff);
//	}
//
//protected:
//	SurfaceMeshObject * const m_mesh;
//
//};
//
//class MeshKinematicActor : public Actor
//{
//public:
//	MeshKinematicActor(SurfaceMeshObject * mesh) :
//		m_mesh(mesh) {}
//
//	virtual void tick(float deltaTime) override
//	{
//		m_mesh->computeNormals();
//	}
//	
//	template <typename V3f>
//	bool setPositions(std::vector<V3f> const & positions)
//	{
//		return m_mesh->resetHandledPositions(positions);
//	}
//
//protected:
//	SurfaceMeshObject * const m_mesh;
//};
//
//class PBDPhysicsActor : public Actor
//{
//public:
//	PBDPhysicsActor(PBDRunner * physics, bool running = false) :
//		m_physics(physics), m_running(running), m_count(0) {}
//
//	virtual void tick(float deltaTime) override
//	{
//		if (m_running)
//		{
//			m_physics->run(deltaTime, m_iterateNumber, 1.0f);
//			m_count += 1;
//		}
//	}
//
//	void flip() { m_running = !m_running; }
//
//protected:
//	PBDRunner * const m_physics;
//	int m_iterateNumber = 6;
//	bool m_running;
//	int m_count;
//};
//
//class FlexPhysicActor : public Actor 
//{
//public:
//	FlexPhysicActor(FlexWrapper * flex, bool running = false) : m_flex(flex), m_running(running) {};
//
//	virtual void tick(float deltaTime) override
//	{
//		if (!m_running) return;
//		m_flex->preprocess();
//		m_flex->setFlex();
//		m_flex->run(deltaTime);
//		m_flex->getFlex();
//		m_flex->postprocess();
//	}
//
//	void flip() 
//	{ 
//		m_running = !m_running;
//	}
//
//protected:
//	FlexWrapper * m_flex;
//	bool m_running;
//};

#endif