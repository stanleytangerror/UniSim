#ifndef COMMAND_H
#define COMMAND_H

#include "Actor.h"
#include "Camera.h"
#include "Clock.h"

#include <queue>

class Command
{
public:
	enum class Type
	{
		FreeCameraMoveCommmand,
		FreeCameraGlanceCommmand,
		FreeCameraFocusCommmand,
		PauseCommand,
		PhysicsCommmand
	};

	Command(Type type) : m_type(type) {}

	virtual void execute(Actor * actor) = 0;
	
	virtual ~Command() {}

	Type m_type;
};

class FreeCameraMoveCommmand : public Command
{
public:
	FreeCameraMoveCommmand(Camera_Movement movement, float distance, bool pressed) :
		Command(Command::Type::FreeCameraMoveCommmand), movement(movement), distance(distance), pressed(pressed) {}

	virtual void execute(Actor * actor) override
	{
		//std::cout << "Move " << movement << " " << distance << " " << (pressed ? "moving" : "stop") << std::endl;
		auto cameraActor = (FreeCameraActor *)actor;
		cameraActor->setMove(movement, distance, pressed);
	}

	Camera_Movement movement;
	float distance;
	bool pressed;
};

class FreeCameraGlanceCommmand : public Command
{
public:
	FreeCameraGlanceCommmand(float scrollX, float scrollY, bool constrainPitch) :
		Command(Command::Type::FreeCameraGlanceCommmand), scrollX(scrollX), scrollY(scrollY), constrainPitch(constrainPitch) {}

	virtual void execute(Actor * actor) override
	{
		//std::cout << "Glance " << scrollX << " " << scrollY << std::endl;
		auto cameraActor = (FreeCameraActor *)actor;
		cameraActor->setGlance(scrollX, scrollY);
	}

	float scrollX;
	float scrollY;
	bool constrainPitch;
};

class FreeCameraFocusCommmand : public Command
{
public:
	FreeCameraFocusCommmand(float offset) :
		Command(Command::Type::FreeCameraFocusCommmand), offset(offset) {}

	virtual void execute(Actor * actor) override
	{
		//std::cout << "Focus " << offset << std::endl;
		auto cameraActor = (FreeCameraActor *)actor;
		cameraActor->setFocus(offset);
	}

	float offset;
};

class PauseCommmand : public Command
{
public:
	PauseCommmand() :
		Command(Command::Type::PauseCommand) {}

	virtual void execute(Actor * actor) override
	{
		(actor);

		if (Clock::Instance()->paused())
			Clock::Instance()->resume();
		else
			Clock::Instance()->pause();
	}
};

class RunALittleCommmand : public Command
{
public:
	RunALittleCommmand() 
		: Command(Command::Type::PauseCommand) {}

	virtual void execute(Actor * actor) override
	{
		(actor);

		Clock::Instance()->RunForFrameCount(10);
	}
};

//class PhysicsCommmand : public Command
//{
//public:
//	PhysicsCommmand() :
//		Command(Command::Type::PhysicsCommmand) {}
//
//	virtual void execute(Actor * actor) override
//	{
//		//std::cout << "Focus " << offset << std::endl;
//		auto physicsActor = (PBDPhysicsActor *)actor;
//		physicsActor->flip();
//	}
//};


class CommandQueue
{
public:
	CommandQueue() = default;

	void push(Command * command)
	{
		m_queue.push(command);
	}

	bool empty() { return m_queue.empty(); }

	Command * pop()
	{
		if (m_queue.empty()) return nullptr;
		auto * c = m_queue.front();
		m_queue.pop();
		return c;
	}

protected:
	std::queue<Command *> m_queue;

};

#endif
