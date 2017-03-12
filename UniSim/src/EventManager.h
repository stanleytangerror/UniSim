#ifndef EVENT_MANAGER_H
#define EVENT_MANAGER_H

#include "Commands.h"

#include <GLFW/glfw3.h>

#include <functional>
#include <list>

class InputHandler
{
public:
	InputHandler() = default;
	
	void key_command_sender(GLFWwindow* window, CommandQueue * const queue)
	{
		//cout << key << endl;

		auto w = glfwGetKey(window, GLFW_KEY_W);
		if (w != w_state)
		{
			w_state = w;
			queue->push(new FreeCameraMoveCommmand(Camera_Movement::FORWARD, 0.05f, w_state == GLFW_PRESS));
		}
		auto s = glfwGetKey(window, GLFW_KEY_S);
		if (s != s_state)
		{
			s_state = s;
			queue->push(new FreeCameraMoveCommmand(Camera_Movement::BACKWARD, 0.05f, s_state == GLFW_PRESS));
		}
		auto a = glfwGetKey(window, GLFW_KEY_A);
		if (a != a_state)
		{
			a_state = a;
			queue->push(new FreeCameraMoveCommmand(Camera_Movement::LEFT, 0.05f, a_state == GLFW_PRESS));
		}
		auto d = glfwGetKey(window, GLFW_KEY_D);
		if (d != d_state)
		{
			d_state = d;
			queue->push(new FreeCameraMoveCommmand(Camera_Movement::RIGHT, 0.05f, d_state == GLFW_PRESS));
		}
		//auto x = glfwGetKey(window, GLFW_KEY_X);
		//if (x == GLFW_PRESS)
		//	queue->push(new PhysicsCommmand());
	}

	void cursor_command_sender(CommandQueue * const queue, double xpos, double ypos)
	{
		//std::cout << "mouse callback" << std::endl;
		if (firstMove)
		{
			firstMove = false;
			X = xpos;
			Y = ypos;
		}
		else
		{
			auto curX = xpos;
			auto curY = ypos;
			firstMove = false;
			auto offsetX = curX - X;
			auto offsetY = Y - curY;  // Reversed since y-coordinates go from bottom to left
			queue->push(new FreeCameraGlanceCommmand(GLfloat(offsetX), GLfloat(offsetY), true));
			X = curX;
			Y = curY;
		}
	}

	void scroll_command_sender(CommandQueue * const queue, double xoffset, double yoffset)
	{
		queue->push(new FreeCameraFocusCommmand(yoffset * 0.05f));
	}

protected:

	int w_state = GLFW_RELEASE;
	int a_state = GLFW_RELEASE;
	int s_state = GLFW_RELEASE;
	int d_state = GLFW_RELEASE;
	// cursor event
	double X = 0.0f, Y = 0.0f;
	bool firstMove = true;

};

class EventDispatcher
{
public:
	static void setEvent(GLFWwindow * window, InputHandler * hdl, CommandQueue * q)
	{
		handler = hdl;
		queue = q;
		glfwSetKeyCallback(window, key_callback);
		glfwSetCursorPosCallback(window, cursor_callback);
		glfwSetScrollCallback(window, scroll_callback);
	}

protected:

	static InputHandler * handler;
	static CommandQueue * queue;

	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
	{
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);
		handler->key_command_sender(window, queue);
	}

	static void cursor_callback(GLFWwindow* window, double xpos, double ypos)
	{
		handler->cursor_command_sender(queue, xpos, ypos);
	}

	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		handler->scroll_command_sender(queue, xoffset, yoffset);
	}

};

#endif