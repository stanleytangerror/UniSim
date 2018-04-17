#ifndef SCREEN_H
#define SCREEN_H

#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW\glfw3.h>

#include <iostream>

/* TODO relate to both control and render module */
class Screen
{
public:
	Screen(unsigned int width = 800, unsigned int height = 600) :
		screenWidth(width), screenHeight(height), aspectRatio((float) width / height)
	{
		// Init GLFW
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

		window = glfwCreateWindow(screenWidth, screenHeight, "Scene", nullptr, nullptr); // Windowed
		glfwMakeContextCurrent(window);

		// Options
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		// Initialize GLEW to setup the OpenGL Function pointers
		glewExperimental = GL_TRUE;
		glewInit();

		std::cout << "INFO::ENV: OpenGL version " << glGetString(GL_VERSION) << std::endl;
		std::cout << "INFO::ENV: sizeof GLfloat " << sizeof GLfloat << std::endl;
		std::cout << "INFO::ENV: sizeof GLuint " << sizeof GLuint << std::endl;

		// Define the viewport dimensions
		glViewport(0, 0, screenWidth, screenHeight);
	}

	int const screenWidth, screenHeight;
	float const aspectRatio;
	GLFWwindow * window;

	GLFWwindow * const getWindow() const { return window; }

	void clear()
	{
		glClearColor(1.f, 1.f, 1.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
	}

	void swapBuffers()
	{
		glfwSwapBuffers(window);
	}

	bool closed()
	{
		return glfwWindowShouldClose(window);
	}

	void pullEvents()
	{
		//// Set frame time
		//GLfloat currentFrame = (GLfloat)glfwGetTime();
		//deltaTime = currentFrame - lastFrame;
		//lastFrame = currentFrame;

		// Check and call events
		glfwPollEvents();
	}

	~Screen()
	{
		glfwTerminate();
	}

};

#endif
