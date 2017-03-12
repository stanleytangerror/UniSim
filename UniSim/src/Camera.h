#ifndef CAMERA_H
#define CAMERA_H

#include "OpenGLContext.h"

// GLM Mathemtics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <iostream>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

enum Posture_Adjustment {
	DRAG,
	SCROLL
};

// Default camera values
const GLfloat YAW = -90.0f;
const GLfloat PITCH = 0.0f;
//const GLfloat SPEED = 0.5f;
//const GLfloat SENSITIVTY = 0.5f;
const GLfloat ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the corresponding Eular Angles, Vectors and Matrices for use in OpenGL
class FreeCamera
{
public:
	// Constructor with vectors
	FreeCamera(
		glm::vec3 position = { 0.0f, 0.0f, 10.0f },
		glm::vec3 up = { 0.0f, 1.0f, 0.0f },
		glm::vec3 front = { 0.0f, 0.0f, -1.0f },
		GLfloat aspectRation = 1.0f,
		GLfloat yaw = YAW, GLfloat pitch = PITCH) :
		Front(front), Zoom(ZOOM), AspectRatio(aspectRation)
	{
		//CameraNo++;
		this->Position = position;
		this->WorldUp = up;
		this->Yaw = yaw;
		this->Pitch = pitch;
		this->updateCameraVectors();
	}

	// Constructor with scalar values
	//FreeCamera(GLfloat posX, GLfloat posY, GLfloat posZ,
	//	GLfloat upX, GLfloat upY, GLfloat upZ,
	//	GLfloat aspectRation,
	//	GLfloat yaw, GLfloat pitch) :
	//	Front(glm::vec3(0.0f, 0.0f, -1.0f)),
	//	// MovementSpeed(SPEED), MouseSensitivity(SENSITIVTY), 
	//	Zoom(ZOOM)
	//{
	//	this->Position = glm::vec3(posX, posY, posZ);
	//	this->WorldUp = glm::vec3(upX, upY, upZ);
	//	this->Yaw = yaw;
	//	this->Pitch = pitch;
	//	this->updateCameraVectors();
	//}

	// Returns the view matrix calculated using Eular Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix() const
	{
		return glm::lookAt(this->Position, this->Position + this->Front, this->Up);
	}

	GLfloat getZoom() const { return Zoom; }
	
	glm::vec3 getPosition() const { return Position; }

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void move(Camera_Movement direction, GLfloat distance)
	{
		if (direction == FORWARD)
			this->Position += this->Front * distance;
		if (direction == BACKWARD)
			this->Position -= this->Front * distance;
		if (direction == LEFT)
			this->Position -= this->Right * distance;
		if (direction == RIGHT)
			this->Position += this->Right * distance;
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void glance(GLfloat scrollX, GLfloat scrollY, GLboolean constrainPitch = true)
	{
		//scrollX *= distance;
		//scrollY *= distance;

		this->Yaw += scrollX;
		this->Pitch += scrollY;

		//std::cout << "Camera #" << Camera::CameraNo << " deltax = " << scrollX << " deltay = " << scrollY << std::endl;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (this->Pitch > 89.0f)
				this->Pitch = 89.0f;
			if (this->Pitch < -89.0f)
				this->Pitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Eular angles
		this->updateCameraVectors();
	}

	// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void focus(GLfloat offsetY)
	{
		if (this->Zoom >= 1.0f && this->Zoom <= 90.0f)
			this->Zoom -= offsetY;
		if (this->Zoom <= 1.0f)
			this->Zoom = 1.0f;
		if (this->Zoom >= 89.0f)
			this->Zoom = 89.0f;
	}

	GLfloat getAspectRatio() const
	{
		return AspectRatio;
	}

protected:
	//static int CameraNo;
	// Camera Attributes
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	// Eular Angles
	GLfloat Yaw;
	GLfloat Pitch;
	// Camera options
	//GLfloat MovementSpeed;
	//GLfloat MouseSensitivity;
	GLfloat Zoom;
	GLfloat AspectRatio;

	// Calculates the front vector from the Camera's (updated) Eular Angles
	void updateCameraVectors()
	{
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(this->Yaw)) * cos(glm::radians(this->Pitch));
		front.y = sin(glm::radians(this->Pitch));
		front.z = sin(glm::radians(this->Yaw)) * cos(glm::radians(this->Pitch));
		this->Front = glm::normalize(front);
		// Also re-calculate the Right and Up vector
		this->Right = glm::normalize(glm::cross(this->Front, this->WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
		this->Up = glm::normalize(glm::cross(this->Right, this->Front));
	}
};

#endif