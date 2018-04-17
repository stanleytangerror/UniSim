#ifndef Renderer_h
#define Renderer_h

#include "Screen.h"
#include "Shader.h"
#include "Camera.h"
#include "Mesh.h"

class SurMeshObjRenderer
{
public:
	SurMeshObjRenderer(SurfaceMeshObject * meshobj, Shader * shader, FreeCamera  * camera) :
		meshobj(meshobj), camera(camera), shader(shader)
	{
		glGenVertexArrays(1, &meshVAO);
		std::cout << "mesh vao " << meshVAO << std::endl;
		glGenBuffers(1, &meshVBO);
		std::cout << "mesh vbo " << meshVBO << std::endl;
		glGenBuffers(1, &meshVNormalBO);
		std::cout << "mesh vbNormalo " << meshVNormalBO << std::endl;
		glGenBuffers(1, &meshEBO);
		std::cout << "mesh ebo " << meshEBO << std::endl;
	}

	void draw(float r, float g, float b) const
	{
		glDepthMask(GL_TRUE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		/* ------- draw cloth piece ------- */
		shader->Use();

		glUniformMatrix4fv(glGetUniformLocation(shader->Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(glGetUniformLocation(shader->Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shader->Program, "model"), 1, GL_FALSE, glm::value_ptr(model));

		glUniform3f(glGetUniformLocation(shader->Program, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniform3f(glGetUniformLocation(shader->Program, "light.direction"), 0.0f, -0.707f, -0.707f);
		glUniform3f(glGetUniformLocation(shader->Program, "light.color"), 0.75f, 0.75f, 0.75f);
		/* from http://devernay.free.fr/cours/opengl/materials.html */
		glUniform3f(glGetUniformLocation(shader->Program, "material.ambient"), 0.6f * r, 0.6f * g, 0.6f * b);
		glUniform3f(glGetUniformLocation(shader->Program, "material.diffuse"), 0.6f * r, 0.6f * g, 0.6f * b);
		glUniform3f(glGetUniformLocation(shader->Program, "material.specular"), 0.1f * r, 0.1f * g, 0.1f * b);
		glUniform1f(glGetUniformLocation(shader->Program, "material.shininess"), 0.07f);

		glBindVertexArray(meshVAO);
		{
			glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
			glBufferData(GL_ARRAY_BUFFER, vercnt * 3 * sizeof(GLfloat), (GLfloat *)positions.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0 * sizeof(GLfloat), (GLvoid *)0);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindBuffer(GL_ARRAY_BUFFER, meshVNormalBO);
			glBufferData(GL_ARRAY_BUFFER, vercnt * 3 * sizeof(GLfloat), (GLfloat *)normals.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0 * sizeof(GLfloat), (GLvoid *)0);
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, elemcnt * sizeof(GLuint), (GLuint *)elements.data(), GL_STATIC_DRAW);
			glDrawElements(GL_TRIANGLES, elemcnt, GL_UNSIGNED_INT, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		}
		glBindVertexArray(0);

	}

	void update()
	{
		projection = glm::perspective(camera->getZoom(), camera->getAspectRatio(), 0.1f, 1000.0f);
		view = camera->GetViewMatrix();
		model = glm::scale(glm::mat4(), glm::vec3(0.20f, 0.20f, 0.20f));
		model = glm::translate(model, glm::vec3(0.0f, 0.40f, 0.0f)); // Translate it down a bit so it's at the center of the scene

		viewPos = camera->getPosition();

		positions = meshobj->vPositions();
		normals = meshobj->vNormals();
		elements = meshobj->vElements();

		vercnt = positions.size();
		elemcnt = elements.size();
	}

private:
	SurfaceMeshObject * meshobj;
	FreeCamera * camera;
	Shader * shader;
	
	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 model;
	glm::vec3 viewPos;

	std::vector<Point3f> positions;
	std::vector<Vec3f> normals;
	std::vector<unsigned int> elements;

	size_t vercnt, elemcnt;

	GLuint meshVAO, meshVBO, meshVNormalBO, meshEBO, conditionVBO;

};

#endif