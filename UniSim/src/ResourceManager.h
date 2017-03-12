#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include "OpenGLContext.h"

#include "Texture.h"
#include "Shader.h"
#include "Mesh.h"

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class ResourceManager
{
public:
	ResourceManager() = delete;
	
	static void LoadShader(std::string const & name, std::string const & vs, std::string const & fs, std::string const & gs);
	static Shader * GetShader(std::string const & name);

	static void LoadTexture(std::string const & name, std::string const & file, GLboolean alpha);
	static Texture2D * GetTexture(std::string const & name);

	static void LoadMeshes(std::string const & name, std::string const & file,
		aiPostProcessSteps options = aiPostProcessSteps(aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices),
		aiComponent remove_components = aiComponent(aiComponent_NORMALS));
	static std::vector<SurfaceMeshObject *> GetMesh(std::string const & name);

private:
	static std::map<std::string, std::unique_ptr<Shader>> Shaders;
	static std::map<std::string, std::unique_ptr<Texture2D>> Textures;
	static std::map<std::string, std::vector<std::unique_ptr<SurfaceMeshObject>>> Meshes;

	//static std::unique_ptr<Shader> ResourceManager::loadShaderFromFile(std::string const & vShaderFile, std::string const & fShaderFile, std::string const & gShaderFile);

	//static std::unique_ptr<Texture2D> loadTextureFromFile(std::string const & file, GLboolean alpha);
	//
	//static std::vector<aiMesh *> loadMeshesFromDir(std::string const & dir, aiPostProcessSteps options, aiComponent remove_components);

};

#endif