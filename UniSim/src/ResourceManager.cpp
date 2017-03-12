/*******************************************************************
** This code is part of Breakout.
**
** Breakout is free software: you can redistribute it and/or modify
** it under the terms of the CC BY 4.0 license as published by
** Creative Commons, either version 4 of the License, or (at your
** option) any later version.
******************************************************************/
#include "ResourceManager.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <SOIL.h>

// Instantiate static physics
std::map<std::string, std::unique_ptr<Shader>> ResourceManager::Shaders;
std::map<std::string, std::unique_ptr<Texture2D>> ResourceManager::Textures;
std::map<std::string, std::vector<std::unique_ptr<SurfaceMeshObject>>> ResourceManager::Meshes;


/******************************************************/
std::unique_ptr<Shader> loadShaderFromFile(std::string const & vShaderFile, std::string const & fShaderFile, std::string const & gShaderFile)
{
	// 1. Retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::string geometryCode;
	try
	{
		// Open files
		std::ifstream vertexShaderFile(vShaderFile);
		std::ifstream fragmentShaderFile(fShaderFile);
		std::stringstream vShaderStream, fShaderStream;
		// Read file's buffer contents into streams
		vShaderStream << vertexShaderFile.rdbuf();
		fShaderStream << fragmentShaderFile.rdbuf();
		// close file handlers
		vertexShaderFile.close();
		fragmentShaderFile.close();
		// Convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
		// If geometry shader path is present, also load a geometry shader
		if (!gShaderFile.empty())
		{
			std::ifstream geometryShaderFile(gShaderFile);
			std::stringstream gShaderStream;
			gShaderStream << geometryShaderFile.rdbuf();
			geometryShaderFile.close();
			geometryCode = gShaderStream.str();
		}
	}
	catch (std::exception e)
	{
		std::cout << "ERROR::SHADER: Failed to read shader files" << std::endl;
	}
#ifdef DEBUG_SHADER
	std::cout << "Vertex src " << std::endl << vertexCode << std::endl;
	std::cout << "Fragment src " << std::endl << fragmentCode << std::endl;
	if (gShaderFile != nullptr) std::cout << "Geometry src " << std::endl << geometryCode << std::endl;
#endif
	const GLchar *vShaderCode = vertexCode.c_str();
	const GLchar *fShaderCode = fragmentCode.c_str();
	const GLchar *gShaderCode = geometryCode.c_str();
	// 2. Now create shader object from source code
	auto shader = std::make_unique<Shader>();
	shader->Compile(vShaderCode, fShaderCode, (!gShaderFile.empty()) ? gShaderCode : nullptr);
	return shader;
}

std::unique_ptr<Texture2D> loadTextureFromFile(std::string const & file, GLboolean alpha)
{
	// Create Texture object
	auto texture = std::make_unique<Texture2D>();
	if (alpha)
	{
		texture->Internal_Format = GL_RGBA;
		texture->Image_Format = GL_RGBA;
	}
	// Load image
	int width, height;
	unsigned char* image = SOIL_load_image(file.c_str(), &width, &height, 0, texture->Image_Format == GL_RGBA ? SOIL_LOAD_RGBA : SOIL_LOAD_RGB);
	// Now generate texture
	texture->Generate(width, height, image);
	// And finally free image data
	SOIL_free_image_data(image);
	return texture;
}

void processNode(std::vector<std::unique_ptr<SurfaceMeshObject>> & res, aiNode* node, const aiScene* scene)
{
	for (int i = 0; i < node->mNumMeshes; i++)
	{
		auto & aimesh = scene->mMeshes[node->mMeshes[i]];
		res.push_back(SurfaceMeshObject::importSurfaceMesh(aimesh));
	}
	for (unsigned int i = 0; i < node->mNumChildren; i++)
		processNode(res, node->mChildren[i], scene);
}

std::vector<std::unique_ptr<SurfaceMeshObject>> loadMeshesFromDir(std::string const & dir, aiPostProcessSteps options, aiComponent remove_components)
{
	std::cout << "INFO::LOAD SCENE:: " << dir.c_str() << std::endl;

	// Read file via ASSIMP
	Assimp::Importer importer;
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, remove_components);
	const aiScene* scene = importer.ReadFile(dir, options);
	//const aiScene* scene = importer.ReadFile(path, /*aiProcess_Triangulate | */aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);
	// Check for errors
	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
	{
		std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
		return{};
	}
	std::vector<std::unique_ptr<SurfaceMeshObject>> res;
	// Process ASSIMP's root node recursively
	processNode(res, scene->mRootNode, scene);
	return res;
}

/******************************************************/


void ResourceManager::LoadShader(std::string const & name, std::string const & vs, std::string const & fs, std::string const & gs)
{
	std::cout << "INFO::LOAD SHADER: " << name.c_str() << std::endl;
	Shaders[name] = loadShaderFromFile(vs, fs, gs);
}

Shader  *  ResourceManager::GetShader(std::string const & name)
{
	return Shaders.at(name).get();
}

void ResourceManager::LoadTexture(std::string const & name, std::string const & file, GLboolean alpha)
{
	std::cout << "INFO::LOAD TEXTURE: " << name.c_str() << std::endl;
	Textures[name] = loadTextureFromFile(file, alpha);
}

Texture2D  *  ResourceManager::GetTexture(std::string const & name)
{
	return Textures.at(name).get();
}


void ResourceManager::LoadMeshes(std::string const & name, std::string const & dir, aiPostProcessSteps options, aiComponent remove_components)
{
	std::cout << "INFO::LOAD SCENE: " << name.c_str() << std::endl;
	Meshes[name] = loadMeshesFromDir(dir, options, remove_components); 
}

std::vector<SurfaceMeshObject *> ResourceManager::GetMesh(std::string const & name)
{
	std::vector<SurfaceMeshObject *> res;
	for (auto & m : Meshes[name])
		res.push_back(m.get());
	return res;
}

