#ifndef MESH_H
#define MESH_H

#define CGAL_EIGEN3_ENABLED

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>

#include <Eigen\Core>
#include <assimp\mesh.h>

#include <string>

// kernel type
using Kernelf = CGAL::Simple_cartesian<float>;
// primitive types
using FT = Kernelf::FT;
using Point3f = Kernelf::Point_3;
using Vec2f = Kernelf::Vector_2;
using Vec3f = Kernelf::Vector_3;
using Triangle3f = Kernelf::Triangle_3;
using Segment3f = Kernelf::Segment_3;
using Plane3f = Kernelf::Plane_3;
using AffTransformf = Kernelf::Aff_transformation_3;
// surface mesh type
using SurfaceMesh3f = CGAL::Surface_mesh<Point3f>;
// primitive index types
using Veridx = SurfaceMesh3f::Vertex_index;
using Faceidx = SurfaceMesh3f::Face_index;
using Edgeidx = SurfaceMesh3f::Edge_index;
using Halfedgeidx = SurfaceMesh3f::Halfedge_index;
// primitive iterator types
using Veriter = SurfaceMesh3f::Vertex_iterator;
using Edgeiter = SurfaceMesh3f::Edge_iterator;
using Faceiter = SurfaceMesh3f::Face_iterator;

class DefPropName
{
public:
	static const std::string vUVs;
	static const unsigned int vUVs_op = 0x01;
	static const std::string vNormals;
	static const unsigned int vNormals_op = 0x02;
	static const std::string vColors;
	static const unsigned int vColors_op = 0x04;
	static const std::string vHandles;
	static const unsigned int vHandles_op = 0x10;
	static const std::string fNormals;
	static const unsigned int fNormals_op = 0x08;
};

class SurfaceMeshObject
{
public:
	SurfaceMeshObject(unsigned int propMask = 0x0) :
		m_propMask(propMask)
	{
		/* initialize property maps */
		if (DefPropName::vNormals_op && m_propMask != 0x0)
			m_vNormals = m_mesh.add_property_map<Veridx, Vec3f>(DefPropName::vNormals).first;
		if (DefPropName::vUVs_op && m_propMask != 0x0)
			m_vUVs = m_mesh.add_property_map<Veridx, Vec2f>(DefPropName::vUVs).first;
		if (DefPropName::vColors_op && m_propMask != 0x0)
			m_vColors = m_mesh.add_property_map<Veridx, Vec3f>(DefPropName::vColors).first;
		if (DefPropName::vHandles_op && m_propMask != 0x0)
			m_vHandles = m_mesh.add_property_map<Veridx, int>(DefPropName::vHandles).first;
		if (DefPropName::fNormals_op && m_propMask != 0x0)
			m_fNormals = m_mesh.add_property_map<Faceidx, Vec3f>(DefPropName::fNormals).first;
	}

	bool computeNormals()
	{
		unsigned int mask = DefPropName::vNormals_op | DefPropName::fNormals_op;
		if ((mask & m_propMask) != mask)
			return false;
		CGAL::Polygon_mesh_processing::compute_normals(m_mesh, m_vNormals, m_fNormals,
			CGAL::Polygon_mesh_processing::parameters::vertex_point_map(m_mesh.points()).geom_traits(Kernelf()));
		return true;
	}

	void affineTransform(AffTransformf const & aff)
	{
		for (auto & vid : m_mesh.vertices())
			m_mesh.point(vid) = aff.transform(m_mesh.point(vid));
	}

	void remesh(float edgeLength, int iterNum)
	{
		CGAL::Polygon_mesh_processing::isotropic_remeshing(
			CGAL::faces(m_mesh), double(edgeLength), m_mesh,
			CGAL::Polygon_mesh_processing::parameters::number_of_iterations(iterNum));
	}

	template <typename V3f>
	bool resetHandledPositions(std::vector<V3f> const & positions)
	{
		if (DefPropName::vHandles_op && m_propMask == 0x0) return false;
		for (auto & vid : m_mesh.vertices())
		{
			auto const & pos = positions[m_vHandles[vid]];
			m_mesh.point(vid) = Point3f{ pos.x(), pos.y(), pos.z() };
		}
		return true;
	}

	static std::unique_ptr<SurfaceMeshObject> importSurfaceMesh(aiMesh const * aimesh)
	{
		auto res = std::make_unique<SurfaceMeshObject>(DefPropName::vNormals_op | DefPropName::fNormals_op);
		auto & mesh = res.get()->getMesh();

		std::cout << "\t Vertices " << aimesh->mNumVertices << std::endl;
		std::cout << "\t Faces " << aimesh->mNumFaces << std::endl;

		Veridx * indiceVector = new Veridx[aimesh->mNumVertices];
		// Walk through each of the mesh's vertices
		for (unsigned int i = 0; i < aimesh->mNumVertices; i++)
		{
			auto vid = mesh.add_vertex();
			indiceVector[i] = vid;
			mesh.point(vid) = Point3f(aimesh->mVertices[i].x, aimesh->mVertices[i].y, aimesh->mVertices[i].z);

			// TODO: Texture Coordinates with property mask
			//if (aimesh->mTextureCoords[0]) // Does the mesh contain texture coordinates?
			//	mesh.m_textureCoor[vid] = Eigen::Vector2f(aimesh->mTextureCoords[0][i].x, aimesh->mTextureCoords[0][i].y);
		}
		// Now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
		for (unsigned int i = 0; i < aimesh->mNumFaces; i++)
		{
			aiFace face = aimesh->mFaces[i];
			// Retrieve all indices of the face and store them in the indices vector
			std::vector<Veridx> indices;
			for (unsigned int j = 0; j < face.mNumIndices; j++)
				indices.push_back(indiceVector[face.mIndices[j]]);
			mesh.add_face(indices);
		}
		return res;
	}

private:
	SurfaceMesh3f m_mesh;

public:
	unsigned int m_propMask;
	SurfaceMesh3f::Property_map<Veridx, Vec3f> m_vNormals;
	SurfaceMesh3f::Property_map<Veridx, Vec2f> m_vUVs;
	SurfaceMesh3f::Property_map<Veridx, Vec3f> m_vColors;
	SurfaceMesh3f::Property_map<Veridx, int> m_vHandles;
	SurfaceMesh3f::Property_map<Faceidx, Vec3f> m_fNormals;

	SurfaceMesh3f & getMesh() { return m_mesh; }
	SurfaceMesh3f const & getMesh() const { return m_mesh; }
	SurfaceMesh3f::Property_map<Veridx, Vec3f> const getVNormals() const { return m_vNormals; }
	SurfaceMesh3f::Property_map<Veridx, Vec2f> const getVUVs() const { return m_vUVs; }
	SurfaceMesh3f::Property_map<Veridx, Vec3f> const getVColors() const { return m_vColors; }
	SurfaceMesh3f::Property_map<Veridx, int> const getVHandles() const { return m_vHandles; }
	SurfaceMesh3f::Property_map<Faceidx, Vec3f> const getFNormals() const { return m_fNormals; }

public:

	std::vector<Point3f> vPositions() const 
	{
		std::vector<Point3f> res;
		res.reserve(m_mesh.number_of_vertices());
		for (auto vid : m_mesh.vertices())
			res.push_back(m_mesh.point(vid));
		return res;
	}

	std::vector<unsigned int> vElements() const
	{
		auto fnum = m_mesh.number_of_faces();
		auto vnum = m_mesh.number_of_vertices();
		std::map<Veridx, unsigned int> vid2index;
		int p = 0;
		for (auto vid : m_mesh.vertices())
			vid2index[vid] = p++;

		std::vector<unsigned int> res;
		res.reserve(fnum);
		for (auto fid : m_mesh.faces())
		{
			CGAL::Vertex_around_face_circulator<SurfaceMesh3f> vbegin(m_mesh.halfedge(fid), m_mesh), vend(vbegin);
			do
				res.push_back(vid2index.at(*vbegin++));
			while (vbegin != vend);
		}

		return res;
	}

	std::vector<Vec3f> vNormals() const
	{
		std::vector<Vec3f> res;
		res.reserve(m_mesh.number_of_vertices());
		for (auto vid : m_mesh.vertices())
			res.push_back(m_vNormals[vid]);
		return res;
	}


};



#endif