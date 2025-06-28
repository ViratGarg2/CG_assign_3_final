#ifndef MESH_H
#define MESH_H

#include <vector>
#include <glm/glm.hpp>
#include "OFFReader.h"
#include "geometry/Plane.h"
#include <GL/glew.h>

void prepareMeshData(
    const OffModel* model,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices,
    std::vector<float>& originalVertices
);

void CreateMeshBuffers(
    GLuint& VAO,
    GLuint& VBO,
    GLuint& normalVBO,
    GLuint& colorVBO,
    GLuint& IBO,
    const std::vector<float>& meshVertices,
    const std::vector<float>& meshNormals,
    const std::vector<float>& meshColors,
    const std::vector<unsigned int>& meshIndices
);

void sliceMesh(
    const Plane& plane,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices
);

#endif // MESH_H
