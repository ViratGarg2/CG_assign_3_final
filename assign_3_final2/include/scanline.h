#ifndef SCANLINE_H
#define SCANLINE_H

#include <vector>
#include <glm/glm.hpp>
#include "geometry/Point3D.h"

struct Edge {
    float yMax;
    float xMin;
    float slope;
    float yMin;
    glm::vec3 color;
};

struct ActiveEdge {
    float x;
    float yMax;
    float slope;
    glm::vec3 color;
};

void initializeEdgeTable(
    const std::vector<Point3D>& customVertices,
    std::vector<std::vector<Edge>>& edgeTable,
    int& scanlineMinY,
    int& scanlineMaxY
);

void scanlineFill(
    std::vector<std::vector<Edge>>& edgeTable,
    std::vector<ActiveEdge>& activeEdgeTable,
    int scanlineMinY,
    int scanlineMaxY,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices,
    void (*updateBuffers)()
);

#endif // SCANLINE_H
