#ifndef GRAPHICS_UTILS_H
#define GRAPHICS_UTILS_H

#include <vector>
#include <GL/glew.h>

void drawLine2DBresenham(float x1, float y1, float x2, float y2,
                         std::vector<float>& lineVertices,
                         std::vector<unsigned int>& lineIndices,
                         GLuint& lineVAO, GLuint& lineVBO, GLuint& lineIBO);

#endif
