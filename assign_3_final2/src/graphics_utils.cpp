#include "graphics_utils.h"
#include <algorithm> // For std::swap
#include <cmath>     // For abs

void drawLine2DBresenham(float x1, float y1, float x2, float y2,
                         std::vector<float>& lineVertices,
                         std::vector<unsigned int>& lineIndices,
                         GLuint& lineVAO, GLuint& lineVBO, GLuint& lineIBO) {
    // Clear any previous line data
    lineVertices.clear();
    lineIndices.clear();
    
    // Use higher scale for better precision
    const int scale = 100;
    int ix1 = static_cast<int>(x1 * scale);
    int iy1 = static_cast<int>(y1 * scale);
    int ix2 = static_cast<int>(x2 * scale);
    int iy2 = static_cast<int>(y2 * scale);
    
    // Determine if the line is steep (|y2-y1| > |x2-x1|)
    bool steep = abs(iy2 - iy1) > abs(ix2 - ix1);
    
    // If line is steep, swap x and y coordinates
    if (steep) {
        std::swap(ix1, iy1);
        std::swap(ix2, iy2);
    }
    
    // If line goes from right to left, swap endpoints
    if (ix1 > ix2) {
        std::swap(ix1, ix2);
        std::swap(iy1, iy2);
    }
    
    int dx = ix2 - ix1;
    int dy = abs(iy2 - iy1);
    int error = dx / 2;
    
    int y = iy1;
    int ystep = (iy1 < iy2) ? 1 : -1;
    
    // Add points to the line
    for (int x = ix1; x <= ix2; x++) {
        float vertX, vertY;
        
        if (steep) {
            vertX = static_cast<float>(y) / scale;
            vertY = static_cast<float>(x) / scale;
        } else {
            vertX = static_cast<float>(x) / scale;
            vertY = static_cast<float>(y) / scale;
        }
        
        // Add vertex coordinates
        lineVertices.push_back(vertX);
        lineVertices.push_back(vertY);
        lineVertices.push_back(0.0f);
        
        // Add color (red)
        lineVertices.push_back(1.0f);
        lineVertices.push_back(0.0f);
        lineVertices.push_back(0.0f);
        
        // Add normal
        lineVertices.push_back(0.0f);
        lineVertices.push_back(0.0f);
        lineVertices.push_back(1.0f);
        
        // Add index
        lineIndices.push_back(lineIndices.size());
        
        error -= dy;
        if (error < 0) {
            y += ystep;
            error += dx;
        }
    }
    
    // Delete previous OpenGL buffers if they exist
    if (lineVAO != 0) {
        glDeleteVertexArrays(1, &lineVAO);
    }
    if (lineVBO != 0) {
        glDeleteBuffers(1, &lineVBO);
    }
    if (lineIBO != 0) {
        glDeleteBuffers(1, &lineIBO);
    }
    
    // Create new buffers
    glGenVertexArrays(1, &lineVAO);
    glBindVertexArray(lineVAO);
    
    glGenBuffers(1, &lineVBO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, lineVertices.size() * sizeof(float), lineVertices.data(), GL_STATIC_DRAW);
    
    // Set up vertex attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    
    glGenBuffers(1, &lineIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, lineIndices.size() * sizeof(unsigned int), lineIndices.data(), GL_STATIC_DRAW);
    
    // Unbind VAO
    glBindVertexArray(0);
}
