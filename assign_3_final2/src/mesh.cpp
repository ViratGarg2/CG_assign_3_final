#include "mesh.h"
#include <vector>
#include <glm/glm.hpp>
#include "OFFReader.h"
#include "geometry/Plane.h"
#include <GL/glew.h>
#include <algorithm>

// Forward declaration for a helper function used by sliceMesh
void createCapTriangles(const std::vector<glm::vec3>& points, const std::vector<glm::vec3>& normals,const glm::vec3& color,const glm::vec3& planeNormal,std::vector<float>& outVertices,std::vector<float>& outNormals,std::vector<float>& outColors,std::vector<unsigned int>& outIndices,bool isPositiveSide,float gap);

void prepareMeshData(
    const OffModel* model,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices,
    std::vector<float>& originalVertices
) {
    meshVertices.clear();
    meshNormals.clear();
    meshColors.clear();
    meshIndices.clear();

    for (int i = 0; i < model->numberOfVertices; i++) {
        meshVertices.push_back(model->vertices[i].x);
        meshVertices.push_back(model->vertices[i].y);
        meshVertices.push_back(model->vertices[i].z);

        meshNormals.push_back(model->vertices[i].normal.x);
        meshNormals.push_back(model->vertices[i].normal.y);
        meshNormals.push_back(model->vertices[i].normal.z);

        float r = 1.0f;
        float g = 0.0f;
        float b = 0.0f;

        meshColors.push_back(r);
        meshColors.push_back(g);
        meshColors.push_back(b);
    }

    for (int i = 0; i < model->numberOfPolygons; i++) {
        if (model->polygons[i].noSides >= 3) {
            for (int j = 1; j < model->polygons[i].noSides - 1; j++) {
                meshIndices.push_back(model->polygons[i].v[0]);
                meshIndices.push_back(model->polygons[i].v[j]);
                meshIndices.push_back(model->polygons[i].v[j + 1]);
            }
        }
    }

    originalVertices = meshVertices;
}

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
) {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, meshVertices.size() * sizeof(float), meshVertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &normalVBO);
    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
    glBufferData(GL_ARRAY_BUFFER, meshNormals.size() * sizeof(float), meshNormals.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, meshColors.size() * sizeof(float), meshColors.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void sliceMesh(
    const Plane& plane,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices
) {
    std::vector<float> newVertices;
    std::vector<float> newNormals;
    std::vector<float> newColors;
    std::vector<unsigned int> newIndices;
    
    std::vector<glm::vec3> posIntersectionPoints;
    std::vector<glm::vec3> negIntersectionPoints;
    std::vector<glm::vec3> posIntersectionNormals;
    std::vector<glm::vec3> negIntersectionNormals;
    
    for (size_t i = 0; i < meshIndices.size(); i += 3) {
        std::vector<glm::vec3> triangleVerts;
        std::vector<glm::vec3> triangleNorms;
        std::vector<glm::vec3> triangleColors;
        std::vector<float> distances;
        
        for (int j = 0; j < 3; j++) {
            unsigned int idx = meshIndices[i + j];
            glm::vec3 vert(meshVertices[idx * 3], meshVertices[idx * 3 + 1], meshVertices[idx * 3 + 2]);
            glm::vec3 norm(meshNormals[idx * 3], meshNormals[idx * 3 + 1], meshNormals[idx * 3 + 2]);
            glm::vec3 color(meshColors[idx * 3], meshColors[idx * 3 + 1], meshColors[idx * 3 + 2]);
            
            triangleVerts.push_back(vert);
            triangleNorms.push_back(norm);
            triangleColors.push_back(color);
            distances.push_back(plane.getSignedDistance(vert));
        }
        
        int positiveSide = 0, negativeSide = 0;
        for (float d : distances) {
            if (d > 0) positiveSide++;
            else if (d < 0) negativeSide++;
        }
        
        if (positiveSide == 3 || negativeSide == 3) {
            unsigned int baseIndex = newVertices.size() / 3;
            
            for (int j = 0; j < 3; j++) {
                glm::vec3 vert = triangleVerts[j];
                
                if (positiveSide == 3) { 
                    vert += plane.normal * (plane.gap * 0.5f);
                } else { 
                    vert -= plane.normal * (plane.gap * 0.5f);
                }
                
                newVertices.push_back(vert.x);
                newVertices.push_back(vert.y);
                newVertices.push_back(vert.z);
                
                newNormals.push_back(triangleNorms[j].x);
                newNormals.push_back(triangleNorms[j].y);
                newNormals.push_back(triangleNorms[j].z);
                
                newColors.push_back(triangleColors[j].x);
                newColors.push_back(triangleColors[j].y);
                newColors.push_back(triangleColors[j].z);
            }
            
            newIndices.push_back(baseIndex);
            newIndices.push_back(baseIndex + 1);
            newIndices.push_back(baseIndex + 2);
            
            continue;
        }
        
        std::vector<glm::vec3> intersections;
        std::vector<glm::vec3> intersectionNorms;
        std::vector<glm::vec3> intersectionColors;
        
        for (int j = 0; j < 3; j++) {
            int k = (j + 1) % 3;
            
            if (distances[j] * distances[k] >= 0.0f && !(distances[j] == 0.0f || distances[k] == 0.0f)) {
                continue;
            }
            
            float t;
            if (distances[j] == 0.0f) {
                t = 0.0f;
            } else if (distances[k] == 0.0f) {
                t = 1.0f;
            } else {
                t = distances[j] / (distances[j] - distances[k]);
            }
            
            glm::vec3 intersection = triangleVerts[j] + t * (triangleVerts[k] - triangleVerts[j]);
            glm::vec3 normal = glm::normalize((1.0f - t) * triangleNorms[j] + t * triangleNorms[k]);
            glm::vec3 color = (1.0f - t) * triangleColors[j] + t * triangleColors[k];
            
            intersections.push_back(intersection);
            intersectionNorms.push_back(normal);
            intersectionColors.push_back(color);
            
            if (positiveSide > 0 && negativeSide > 0) {  
                posIntersectionPoints.push_back(intersection + plane.normal * (plane.gap * 0.5f));
                negIntersectionPoints.push_back(intersection - plane.normal * (plane.gap * 0.5f));
                posIntersectionNormals.push_back(normal);
                negIntersectionNormals.push_back(normal);
            }
        }
        
        if (intersections.size() == 2) {
            std::vector<int> posIndices, negIndices;
            
            for (int j = 0; j < 3; j++) {
                if (distances[j] > 0) {
                    posIndices.push_back(j);
                } else if (distances[j] < 0) {
                    negIndices.push_back(j);
                } else {
                    posIndices.push_back(j);
                    negIndices.push_back(j);
                }
            }
            
            if (posIndices.size() > 0) {
                if (posIndices.size() == 1) {
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    glm::vec3 posVert = triangleVerts[posIndices[0]] + plane.normal * (plane.gap * 0.5f);
                    newVertices.push_back(posVert.x);
                    newVertices.push_back(posVert.y);
                    newVertices.push_back(posVert.z);
                    
                    newNormals.push_back(triangleNorms[posIndices[0]].x);
                    newNormals.push_back(triangleNorms[posIndices[0]].y);
                    newNormals.push_back(triangleNorms[posIndices[0]].z);
                    
                    newColors.push_back(triangleColors[posIndices[0]].x);
                    newColors.push_back(triangleColors[posIndices[0]].y);
                    newColors.push_back(triangleColors[posIndices[0]].z);
                    
                    for (size_t k = 0; k < 2; k++) {
                        glm::vec3 intPoint = intersections[k] + plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(intPoint.x);
                        newVertices.push_back(intPoint.y);
                        newVertices.push_back(intPoint.z);
                        
                        newNormals.push_back(intersectionNorms[k].x);
                        newNormals.push_back(intersectionNorms[k].y);
                        newNormals.push_back(intersectionNorms[k].z);
                        
                        newColors.push_back(intersectionColors[k].x);
                        newColors.push_back(intersectionColors[k].y);
                        newColors.push_back(intersectionColors[k].z);
                    }
                    
                    newIndices.push_back(baseIndex);
                    newIndices.push_back(baseIndex + 1);
                    newIndices.push_back(baseIndex + 2);
                } else if (posIndices.size() == 2) {
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    for (int idx : posIndices) {
                        glm::vec3 posVert = triangleVerts[idx] + plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(posVert.x);
                        newVertices.push_back(posVert.y);
                        newVertices.push_back(posVert.z);
                        
                        newNormals.push_back(triangleNorms[idx].x);
                        newNormals.push_back(triangleNorms[idx].y);
                        newNormals.push_back(triangleNorms[idx].z);
                        
                        newColors.push_back(triangleColors[idx].x);
                        newColors.push_back(triangleColors[idx].y);
                        newColors.push_back(triangleColors[idx].z);
                    }
                    
                    for (size_t k = 0; k < 2; k++) {
                        glm::vec3 intPoint = intersections[k] + plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(intPoint.x);
                        newVertices.push_back(intPoint.y);
                        newVertices.push_back(intPoint.z);
                        
                        newNormals.push_back(intersectionNorms[k].x);
                        newNormals.push_back(intersectionNorms[k].y);
                        newNormals.push_back(intersectionNorms[k].z);
                        
                        newColors.push_back(intersectionColors[k].x);
                        newColors.push_back(intersectionColors[k].y);
                        newColors.push_back(intersectionColors[k].z);
                    }
                    
                    newIndices.push_back(baseIndex);     
                    newIndices.push_back(baseIndex + 1); 
                    newIndices.push_back(baseIndex + 2); 
                    
                    newIndices.push_back(baseIndex + 1); 
                    newIndices.push_back(baseIndex + 3); 
                    newIndices.push_back(baseIndex + 2); 
                }
            }
            
            if (negIndices.size() > 0) {
                if (negIndices.size() == 1) {
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    glm::vec3 negVert = triangleVerts[negIndices[0]] - plane.normal * (plane.gap * 0.5f);
                    newVertices.push_back(negVert.x);
                    newVertices.push_back(negVert.y);
                    newVertices.push_back(negVert.z);
                    
                    newNormals.push_back(triangleNorms[negIndices[0]].x);
                    newNormals.push_back(triangleNorms[negIndices[0]].y);
                    newNormals.push_back(triangleNorms[negIndices[0]].z);
                    
                    newColors.push_back(triangleColors[negIndices[0]].x);
                    newColors.push_back(triangleColors[negIndices[0]].y);
                    newColors.push_back(triangleColors[negIndices[0]].z);
                    
                    for (int k = 1; k >= 0; k--) {
                        glm::vec3 intPoint = intersections[k] - plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(intPoint.x);
                        newVertices.push_back(intPoint.y);
                        newVertices.push_back(intPoint.z);
                        
                        newNormals.push_back(intersectionNorms[k].x);
                        newNormals.push_back(intersectionNorms[k].y);
                        newNormals.push_back(intersectionNorms[k].z);
                        
                        newColors.push_back(intersectionColors[k].x);
                        newColors.push_back(intersectionColors[k].y);
                        newColors.push_back(intersectionColors[k].z);
                    }
                    
                    newIndices.push_back(baseIndex);
                    newIndices.push_back(baseIndex + 1);
                    newIndices.push_back(baseIndex + 2);
                } else if (negIndices.size() == 2) {
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    for (int idx : negIndices) {
                        glm::vec3 negVert = triangleVerts[idx] - plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(negVert.x);
                        newVertices.push_back(negVert.y);
                        newVertices.push_back(negVert.z);
                        
                        newNormals.push_back(triangleNorms[idx].x);
                        newNormals.push_back(triangleNorms[idx].y);
                        newNormals.push_back(triangleNorms[idx].z);
                        
                        newColors.push_back(triangleColors[idx].x);
                        newColors.push_back(triangleColors[idx].y);
                        newColors.push_back(triangleColors[idx].z);
                    }
                    
                    for (int k = 1; k >= 0; k--) {
                        glm::vec3 intPoint = intersections[k] - plane.normal * (plane.gap * 0.5f);
                        newVertices.push_back(intPoint.x);
                        newVertices.push_back(intPoint.y);
                        newVertices.push_back(intPoint.z);
                        
                        newNormals.push_back(intersectionNorms[k].x);
                        newNormals.push_back(intersectionNorms[k].y);
                        newNormals.push_back(intersectionNorms[k].z);
                        
                        newColors.push_back(intersectionColors[k].x);
                        newColors.push_back(intersectionColors[k].y);
                        newColors.push_back(intersectionColors[k].z);
                    }
                    
                    newIndices.push_back(baseIndex);
                    newIndices.push_back(baseIndex + 1);
                    newIndices.push_back(baseIndex + 2);
                    
                    newIndices.push_back(baseIndex + 1);
                    newIndices.push_back(baseIndex + 3);
                    newIndices.push_back(baseIndex + 2);
                }
            }
        }
    }
    
    if (posIntersectionPoints.size() >= 3) {
        createCapTriangles(posIntersectionPoints, posIntersectionNormals, plane.color, 
                          plane.normal, newVertices, newNormals, newColors, newIndices, true, plane.gap);
    }
    
    if (negIntersectionPoints.size() >= 3) {
        createCapTriangles(negIntersectionPoints, negIntersectionNormals, plane.color, 
                          -plane.normal, newVertices, newNormals, newColors, newIndices, false, plane.gap);
    }
    
    meshVertices = newVertices;
    meshNormals = newNormals;
    meshColors = newColors;
    meshIndices = newIndices;
}

void createCapTriangles(const std::vector<glm::vec3>& points, const std::vector<glm::vec3>& normals,const glm::vec3& color,const glm::vec3& planeNormal,std::vector<float>& outVertices,std::vector<float>& outNormals,std::vector<float>& outColors,std::vector<unsigned int>& outIndices,bool isPositiveSide,float gap) {
    if (points.size() < 3) return;

    glm::vec3 u, v;
    if (std::abs(planeNormal.x) < std::abs(planeNormal.y) &&
    std::abs(planeNormal.x) < std::abs(planeNormal.z)) {
    u = glm::normalize(glm::cross(planeNormal, glm::vec3(1.0f, 0.0f, 0.0f)));
    } else if (std::abs(planeNormal.y) < std::abs(planeNormal.z)) {
    u = glm::normalize(glm::cross(planeNormal, glm::vec3(0.0f, 1.0f, 0.0f)));
    } else {
    u = glm::normalize(glm::cross(planeNormal, glm::vec3(0.0f, 0.0f, 1.0f)));
    }
    v = glm::normalize(glm::cross(planeNormal, u));

    std::vector<glm::vec2> points2D;
    for (const auto& p : points) {
        float uCoord = glm::dot(p, u);
        float vCoord = glm::dot(p, v);
        points2D.push_back(glm::vec2(uCoord, vCoord));
    }

    glm::vec2 centroid(0.0f);
    for (const auto& p : points2D) {
        centroid += p;
    }
    centroid /= points2D.size();

    std::vector<size_t> indices(points.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        glm::vec2 vecA = points2D[a] - centroid;
        glm::vec2 vecB = points2D[b] - centroid;
        return std::atan2(vecA.y, vecA.x) > std::atan2(vecB.y, vecB.x);
    });


    unsigned int baseIndex = outVertices.size() / 3;

    for (size_t idx : indices) {
        outVertices.push_back(points[idx].x);
        outVertices.push_back(points[idx].y);
        outVertices.push_back(points[idx].z);

        outNormals.push_back(planeNormal.x);
        outNormals.push_back(planeNormal.y);
        outNormals.push_back(planeNormal.z);

        outColors.push_back(color.x);
        outColors.push_back(color.y);
        outColors.push_back(color.z);
    }

    for (size_t i = 1; i < indices.size() - 1; i++) {
        if (isPositiveSide) {
            outIndices.push_back(baseIndex);
            outIndices.push_back(baseIndex + i);
            outIndices.push_back(baseIndex + i + 1);
        } else {
            outIndices.push_back(baseIndex);
            outIndices.push_back(baseIndex + i + 1);
            outIndices.push_back(baseIndex + i);
        }
    }
}
