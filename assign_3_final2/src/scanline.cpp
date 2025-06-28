#include "scanline.h"
#include <algorithm>
#include <climits>
#include <functional>

void initializeEdgeTable(
    const std::vector<Point3D>& customVertices,
    std::vector<std::vector<Edge>>& edgeTable,
    int& scanlineMinY,
    int& scanlineMaxY
) {
    edgeTable.clear();
    
    if (customVertices.empty() || customVertices.size() < 3) return;
    
    scanlineMinY = INT_MAX;
    scanlineMaxY = INT_MIN;
    
    for (const auto& v : customVertices) {
        int y = static_cast<int>(v.y * 1000);
        scanlineMinY = std::min(scanlineMinY, y);
        scanlineMaxY = std::max(scanlineMaxY, y);
    }
    
    if (scanlineMinY == INT_MAX || scanlineMaxY == INT_MIN) return;
    
    edgeTable.resize(scanlineMaxY - scanlineMinY + 1);
    
    for (size_t i = 0; i < customVertices.size(); i++) {
        size_t j = (i + 1) % customVertices.size();
        size_t k = (i + customVertices.size() - 1) % customVertices.size();
        
        float yi = customVertices[i].y;
        float yj = customVertices[j].y;
        float yk = customVertices[k].y;
        
        if (yi == yj && yi == yk) continue;
        
        if (yi != yj) {
            Edge e;
            if (yi < yj) {
                e.yMin = static_cast<int>(yi * 1000);
                e.yMax = static_cast<int>(yj * 1000);
                e.xMin = customVertices[i].x;
                e.slope = (customVertices[j].x - customVertices[i].x) / (yj - yi);
            } else {
                e.yMin = static_cast<int>(yj * 1000);
                e.yMax = static_cast<int>(yi * 1000);
                e.xMin = customVertices[j].x;
                e.slope = (customVertices[i].x - customVertices[j].x) / (yi - yj);
            }
            
            e.color = glm::vec3(1.0f, 0.0f, 0.0f);
            
            int yIndex = e.yMin - scanlineMinY;
            if (yIndex >= 0 && yIndex < edgeTable.size()) {
                edgeTable[yIndex].push_back(e);
            }
        }
        
        bool isLocalMin = (yi < yj && yi < yk);
        bool isLocalMax = (yi > yj && yi > yk);
        bool isHorizontalIntersection = (yi == yj || yi == yk) && !(yi == yj && yi == yk);
        
        if (isLocalMin || isLocalMax || isHorizontalIntersection) {
            Edge vertexEdge;
            vertexEdge.yMin = static_cast<int>(yi * 1000);
            vertexEdge.yMax = vertexEdge.yMin;
            vertexEdge.xMin = customVertices[i].x;
            vertexEdge.color = glm::vec3(1.0f, 0.0f, 0.0f);
            
            if (isLocalMin) {
                vertexEdge.slope = 0.0f;
            } else if (isLocalMax) {
                vertexEdge.slope = 0.0f;
            } else if (isHorizontalIntersection) {
                vertexEdge.slope = 1.0f;
            }
            
            int yIndex = vertexEdge.yMin - scanlineMinY;
            if (yIndex >= 0 && yIndex < edgeTable.size()) {
                edgeTable[yIndex].push_back(vertexEdge);
            }
        }
    }
}

void scanlineFill(
    std::vector<std::vector<Edge>>& edgeTable,
    std::vector<ActiveEdge>& activeEdgeTable,
    int scanlineMinY,
    int scanlineMaxY,
    std::vector<float>& meshVertices,
    std::vector<float>& meshNormals,
    std::vector<float>& meshColors,
    std::vector<unsigned int>& meshIndices,
    const std::function<void()>& updateBuffers
) {
    if (edgeTable.empty()) return;
    
    std::vector<float> fillVertices;
    std::vector<float> fillNormals;
    std::vector<float> fillColors;
    std::vector<unsigned int> fillIndices;
    
    unsigned int baseIndex = meshVertices.size() / 3;
    activeEdgeTable.clear();
    
    for (int y = scanlineMinY; y <= scanlineMaxY; y++) {
        float currentY = y / 1000.0f;
        int yIndex = y - scanlineMinY;
        
        activeEdgeTable.erase(
            std::remove_if(activeEdgeTable.begin(), activeEdgeTable.end(),
                [y](const ActiveEdge& ae) { return ae.yMax <= y; }),
            activeEdgeTable.end()
        );
        
        if (yIndex >= 0 && yIndex < edgeTable.size()) {
            for (const Edge& e : edgeTable[yIndex]) {
                ActiveEdge ae;
                ae.x = e.xMin;
                ae.yMax = e.yMax;
                ae.slope = e.slope;
                ae.color = e.color;
                
                if (e.yMin == e.yMax) {
                    if (e.slope == 1.0f) {
                        activeEdgeTable.push_back(ae);
                        activeEdgeTable.push_back(ae);
                    } else {
                        activeEdgeTable.push_back(ae);
                    }
                } else {
                    activeEdgeTable.push_back(ae);
                }
            }
        }
        
        std::sort(activeEdgeTable.begin(), activeEdgeTable.end(),
            [](const ActiveEdge& a, const ActiveEdge& b) {
                return a.x < b.x;
            });
        
        for (size_t i = 0; i + 1 < activeEdgeTable.size(); i += 2) {
            float x1 = activeEdgeTable[i].x;
            float x2 = activeEdgeTable[i+1].x;
            
            if (x2 <= x1) continue;
            
            fillVertices.push_back(x1);
            fillVertices.push_back(currentY);
            fillVertices.push_back(0.0f);
            
            fillVertices.push_back(x2);
            fillVertices.push_back(currentY);
            fillVertices.push_back(0.0f);
            
            fillVertices.push_back(x2);
            fillVertices.push_back(currentY + 0.001f);
            fillVertices.push_back(0.0f);
            
            fillVertices.push_back(x1);
            fillVertices.push_back(currentY + 0.001f);
            fillVertices.push_back(0.0f);
            
            for (int j = 0; j < 4; j++) {
                fillColors.push_back(1.0f);
                fillColors.push_back(0.0f);
                fillColors.push_back(0.0f);
            }
            
            for (int j = 0; j < 4; j++) {
                fillNormals.push_back(0.0f);
                fillNormals.push_back(0.0f);
                fillNormals.push_back(1.0f);
            }
            
            unsigned int quadBaseIndex = (fillVertices.size() / 3) - 4;
            fillIndices.push_back(baseIndex + quadBaseIndex);
            fillIndices.push_back(baseIndex + quadBaseIndex + 1);
            fillIndices.push_back(baseIndex + quadBaseIndex + 2);
            
            fillIndices.push_back(baseIndex + quadBaseIndex);
            fillIndices.push_back(baseIndex + quadBaseIndex + 2);
            fillIndices.push_back(baseIndex + quadBaseIndex + 3);
        }
        
        for (ActiveEdge& ae : activeEdgeTable) {
            if (ae.yMax > y) {
                ae.x += ae.slope * 0.001f;
            }
        }
    }
    
    meshVertices.insert(meshVertices.end(), fillVertices.begin(), fillVertices.end());
    meshNormals.insert(meshNormals.end(), fillNormals.begin(), fillNormals.end());
    meshColors.insert(meshColors.end(), fillColors.begin(), fillColors.end());
    meshIndices.insert(meshIndices.end(), fillIndices.begin(), fillIndices.end());
    
    updateBuffers();
}
