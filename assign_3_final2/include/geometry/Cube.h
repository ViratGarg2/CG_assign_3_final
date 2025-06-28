#ifndef CUBE_H
#define CUBE_H

#include "Ray.h"
#include <glm/glm.hpp>
#include <algorithm> // For std::swap
#include <limits> // For std::numeric_limits

struct Cube {
    glm::vec3 min;
    glm::vec3 max;
    glm::vec3 color;
    float reflectivity;
    
    Cube(const glm::vec3& min, const glm::vec3& max, const glm::vec3& col, float refl = 0.0f)
        : min(min), max(max), color(col), reflectivity(refl) {}

    bool intersect(const Ray& ray, float& t) const {
        float tMin = -std::numeric_limits<float>::infinity();
        float tMax = std::numeric_limits<float>::infinity();
        
        for (int i = 0; i < 3; i++) {
            if (fabs(ray.direction[i]) < 1e-6f) {
                if (ray.origin[i] < min[i] || ray.origin[i] > max[i]) {
                    return false;
                }
            } else {
                float invD = 1.0f / ray.direction[i];
                float t0 = (min[i] - ray.origin[i]) * invD;
                float t1 = (max[i] - ray.origin[i]) * invD;
                
                if (t0 > t1) std::swap(t0, t1);
                
                tMin = std::max(tMin, t0);
                tMax = std::min(tMax, t1);
                
                if (tMin > tMax) return false;
            }
        }
        
        if (tMin > 0) {
            t = tMin;
            return true;
        } else if (tMax > 0) {
            t = tMax;
            return true;
        }
        
        return false;
    }
    
    glm::vec3 getNormal(const glm::vec3& point) const {
        const float EPSILON = 1e-6f;
        
        if (fabs(point.x - min.x) < EPSILON) return glm::vec3(-1.0f, 0.0f, 0.0f);
        if (fabs(point.x - max.x) < EPSILON) return glm::vec3(1.0f, 0.0f, 0.0f);
        if (fabs(point.y - min.y) < EPSILON) return glm::vec3(0.0f, -1.0f, 0.0f);
        if (fabs(point.y - max.y) < EPSILON) return glm::vec3(0.0f, 1.0f, 0.0f);
        if (fabs(point.z - min.z) < EPSILON) return glm::vec3(0.0f, 0.0f, -1.0f);
        if (fabs(point.z - max.z) < EPSILON) return glm::vec3(0.0f, 0.0f, 1.0f);
        
        return glm::normalize(point - (min + max) * 0.5f);
    }
};

#endif // CUBE_H
