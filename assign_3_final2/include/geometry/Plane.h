#ifndef PLANE_H
#define PLANE_H

#include "Ray.h"
#include <glm/glm.hpp>

struct Plane {
    glm::vec3 normal;
    float distance;
    float gap;
    glm::vec3 color;
    float reflectivity;
    
    Plane(const glm::vec3& n, float d, float g = 0.0f, const glm::vec3& c = glm::vec3(1.0f), float refl = 0.0f)
        : normal(glm::normalize(n)), distance(d), gap(g), color(c), reflectivity(refl) {}
    
    float getSignedDistance(const glm::vec3& point) const {
        return glm::dot(normal, point) - distance;
    }
    
    bool intersect(const Ray& ray, float& t) const {
        float denom = glm::dot(normal, ray.direction);
        if (abs(denom) < 1e-6) return false;
        
        t = -(glm::dot(normal, ray.origin) + distance) / denom;
        return t > 0;
    }
};

#endif // PLANE_H
