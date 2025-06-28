#ifndef SPHERE_H
#define SPHERE_H

#include "Ray.h"
#include <glm/glm.hpp>

struct Sphere {
    glm::vec3 center;
    float radius;
    glm::vec3 color;
    float reflectivity;
    
    Sphere(const glm::vec3& c, float r, const glm::vec3& col, float refl = 0.0f) 
        : center(c), radius(r), color(col), reflectivity(refl) {}

    // Use the ray-sphere intersection formula
    // https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    // (ray.origin + t * ray.direction - center) . (ray.origin + t * ray.direction - center) = radius^2
    bool intersect(const Ray& ray, float& t) const {
        glm::vec3 oc = ray.origin - center;
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2.0f * glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) return false;

        float sqrtd = sqrt(discriminant);
        float root = (-b - sqrtd) / (2.0f * a);
        
        if (root < 0.001f) {
            root = (-b + sqrtd) / (2.0f * a);
            if (root < 0.001f) return false;
        }
        
        t = root;
        return true;
    }
    
    glm::vec3 getNormal(const glm::vec3& point) const {
        return glm::normalize(point - center);
    }
};

#endif // SPHERE_H
