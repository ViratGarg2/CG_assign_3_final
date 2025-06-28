#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>

struct Light {
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    
    Light(const glm::vec3& dir, const glm::vec3& col, float i = 1.0f);
};

#endif
