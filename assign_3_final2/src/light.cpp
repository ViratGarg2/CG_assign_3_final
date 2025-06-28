#include "light.h"

Light::Light(const glm::vec3& dir, const glm::vec3& col, float i)
    : direction(glm::normalize(dir)), color(col), intensity(i) {}
