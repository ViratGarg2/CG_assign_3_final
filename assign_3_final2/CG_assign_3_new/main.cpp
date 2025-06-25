#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <algorithm>  // Add this for std::remove_if and std::sort


#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "file_utils.h"
#include "math_utils.h"
#include "OFFReader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

int theWindowWidth = 700, theWindowHeight = 700;
int theWindowPositionX = 40, theWindowPositionY = 40;

// Ray tracing structures and functions
struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    
    Ray(const glm::vec3& o, const glm::vec3& d) : origin(o), direction(glm::normalize(d)) {}
};

struct Sphere {
    glm::vec3 center;
    float radius;
    glm::vec3 color;
    float reflectivity;  // 0 to 1, where 1 is fully reflective
    
    Sphere(const glm::vec3& c, float r, const glm::vec3& col, float refl = 0.0f) 
        : center(c), radius(r), color(col), reflectivity(refl) {}
    
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
                
                // Update tMin and tMax
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

std::vector<Cube> cubes;

struct Plane {
    glm::vec3 normal;
    float distance;
    float gap;  // For mesh slicing
    glm::vec3 color;
    float reflectivity;  // For ray tracing
    
    Plane(const glm::vec3& n, float d, float g = 0.0f, const glm::vec3& c = glm::vec3(1.0f), float refl = 0.0f)
        : normal(glm::normalize(n)), distance(d), gap(g), color(c), reflectivity(refl) {}
    
    // For mesh slicing
    float getSignedDistance(const glm::vec3& point) const {
        return glm::dot(normal, point) - distance;
    }
    
    // For ray tracing
    bool intersect(const Ray& ray, float& t) const {
        float denom = glm::dot(normal, ray.direction);
        if (abs(denom) < 1e-6) return false;
        
        t = -(glm::dot(normal, ray.origin) + distance) / denom;
        return t > 0;
    }
};


std::vector<Plane> slicingPlanes = {
    Plane(glm::vec3(1.0f, 0.0f, 0.0f), 0.0f, 0.0f, glm::vec3(1.0f, 0.0f, 0.0f)),  // Red plane
    Plane(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, glm::vec3(0.0f, 1.0f, 0.0f)),  // Green plane
    Plane(glm::vec3(0.0f, 0.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f, 0.0f, 1.0f)),  // Blue plane
    Plane(glm::vec3(1.0f, 1.0f, 1.0f), 0.5f, 0.0f, glm::vec3(1.0f, 1.0f, 0.0f))   // Yellow plane
};

struct Light {
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    
    Light(const glm::vec3& dir, const glm::vec3& col, float i = 1.0f)
        : direction(glm::normalize(dir)), color(col), intensity(i) {}
};

// make it false to see other outputs->scanlineFill,Bresssenman and slicing
bool rayTracingMode = true;
OffModel* rayTracingModel = nullptr;
std::vector<Sphere> spheres;
std::vector<Plane> planes;
Light directionalLight(glm::vec3(1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
glm::vec3 backgroundColor(0.2f, 0.2f, 0.2f);

glm::vec3 reflect(const glm::vec3& incident, const glm::vec3& normal) {
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

bool rayTriangleIntersect(const Ray& ray, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float& t) {
    const float EPSILON = 1e-6f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);
    if (fabs(a) < EPSILON) return false;  // Ray is parallel to the triangle

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * glm::dot(edge2, q);
    return t > EPSILON;
}

glm::vec3 traceRay(const Ray& ray,glm::mat4 worldMat,int depth = 0) {
    if (depth > 5) return backgroundColor;  // Maximum recursion depth
    
    float closestT = std::numeric_limits<float>::infinity();
    bool hit = false;
    glm::vec3 hitColor;
    glm::vec3 hitNormal;
    float hitReflectivity = 0.0f;
    glm::vec3 hitPoint;
    
    // Check sphere intersections
    for (const auto& sphere : spheres) {
        float t;
        if (sphere.intersect(ray, t) && t < closestT) {
            closestT = t;
            hit = true;
            hitPoint = ray.origin + t * ray.direction;
            hitNormal = sphere.getNormal(hitPoint);
            hitColor = sphere.color;
            hitReflectivity = sphere.reflectivity;
        }
    }

    for (const auto& cube : cubes) {
        float t;
        if (cube.intersect(ray, t) && t < closestT) {
            closestT = t;
            hit = true;
            hitPoint = ray.origin + t * ray.direction;
            hitNormal = cube.getNormal(hitPoint);
            hitColor = cube.color;
            hitReflectivity = cube.reflectivity;
        }
    }

    if (rayTracingModel) {

        for (int i = 0; i < rayTracingModel->numberOfPolygons; i++) {
            const Polygon& poly = rayTracingModel->polygons[i];
            if (poly.noSides >= 3) {

                const Vertex& v0 = rayTracingModel->vertices[poly.v[0]];
                const Vertex& v1 = rayTracingModel->vertices[poly.v[1]];
                const Vertex& v2 = rayTracingModel->vertices[poly.v[2]];

                glm::vec4 p0_transformed = worldMat * glm::vec4(v0.x, v0.y, v0.z, 1.0f);
            glm::vec4 p1_transformed = worldMat * glm::vec4(v1.x, v1.y, v1.z, 1.0f);
            glm::vec4 p2_transformed = worldMat * glm::vec4(v2.x, v2.y, v2.z, 1.0f);

            glm::vec3 p0(p0_transformed.x, p0_transformed.y, p0_transformed.z);
            glm::vec3 p1(p1_transformed.x, p1_transformed.y, p1_transformed.z);
            glm::vec3 p2(p2_transformed.x, p2_transformed.y, p2_transformed.z);
            
            

                float t;
                if (rayTriangleIntersect(ray, p0, p1, p2, t) && t < closestT) {
                    closestT = t;
                    hit = true;
                    hitPoint = ray.origin + t * ray.direction;
                    hitNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
                    hitColor = glm::vec3(1.0f, 0.0f, 0.0f);  // Default gray color for the mesh
                    hitReflectivity = 0.5f;  // No reflectivity for the mesh
                }
            }
        }
    }

    if (!hit) {
        // Create a simple sky gradient
        float t = 0.5f * (glm::normalize(ray.direction).y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }
    

    glm::vec3 finalColor = hitColor * 0.1f;  // Ambient light
    
    Ray shadowRay(hitPoint + hitNormal * 0.001f, -directionalLight.direction);
    bool inShadow = false;
    
    for (const auto& sphere : spheres) {
        float t;
        if (sphere.intersect(shadowRay, t)) {
            inShadow = true;
            break;
        }
    }

    if (!inShadow && rayTracingModel) {
        for (int i = 0; i < rayTracingModel->numberOfPolygons; i++) {
            const Polygon& poly = rayTracingModel->polygons[i];
            if (poly.noSides >= 3) {
                const Vertex& v0 = rayTracingModel->vertices[poly.v[0]];
                const Vertex& v1 = rayTracingModel->vertices[poly.v[1]];
                const Vertex& v2 = rayTracingModel->vertices[poly.v[2]];
    
                glm::vec4 p0_transformed = worldMat * glm::vec4(v0.x, v0.y, v0.z, 1.0f);
                glm::vec4 p1_transformed = worldMat * glm::vec4(v1.x, v1.y, v1.z, 1.0f);
                glm::vec4 p2_transformed = worldMat * glm::vec4(v2.x, v2.y, v2.z, 1.0f);
    
                glm::vec3 p0(p0_transformed.x, p0_transformed.y, p0_transformed.z);
                glm::vec3 p1(p1_transformed.x, p1_transformed.y, p1_transformed.z);
                glm::vec3 p2(p2_transformed.x, p2_transformed.y, p2_transformed.z);
    
                float t;
                if (rayTriangleIntersect(shadowRay, p0, p1, p2, t)) {
                    inShadow = true;
                    break;
                }
            }
        }
    }

    if (!inShadow) {
        for (const auto& cube : cubes) {
            float t;
            if (cube.intersect(shadowRay, t)) {
                inShadow = true;
                break;
            }
        }
    }
    
    if (!inShadow) {
        float diffuse = std::max(0.0f, glm::dot(hitNormal, -directionalLight.direction));
        finalColor += hitColor * directionalLight.color * diffuse * directionalLight.intensity;
    }
    
    if (hitReflectivity > 0.0f && depth < 5) {
        glm::vec3 reflectedDir = reflect(ray.direction, hitNormal);
        Ray reflectedRay(hitPoint + hitNormal * 0.001f, reflectedDir);
        glm::vec3 reflectedColor = traceRay(reflectedRay,worldMat,depth + 1);
        finalColor = glm::mix(finalColor, reflectedColor, hitReflectivity);
    }
    
    finalColor = glm::clamp(finalColor, glm::vec3(0.0f), glm::vec3(1.0f));
    return finalColor;
}

void renderRayTracedScene(Matrix4f worldMatrix,Matrix4f viewMatrix) {

    glm::mat4 worldMat(
        worldMatrix.m[0][0], worldMatrix.m[1][0], worldMatrix.m[2][0], worldMatrix.m[3][0],
        worldMatrix.m[0][1], worldMatrix.m[1][1], worldMatrix.m[2][1], worldMatrix.m[3][1],
        worldMatrix.m[0][2], worldMatrix.m[1][2], worldMatrix.m[2][2], worldMatrix.m[3][2],
        worldMatrix.m[0][3], worldMatrix.m[1][3], worldMatrix.m[2][3], worldMatrix.m[3][3]
    );
    
    glm::mat4 viewMat(
        viewMatrix.m[0][0], viewMatrix.m[1][0], viewMatrix.m[2][0], viewMatrix.m[3][0],
        viewMatrix.m[0][1], viewMatrix.m[1][1], viewMatrix.m[2][1], viewMatrix.m[3][1],
        viewMatrix.m[0][2], viewMatrix.m[1][2], viewMatrix.m[2][2], viewMatrix.m[3][2],
        viewMatrix.m[0][3], viewMatrix.m[1][3], viewMatrix.m[2][3], viewMatrix.m[3][3]
    );

    std::vector<glm::vec3> framebuffer(theWindowWidth * theWindowHeight);
    
    float aspectRatio = float(theWindowWidth) / float(theWindowHeight);
    float viewportHeight = 2.0f;
    float viewportWidth = aspectRatio * viewportHeight;
    float focalLength = 1.0f;

    glm::vec3 cameraPos = glm::vec3(glm::inverse(viewMat)[3]);
glm::vec3 cameraFront = -glm::vec3(viewMat[2]); // Forward direction
glm::vec3 cameraUp = glm::vec3(viewMat[1]);     // Up direction

glm::vec3 origin = cameraPos;
    glm::vec3 horizontal(viewportWidth, 0.0f, 0.0f);
    glm::vec3 vertical(0.0f, viewportHeight, 0.0f);
    glm::vec3 lowerLeftCorner = origin - horizontal/2.0f - vertical/2.0f - glm::vec3(0.0f, 0.0f, focalLength);

    for (int j = 0; j < theWindowHeight; j++) {
        for (int i = 0; i < theWindowWidth; i++) {
            float u = float(i) / float(theWindowWidth - 1);
            float v = float(j) / float(theWindowHeight - 1);  // Invert v to fix orientation
            
            Ray ray(origin, lowerLeftCorner + u*horizontal + v*vertical - origin);
            glm::vec3 color = traceRay(ray,worldMat);
            
            framebuffer[j * theWindowWidth + i] = color;
        }
    }

    // printf("Ray tracing complete. Sample colors:\n");
    // printf("Center pixel: (%f, %f, %f)\n", 
    //        framebuffer[theWindowHeight/2 * theWindowWidth + theWindowWidth/2].x,
    //        framebuffer[theWindowHeight/2 * theWindowWidth + theWindowWidth/2].y,
    //        framebuffer[theWindowHeight/2 * theWindowWidth + theWindowWidth/2].z);

    static GLuint texID = 0;
    if (texID == 0) {
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    glBindTexture(GL_TEXTURE_2D, texID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, theWindowWidth, theWindowHeight, 0, GL_RGB, GL_FLOAT, framebuffer.data());

    static GLuint quadVAO = 0;
    static GLuint quadVBO = 0;
    if (quadVAO == 0) {
        float quadVertices[] = {
            // positions        // texture coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }

    static GLuint screenShaderProgram = 0;
    if (screenShaderProgram == 0) {
        const char* vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoords;
            out vec2 TexCoords;
            void main() {
                TexCoords = aTexCoords;
                gl_Position = vec4(aPos, 1.0);
            }
        )";
        
        const char* fragmentShaderSource = R"(
            #version 330 core
            in vec2 TexCoords;
            out vec4 FragColor;
            uniform sampler2D screenTexture;
            void main() {
                FragColor = texture(screenTexture, TexCoords);
            }
        )";

        screenShaderProgram = glCreateProgram();
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        
        glCompileShader(vertexShader);
        glCompileShader(fragmentShader);
        
        glAttachShader(screenShaderProgram, vertexShader);
        glAttachShader(screenShaderProgram, fragmentShader);
        glLinkProgram(screenShaderProgram);
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    glUseProgram(screenShaderProgram);
    glBindVertexArray(quadVAO);
    glDisable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texID);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        printf("OpenGL error: 0x%x\n", error);
    }
}

#define GL_SILENCE_DEPRECATION

/********************************************************************/
/*   Variables */

char theProgramTitle[] = "Sample";

bool isFullScreen = false;
bool isAnimating = true;
float rotation = 0.0f;
GLuint VBO, VAO, IBO;
GLuint normalVBO, colorVBO;
// GLuint gWorldLocation;
GLuint gWorldLocation, gViewLocation, gProjectionLocation;
GLuint ShaderProgram;

const int ANIMATION_DELAY = 20; /* milliseconds between rendering */
const char *pVSFileName = "shaders/shader.vs";
const char *pFSFileName = "shaders/shader.fs";

float cameraPos[3] = {0.0f, 0.0f, 3.0f};
float cameraFront[3] = {0.0f, 0.0f, -1.0f};
float cameraUp[3] = {0.0f, 1.0f, 0.0f};

OffModel* model = nullptr;
std::vector<float> meshVertices;
std::vector<float> meshNormals;
std::vector<float> meshColors;
std::vector<unsigned int> meshIndices;
std::vector<float> originalVertices;



bool showScanlineFill = false;
bool customPolygonMode = false;

struct Point3D {
    float x;
    float y;
    float z;
    
    Point3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

std::vector<Point3D> customVertices;

// Vertices for rendering the line
std::vector<float> lineVertices;
std::vector<unsigned int> lineIndices;
GLuint lineVAO = 0, lineVBO = 0, lineIBO = 0;
bool showLine = false;

struct Edge {
    float yMax;        // Maximum y-coordinate
    float xMin;        // x-coordinate at yMin
    float slope;       // 1/m (inverse slope)
    float yMin;        // Minimum y-coordinate
    glm::vec3 color;   // Color at this edge
};

struct ActiveEdge {
    float x;           // Current x-coordinate
    float yMax;        // Maximum y-coordinate
    float slope;       // 1/m (inverse slope)
    glm::vec3 color;   // Color at this edge
};

std::vector<std::vector<Edge>> edgeTable;
std::vector<ActiveEdge> activeEdgeTable;
int scanlineMinY = 0;
int scanlineMaxY = 0;


void createProjectionMatrix(Matrix4f& Projection) {
    // Simple orthographic projection
    float aspectRatio = (float)theWindowWidth / (float)theWindowHeight;
    float scale = 1.0f / model->extent * 1.5f; // Scale based on model size
    
    // Create orthographic projection matrix
    Projection.m[0][0] = scale / aspectRatio;
    Projection.m[0][1] = 0.0f;
    Projection.m[0][2] = 0.0f;
    Projection.m[0][3] = 0.0f;
    
    Projection.m[1][0] = 0.0f;
    Projection.m[1][1] = scale;
    Projection.m[1][2] = 0.0f;
    Projection.m[1][3] = 0.0f;
    
    Projection.m[2][0] = 0.0f;
    Projection.m[2][1] = 0.0f;
    Projection.m[2][2] = scale;
    Projection.m[2][3] = 0.0f;
    
    Projection.m[3][0] = 0.0f;
    Projection.m[3][1] = 0.0f;
    Projection.m[3][2] = 0.0f;
    Projection.m[3][3] = 1.0f;
}

void createViewMatrix(Matrix4f& View) {
    float lookX = cameraPos[0] + cameraFront[0];
    float lookY = cameraPos[1] + cameraFront[1];
    float lookZ = cameraPos[2] + cameraFront[2];
    
  
    Vector3f position = {cameraPos[0], cameraPos[1], cameraPos[2]};
    Vector3f target = {lookX, lookY, lookZ};
    Vector3f up = {cameraUp[0], cameraUp[1], cameraUp[2]};
    
 
    Vector3f zaxis = {
        position.x - target.x,
        position.y - target.y,
        position.z - target.z
    };

    float length = sqrt(zaxis.x * zaxis.x + zaxis.y * zaxis.y + zaxis.z * zaxis.z);
    zaxis.x /= length;
    zaxis.y /= length;
    zaxis.z /= length;
    
    Vector3f xaxis = {
        up.y * zaxis.z - up.z * zaxis.y,
        up.z * zaxis.x - up.x * zaxis.z,
        up.x * zaxis.y - up.y * zaxis.x
    };
    length = sqrt(xaxis.x * xaxis.x + xaxis.y * xaxis.y + xaxis.z * xaxis.z);
    xaxis.x /= length;
    xaxis.y /= length;
    xaxis.z /= length;
    
    Vector3f yaxis = {
        zaxis.y * xaxis.z - zaxis.z * xaxis.y,
        zaxis.z * xaxis.x - zaxis.x * xaxis.z,
        zaxis.x * xaxis.y - zaxis.y * xaxis.x
    };
    
    View.m[0][0] = xaxis.x;
    View.m[0][1] = xaxis.y;
    View.m[0][2] = xaxis.z;
    View.m[0][3] = -xaxis.x * position.x - xaxis.y * position.y - xaxis.z * position.z;
    
    View.m[1][0] = yaxis.x;
    View.m[1][1] = yaxis.y;
    View.m[1][2] = yaxis.z;
    View.m[1][3] = -yaxis.x * position.x - yaxis.y * position.y - yaxis.z * position.z;
    
    View.m[2][0] = zaxis.x;
    View.m[2][1] = zaxis.y;
    View.m[2][2] = zaxis.z;
    View.m[2][3] = -zaxis.x * position.x - zaxis.y * position.y - zaxis.z * position.z;
    
    View.m[3][0] = 0.0f;
    View.m[3][1] = 0.0f;
    View.m[3][2] = 0.0f;
    View.m[3][3] = 1.0f;
}

void createCapTriangles(const std::vector<glm::vec3>& points, 
    const std::vector<glm::vec3>& normals,
    const glm::vec3& color,
    const glm::vec3& planeNormal,
    std::vector<float>& outVertices,
    std::vector<float>& outNormals,
    std::vector<float>& outColors,
    std::vector<unsigned int>& outIndices,
    bool isPositiveSide,
    float gap) {
// Check if we have enough points
if (points.size() < 3) return;

// Project points to 2D for triangulation
// Find basis vectors for the cutting plane
glm::vec3 u, v;
if (std::abs(planeNormal.x) < std::abs(planeNormal.y) &&
std::abs(planeNormal.x) < std::abs(planeNormal.z)) {
// x is smallest
u = glm::normalize(glm::cross(planeNormal, glm::vec3(1.0f, 0.0f, 0.0f)));
} else if (std::abs(planeNormal.y) < std::abs(planeNormal.z)) {
// y is smallest
u = glm::normalize(glm::cross(planeNormal, glm::vec3(0.0f, 1.0f, 0.0f)));
} else {
// z is smallest
u = glm::normalize(glm::cross(planeNormal, glm::vec3(0.0f, 0.0f, 1.0f)));
}
v = glm::normalize(glm::cross(planeNormal, u));

std::vector<glm::vec2> points2D;
for (const auto& p : points) {
float uCoord = glm::dot(p, u);
float vCoord = glm::dot(p, v);
points2D.push_back(glm::vec2(uCoord, vCoord));
}

// Calculate centroid for sorting
glm::vec2 centroid(0.0f);
for (const auto& p : points2D) {
centroid += p;
}
centroid /= points2D.size();

// Sort points in clockwise/counterclockwise order
std::vector<size_t> indices(points.size());
for (size_t i = 0; i < indices.size(); i++) {
indices[i] = i;
}

std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
glm::vec2 vecA = points2D[a] - centroid;
glm::vec2 vecB = points2D[b] - centroid;
return std::atan2(vecA.y, vecA.x) > std::atan2(vecB.y, vecB.x);
});

// Create triangles using a simple fan triangulation
// This works well for convex polygons
unsigned int baseIndex = outVertices.size() / 3;

// Add all vertices
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

// Create fan triangulation
for (size_t i = 1; i < indices.size() - 1; i++) {
if (isPositiveSide) {
// Positive cap - clockwise winding
outIndices.push_back(baseIndex);
outIndices.push_back(baseIndex + i);
outIndices.push_back(baseIndex + i + 1);
} else {
// Negative cap - counterclockwise winding
outIndices.push_back(baseIndex);
outIndices.push_back(baseIndex + i + 1);
outIndices.push_back(baseIndex + i);
}
}
}


void sliceMesh(const Plane& plane) {
    std::vector<float> newVertices;
    std::vector<float> newNormals;
    std::vector<float> newColors;
    std::vector<unsigned int> newIndices;
    
    // Maps to store intersection points for creating cap triangles
    std::vector<glm::vec3> posIntersectionPoints;
    std::vector<glm::vec3> negIntersectionPoints;
    std::vector<glm::vec3> posIntersectionNormals;
    std::vector<glm::vec3> negIntersectionNormals;
    
    // Process each triangle
    for (size_t i = 0; i < meshIndices.size(); i += 3) {
        std::vector<glm::vec3> triangleVerts;
        std::vector<glm::vec3> triangleNorms;
        std::vector<glm::vec3> triangleColors;
        std::vector<float> distances;
        
        // Get triangle vertices and compute distances to plane
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
        
        // Count vertices on each side of the plane
        int positiveSide = 0, negativeSide = 0;
        for (float d : distances) {
            if (d > 0) positiveSide++;
            else if (d < 0) negativeSide++;
        }
        
        // If all vertices are on one side, add triangle to appropriate segment
        if (positiveSide == 3 || negativeSide == 3) {
            unsigned int baseIndex = newVertices.size() / 3;
            
            for (int j = 0; j < 3; j++) {
                glm::vec3 vert = triangleVerts[j];
                
                // Apply gap translation if needed
                if (positiveSide == 3) { // Positive side
                    vert += plane.normal * (plane.gap * 0.5f);
                } else { // Negative side
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
            
            // Keep original winding order
            newIndices.push_back(baseIndex);
            newIndices.push_back(baseIndex + 1);
            newIndices.push_back(baseIndex + 2);
            
            continue;
        }
        
        // Triangle intersects the plane - compute intersection points
        std::vector<glm::vec3> intersections;
        std::vector<glm::vec3> intersectionNorms;
        std::vector<glm::vec3> intersectionColors;
        
        for (int j = 0; j < 3; j++) {
            int k = (j + 1) % 3;
            
            // Skip if vertices are on the same side of the plane
            if (distances[j] * distances[k] >= 0.0f && !(distances[j] == 0.0f || distances[k] == 0.0f)) {
                continue;
            }
            
            // Calculate intersection parameter
            float t;
            if (distances[j] == 0.0f) {
                t = 0.0f;
            } else if (distances[k] == 0.0f) {
                t = 1.0f;
            } else {
                t = distances[j] / (distances[j] - distances[k]);
            }
            
            // Calculate intersection point with better precision
            glm::vec3 intersection = triangleVerts[j] + t * (triangleVerts[k] - triangleVerts[j]);
            
            // Properly interpolate normal and color
            glm::vec3 normal = glm::normalize((1.0f - t) * triangleNorms[j] + t * triangleNorms[k]);
            glm::vec3 color = (1.0f - t) * triangleColors[j] + t * triangleColors[k];
            
            intersections.push_back(intersection);
            intersectionNorms.push_back(normal);
            intersectionColors.push_back(color);
            
            // Store intersection points for cap triangles
            if (positiveSide > 0 && negativeSide > 0) {  // Only add if triangle truly intersects
                posIntersectionPoints.push_back(intersection + plane.normal * (plane.gap * 0.5f));
                negIntersectionPoints.push_back(intersection - plane.normal * (plane.gap * 0.5f));
                posIntersectionNormals.push_back(normal);
                negIntersectionNormals.push_back(normal);
            }
        }
        
        // Handle specific intersection cases
        if (intersections.size() == 2) {
            // Sort vertices by their sign (positive or negative side)
            std::vector<int> posIndices, negIndices;
            
            for (int j = 0; j < 3; j++) {
                if (distances[j] > 0) {
                    posIndices.push_back(j);
                } else if (distances[j] < 0) {
                    negIndices.push_back(j);
                } else {
                    // Handle vertices exactly on the plane
                    // For simplicity, we'll put them on both sides
                    posIndices.push_back(j);
                    negIndices.push_back(j);
                }
            }
            
            // Create triangles for positive side
            if (posIndices.size() > 0) {
                // Create one or two triangles depending on how vertices are distributed
                if (posIndices.size() == 1) {
                    // Create one triangle with the positive vertex and the two intersection points
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    // Add positive vertex
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
                    
                    // Add intersection points
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
                    
                    // Add triangle with correct winding order
                    newIndices.push_back(baseIndex);
                    newIndices.push_back(baseIndex + 1);
                    newIndices.push_back(baseIndex + 2);
                } else if (posIndices.size() == 2) {
                    // Create two triangles from the two positive vertices and two intersection points
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    // Add positive vertices
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
                    
                    // Add intersection points
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
                    
                    // Add triangles with correct winding order
                    newIndices.push_back(baseIndex);     // First vertex
                    newIndices.push_back(baseIndex + 1); // Second vertex
                    newIndices.push_back(baseIndex + 2); // First intersection
                    
                    newIndices.push_back(baseIndex + 1); // Second vertex
                    newIndices.push_back(baseIndex + 3); // Second intersection
                    newIndices.push_back(baseIndex + 2); // First intersection
                }
            }
            
            // Create triangles for negative side - similar logic as positive side
            if (negIndices.size() > 0) {
                // Create one or two triangles depending on how vertices are distributed
                if (negIndices.size() == 1) {
                    // Create one triangle with the negative vertex and the two intersection points
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    // Add negative vertex
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
                    // Create two triangles from the two negative vertices and two intersection points
                    unsigned int baseIndex = newVertices.size() / 3;
                    
                    // Add negative vertices
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
                    
                    // Add intersection points - ensure correct winding order
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
                    
                    // Add triangles with correct winding order
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

void drawLine2DBresenham(float x1, float y1, float x2, float y2) {
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
// Function to render the line
void renderLine() {
    if (!showLine || lineVertices.empty() || lineIndices.empty()) return;
    
    if (lineVAO == 0 || lineVBO == 0 || lineIBO == 0) {
        return;
    }

    glUseProgram(ShaderProgram);
    
    // Set up transformation matrices
    Matrix4f World;
    // Initialize identity matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            World.m[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    Matrix4f View;
    createViewMatrix(View);
    
    Matrix4f Projection;
    createProjectionMatrix(Projection);
    
    // Apply transformations
    glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, &World.m[0][0]);
    glUniformMatrix4fv(gViewLocation, 1, GL_TRUE, &View.m[0][0]);
    glUniformMatrix4fv(gProjectionLocation, 1, GL_TRUE, &Projection.m[0][0]);

    GLint prevVAO;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    
    glBindVertexArray(lineVAO);
    
    glLineWidth(4.0f);
    
    glDrawArrays(GL_LINE_STRIP, 0, lineVertices.size() / 9);
    
    glBindVertexArray(prevVAO);
    glLineWidth(1.0f);
    
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error during line rendering: %d\n", err);
    }
}


void prepareMeshData() {
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

void CreateMeshBuffers() {
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

void initializeEdgeTable() {
    edgeTable.clear();
    activeEdgeTable.clear();
    
    if (customVertices.empty() || customVertices.size() < 3) return;
    
    // Find min and max y coordinates
    scanlineMinY = INT_MAX;
    scanlineMaxY = INT_MIN;
    
    for (const auto& v : customVertices) {
        int y = static_cast<int>(v.y * 1000); // Scale for precision
        scanlineMinY = std::min(scanlineMinY, y);
        scanlineMaxY = std::max(scanlineMaxY, y);
    }
    
    if (scanlineMinY == INT_MAX || scanlineMaxY == INT_MIN) return;
    
    // Initialize edge table with empty vectors
    edgeTable.resize(scanlineMaxY - scanlineMinY + 1);
    
    // Process each edge of the polygon
    for (size_t i = 0; i < customVertices.size(); i++) {
        size_t j = (i + 1) % customVertices.size(); // Next vertex
        size_t k = (i + customVertices.size() - 1) % customVertices.size(); // Previous vertex
        
        // Get vertex y-coordinates
        float yi = customVertices[i].y;
        float yj = customVertices[j].y;
        float yk = customVertices[k].y;
        
        // Skip if both adjacent edges are horizontal
        if (yi == yj && yi == yk) continue;
        
        // Handle non-horizontal edges
        if (yi != yj) {
            Edge e;
            // Determine yMin and yMax
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
            
            // Set color
            e.color = glm::vec3(1.0f, 0.0f, 0.0f); // Default to red
            
            // Add edge to edge table
            int yIndex = e.yMin - scanlineMinY;
            if (yIndex >= 0 && yIndex < edgeTable.size()) {
                edgeTable[yIndex].push_back(e);
            }
        }
        
        // Handle vertex intersection cases
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
                // For local minimum (like point B), count as single intersection
                vertexEdge.slope = 0.0f;
            } else if (isLocalMax) {
                // For local maximum (like point E), count as single intersection
                vertexEdge.slope = 0.0f;
            } else if (isHorizontalIntersection) {
                // For horizontal intersection (like point D), count as double intersection
                vertexEdge.slope = 1.0f;
            }
            
            int yIndex = vertexEdge.yMin - scanlineMinY;
            if (yIndex >= 0 && yIndex < edgeTable.size()) {
                edgeTable[yIndex].push_back(vertexEdge);
            }
        }
    }
}

void scanlineFill() {
    if (edgeTable.empty()) return;
    
    std::vector<float> fillVertices;
    std::vector<float> fillNormals;
    std::vector<float> fillColors;
    std::vector<unsigned int> fillIndices;
    
    unsigned int baseIndex = meshVertices.size() / 3;
    activeEdgeTable.clear();
    
    // Process each scanline
    for (int y = scanlineMinY; y <= scanlineMaxY; y++) {
        float currentY = y / 1000.0f; // Convert back to original scale
        int yIndex = y - scanlineMinY;
        
        // Remove finished edges
        activeEdgeTable.erase(
            std::remove_if(activeEdgeTable.begin(), activeEdgeTable.end(),
                [y](const ActiveEdge& ae) { return ae.yMax <= y; }),
            activeEdgeTable.end()
        );
        
        // Add new edges to active edge table
        if (yIndex >= 0 && yIndex < edgeTable.size()) {
            for (const Edge& e : edgeTable[yIndex]) {
                ActiveEdge ae;
                ae.x = e.xMin;
                ae.yMax = e.yMax;
                ae.slope = e.slope;
                ae.color = e.color;
                
                if (e.yMin == e.yMax) {
                    // Special handling for vertex intersections
                    if (e.slope == 1.0f) {
                        // Double intersection (horizontal intersection point)
                        activeEdgeTable.push_back(ae);
                        activeEdgeTable.push_back(ae);
                    } else {
                        // Single intersection (local min/max)
                        activeEdgeTable.push_back(ae);
                    }
                } else {
                    // Regular edge
                    activeEdgeTable.push_back(ae);
                }
            }
        }
        
        // Sort active edges by x coordinate
        std::sort(activeEdgeTable.begin(), activeEdgeTable.end(),
            [](const ActiveEdge& a, const ActiveEdge& b) {
                return a.x < b.x;
            });
        
        // Fill between pairs of intersections
        for (size_t i = 0; i + 1 < activeEdgeTable.size(); i += 2) {
            float x1 = activeEdgeTable[i].x;
            float x2 = activeEdgeTable[i+1].x;
            
            // Skip invalid segments
            if (x2 <= x1) continue;
            
            // Create vertices for the scanline segment
            // Bottom vertices
            fillVertices.push_back(x1);
            fillVertices.push_back(currentY);
            fillVertices.push_back(0.0f);
            
            fillVertices.push_back(x2);
            fillVertices.push_back(currentY);
            fillVertices.push_back(0.0f);
            
            // Top vertices (slightly higher)
            fillVertices.push_back(x2);
            fillVertices.push_back(currentY + 0.001f);
            fillVertices.push_back(0.0f);
            
            fillVertices.push_back(x1);
            fillVertices.push_back(currentY + 0.001f);
            fillVertices.push_back(0.0f);
            
            // Add colors (red fill)
            for (int j = 0; j < 4; j++) {
                fillColors.push_back(1.0f); // R
                fillColors.push_back(0.0f); // G
                fillColors.push_back(0.0f); // B
            }
            
            // Add normals (facing forward)
            for (int j = 0; j < 4; j++) {
                fillNormals.push_back(0.0f);
                fillNormals.push_back(0.0f);
                fillNormals.push_back(1.0f);
            }
            
            // Add indices for two triangles
            unsigned int quadBaseIndex = (fillVertices.size() / 3) - 4;
            fillIndices.push_back(baseIndex + quadBaseIndex);
            fillIndices.push_back(baseIndex + quadBaseIndex + 1);
            fillIndices.push_back(baseIndex + quadBaseIndex + 2);
            
            fillIndices.push_back(baseIndex + quadBaseIndex);
            fillIndices.push_back(baseIndex + quadBaseIndex + 2);
            fillIndices.push_back(baseIndex + quadBaseIndex + 3);
        }
        
        // Update x coordinates for next scanline
        for (ActiveEdge& ae : activeEdgeTable) {
            if (ae.yMax > y) { // Only update if edge continues
                ae.x += ae.slope * 0.001f; // Account for scaling
            }
        }
    }
    
    // Add the fill data to the mesh
    meshVertices.insert(meshVertices.end(), fillVertices.begin(), fillVertices.end());
    meshNormals.insert(meshNormals.end(), fillNormals.begin(), fillNormals.end());
    meshColors.insert(meshColors.end(), fillColors.begin(), fillColors.end());
    meshIndices.insert(meshIndices.end(), fillIndices.begin(), fillIndices.end());
    
    // Update the mesh buffers
    CreateMeshBuffers();
}



void computeFPS()
{
	static int frameCount = 0;
	static int lastFrameTime = 0;
	static char *title = NULL;
	int currentTime;

	if (!title)
		title = (char *)malloc((strlen(theProgramTitle) + 20) * sizeof(char));
	frameCount++;
	currentTime = 0;
	if (currentTime - lastFrameTime > 1000)
	{
		sprintf(title, "%s [ FPS: %4.2f ]",
				theProgramTitle,
				frameCount * 1000.0 / (currentTime - lastFrameTime));
		lastFrameTime = currentTime;
		frameCount = 0;
	}
}


static void AddShader(GLuint ShaderProgram, const char *pShaderText, GLenum ShaderType)
{
	GLuint ShaderObj = glCreateShader(ShaderType);

	if (ShaderObj == 0)
	{
		fprintf(stderr, "Error creating shader type %d\n", ShaderType);
		exit(0);
	}

	const GLchar *p[1];
	p[0] = pShaderText;
	GLint Lengths[1];
	Lengths[0] = strlen(pShaderText);
	glShaderSource(ShaderObj, 1, p, Lengths);
	glCompileShader(ShaderObj);
	GLint success;
	glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		GLchar InfoLog[1024];
		glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
		fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
		exit(1);
	}

	glAttachShader(ShaderProgram, ShaderObj);
}

using namespace std;

static void CompileShaders()
{
    ShaderProgram = glCreateProgram();

    if (ShaderProgram == 0)
    {
        fprintf(stderr, "Error creating shader program\n");
        exit(1);
    }

    string vs, fs;

    if (!ReadFile(pVSFileName, vs))
    {
        exit(1);
    }

    if (!ReadFile(pFSFileName, fs))
    {
        exit(1);
    }

    AddShader(ShaderProgram, vs.c_str(), GL_VERTEX_SHADER);
    AddShader(ShaderProgram, fs.c_str(), GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = {0};

    glLinkProgram(ShaderProgram);
    glGetProgramiv(ShaderProgram, GL_LINK_STATUS, &Success);
    if (Success == 0)
    {
        glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
    }

    glValidateProgram(ShaderProgram);
    glGetProgramiv(ShaderProgram, GL_VALIDATE_STATUS, &Success);
    if (!Success)
    {
        glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }

    glUseProgram(ShaderProgram);
    gWorldLocation = glGetUniformLocation(ShaderProgram, "gWorld");
    gViewLocation = glGetUniformLocation(ShaderProgram, "gView");
    gProjectionLocation = glGetUniformLocation(ShaderProgram, "gProjection");
}


/********************************************************************
 Callback Functions
 */

void onInit(int argc, char *argv[]) {

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glDrawBuffer(GL_BACK);

    lineVAO = 0;
    lineVBO = 0;
    lineIBO = 0;

    spheres.clear();
    planes.clear();
    cubes.clear();
    
    spheres.push_back(Sphere(glm::vec3(0.0f, 0.0f, -3.0f), 1.0f, glm::vec3(0.8f, 0.2f, 0.2f), 0.5f));
    spheres.push_back(Sphere(glm::vec3(0.0f, -101.0f, -3.0f), 100.0f, glm::vec3(0.8f, 0.8f, 0.0f), 0.3f));
    cubes.push_back(Cube(glm::vec3(-2.5f, -0.5f, -2.0f), glm::vec3(-1.5f, 0.5f, -1.0f),glm::vec3(0.2f, 0.8f, 0.2f), 0.7f));
    model = readOffFile("isohedron.off");
    if (!model) {
        fprintf(stderr, "Failed to load isohedron.off\n");
        exit(1);
    }
    prepareMeshData();
    CreateMeshBuffers();

    rayTracingModel = model;

    CompileShaders();
    glEnable(GL_DEPTH_TEST);
}



static void onDisplay() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (rayTracingMode) {

        GLboolean depth_test_enabled;
        glGetBooleanv(GL_DEPTH_TEST, &depth_test_enabled);

        Matrix4f World;
        Matrix4f View;
        Matrix4f Projection;
        float scale = 0.5f;
        World.m[0][0] = scale*1.0f;
        World.m[0][1] = 0.0f;
        World.m[0][2] = 0.0f;
        World.m[0][3] = 0.0f;
        World.m[1][0] = 0.0f;
        World.m[1][1] = scale*1.0f;
        World.m[1][2] = 0.0f;
        World.m[1][3] = 0.0f;
        World.m[2][0] = 0.0f;
        World.m[2][1] = 0.0f;
        World.m[2][2] = scale*1.0f;
        World.m[2][3] = 0.0f;
        World.m[3][0] = 0.0f;
        World.m[3][1] = 0.0f;
        World.m[3][2] = 0.0f;
        World.m[3][3] = scale*1.0f;
        
        World.m[0][3] = -(rayTracingModel->minX + rayTracingModel->maxX) / 2.0f;
        World.m[1][3] = -(rayTracingModel->minY + rayTracingModel->maxY) / 2.0f;
        World.m[2][3] = -(rayTracingModel->minZ + rayTracingModel->maxZ) / 2.0f;

        createViewMatrix(View);
        // createProjectionMatrix(Projection);
        
        glUseProgram(ShaderProgram);
        glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, &World.m[0][0]);
        glUniformMatrix4fv(gViewLocation, 1, GL_TRUE, &View.m[0][0]);
        // glUniformMatrix4fv(gProjectionLocation, 1, GL_TRUE, &Projection.m[0][0]);

        // std:: cout << "entering the condition";
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, meshIndices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        

        
        // Disable depth testing for 2D rendering
        glDisable(GL_DEPTH_TEST);
        
        // Restore depth testing if it was enabled
        if (depth_test_enabled) {
            glEnable(GL_DEPTH_TEST);
        }
        
        renderRayTracedScene(World,View);
        return;
    }
    
    // Matrix setup
    Matrix4f World;
    Matrix4f View;
    Matrix4f Projection;
    
    if (customPolygonMode) {
        // No rotation in custom polygon mode
        World.InitIdentity();
        
        // Apply scaling for custom polygon mode
        float scaleValue = 0.5f; // Adjust scale as needed
        World.m[0][0] = scaleValue;
        World.m[1][1] = scaleValue;
        World.m[2][2] = scaleValue;
        
        // Center the polygon
        if (!customVertices.empty()) {
            float minX = FLT_MAX, minY = FLT_MAX;
            float maxX = -FLT_MAX, maxY = -FLT_MAX;
            
            for (const auto& v : customVertices) {
                minX = std::min(minX, v.x);
                minY = std::min(minY, v.y);
                maxX = std::max(maxX, v.x);
                maxY = std::max(maxY, v.y);
            }
            
            World.m[0][3] = -(minX + maxX) / 2.0f;
            World.m[1][3] = -(minY + maxY) / 2.0f;
            World.m[2][3] = 0.0f;
        }
    } else {
        float scale = 0.5f;
        World.m[0][0] = cosf(rotation);
        World.m[0][1] = 0.0f;
        World.m[0][2] = -sinf(rotation);
        World.m[0][3] = 0.0f;
        World.m[1][0] = 0.0f;
        World.m[1][1] = 1.0f;
        World.m[1][2] = 0.0f;
        World.m[1][3] = 0.0f;
        World.m[2][0] = sinf(rotation);
        World.m[2][1] = 0.0f;
        World.m[2][2] = cosf(rotation);
        World.m[2][3] = 0.0f;
        World.m[3][0] = 0.0f;
        World.m[3][1] = 0.0f;
        World.m[3][2] = 0.0f;
        World.m[3][3] = 1.0f;
        
        World.m[0][3] = -(model->minX + model->maxX) / 2.0f;
        World.m[1][3] = -(model->minY + model->maxY) / 2.0f;
        World.m[2][3] = -(model->minZ + model->maxZ) / 2.0f;
    }
    
    createViewMatrix(View);
    createProjectionMatrix(Projection);
    
    glUseProgram(ShaderProgram);
    glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, &World.m[0][0]);
    glUniformMatrix4fv(gViewLocation, 1, GL_TRUE, &View.m[0][0]);
    glUniformMatrix4fv(gProjectionLocation, 1, GL_TRUE, &Projection.m[0][0]);
    
    if (!showLine) {
        if (customPolygonMode) {
            if (!meshVertices.empty()) {
                glBindVertexArray(VAO);
                glDrawElements(GL_TRIANGLES, meshIndices.size() - customVertices.size(), GL_UNSIGNED_INT, 
                               (void*)(customVertices.size() * sizeof(unsigned int)));
                glDrawElements(GL_LINE_LOOP, customVertices.size(), GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }
        } else {
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, meshIndices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
    
    if (showLine) {
        Matrix4f LineWorld;
        LineWorld.InitIdentity(); // Reset to identity matrix
        
        // Apply scaling if needed
        float lineScale = 0.5f;
        LineWorld.m[0][0] = lineScale;
        LineWorld.m[1][1] = lineScale;
        LineWorld.m[2][2] = lineScale;
        
        // Apply the line-specific transformation
        glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, &LineWorld.m[0][0]);
        
        // Render the line
        glBindVertexArray(lineVAO);
        glLineWidth(2.0f); // Make line more visible
        glDrawElements(GL_LINE_STRIP, lineIndices.size(), GL_UNSIGNED_INT, 0);
        glLineWidth(1.0f); // Reset line width
        glBindVertexArray(0);
    }
}

void InitImGui(GLFWwindow *window)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	(void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

// Render ImGui
void RenderImGui() {
    ImGui::Begin("Mesh Slicer Controls");
    
    // Add ray tracing mode toggle
    if(rayTracingMode){
    if (true) {
        
        if (rayTracingMode) {
            // Sphere controls
            if (ImGui::CollapsingHeader("Sphere Properties")) {
                static float sphereReflectivity = 0.5f;
                static float sphereRadius = 1.0f;
                static float sphereColor[3] = {0.8f, 0.2f, 0.2f};
                
                ImGui::PushID("sphere_reflectivity");
                if (ImGui::DragFloat("Reflectivity", &sphereReflectivity, 0.01f, 0.0f, 1.0f)) {
                    if (!spheres.empty()) {
                        spheres[0].reflectivity = sphereReflectivity;
                    }
                }
                ImGui::PopID();
                
                ImGui::PushID("sphere_radius");
                if (ImGui::DragFloat("Radius", &sphereRadius, 0.1f, 0.1f, 5.0f)) {
                    if (!spheres.empty()) {
                        spheres[0].radius = sphereRadius;
                    }
                }
                ImGui::PopID();
                
                ImGui::PushID("sphere_color");
                if (ImGui::ColorEdit3("Color", sphereColor)) {
                    if (!spheres.empty()) {
                        spheres[0].color = glm::vec3(sphereColor[0], sphereColor[1], sphereColor[2]);
                    }
                }
                ImGui::PopID();
            }
            
            // Light controls
            if (ImGui::CollapsingHeader("Light Properties")) {
                static float lightDir[3] = {1.0f, -1.0f, -2.0f};
                static float lightColor[3] = {1.0f, 1.0f, 1.0f};
                static float lightIntensity = 1.0f;
                
                ImGui::PushID("light_direction");
                if (ImGui::DragFloat3("Direction", lightDir, 0.1f)) {
                    directionalLight.direction = glm::normalize(glm::vec3(lightDir[0], lightDir[1], lightDir[2]));
                }
                ImGui::PopID();
                
                ImGui::PushID("light_color");
                if (ImGui::ColorEdit3("Color", lightColor)) {
                    directionalLight.color = glm::vec3(lightColor[0], lightColor[1], lightColor[2]);
                }
                ImGui::PopID();
                
                ImGui::PushID("light_intensity");
                if (ImGui::DragFloat("Intensity", &lightIntensity, 0.1f, 0.0f, 5.0f)) {
                    directionalLight.intensity = lightIntensity;
                }
                ImGui::PopID();
            }
            
            // Background color control
            static float bgColor[3] = {0.2f, 0.2f, 0.2f};
            ImGui::PushID("background_color");
            if (ImGui::ColorEdit3("Background Color", bgColor)) {
                backgroundColor = glm::vec3(bgColor[0], bgColor[1], bgColor[2]);
            }
            ImGui::PopID();
        }
    }
}
    
    // Add mesh mode controls
    if (!rayTracingMode) {
        // if (ImGui::Button("Switch to Ray Tracing Mode")) {
        //     rayTracingMode = true;
        //     rotation = 0.0f;
        // }
        
        static bool showPlaneControls = true;
        if (ImGui::CollapsingHeader("Slicing Planes", &showPlaneControls))
        {
            for (int i = 0; i < slicingPlanes.size(); i++)
            {
                ImGui::PushID(i);
                
                ImGui::Text("Plane %d", i + 1);
                
                float normal[3] = {slicingPlanes[i].normal.x, slicingPlanes[i].normal.y, slicingPlanes[i].normal.z};
                if (ImGui::DragFloat3("Normal", normal, 0.01f))
                {
                    slicingPlanes[i].normal = glm::normalize(glm::vec3(normal[0], normal[1], normal[2]));
                }
                
                float distance = slicingPlanes[i].distance;
                if (ImGui::DragFloat("Distance", &distance, 0.01f))
                {
                    slicingPlanes[i].distance = distance;
                }
                
                float gap = slicingPlanes[i].gap;
                if (ImGui::DragFloat("Gap", &gap, 0.01f, 0.0f, 2.0f))
                {
                    slicingPlanes[i].gap = gap;
                }
                
                if (ImGui::Button("Apply Slice"))
                {
                    // Store original mesh data
                    std::vector<float> origVertices = meshVertices;
                    std::vector<float> origNormals = meshNormals;
                    std::vector<float> origColors = meshColors;
                    std::vector<unsigned int> origIndices = meshIndices;
                    
                    // Apply slicing
                    sliceMesh(slicingPlanes[i]);
                    
                    // Update buffers
                    glBindBuffer(GL_ARRAY_BUFFER, VBO);
                    glBufferData(GL_ARRAY_BUFFER, meshVertices.size() * sizeof(float), meshVertices.data(), GL_STATIC_DRAW);
                    
                    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
                    glBufferData(GL_ARRAY_BUFFER, meshNormals.size() * sizeof(float), meshNormals.data(), GL_STATIC_DRAW);
                    
                    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
                    glBufferData(GL_ARRAY_BUFFER, meshColors.size() * sizeof(float), meshColors.data(), GL_STATIC_DRAW);
                    
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_STATIC_DRAW);
                }
                
                ImGui::PopID();
                ImGui::Separator();
            }
          
            
            if (ImGui::Button("Reset Mesh"))
            {
                // Reset mesh to original state
                prepareMeshData();
                CreateMeshBuffers();
            }
        }
    
    
    if (ImGui::CollapsingHeader("2D Bresenham Line Drawing"))
    {
        static float x1 = 0.0f, y1 = 0.0f;
        static float x2 = 0.0f, y2 = 0.0f;
        
        ImGui::Text("Line Start Point (2D)");
        ImGui::DragFloat("X1", &x1, 0.01f, -2.0f, 2.0f);
        ImGui::DragFloat("Y1", &y1, 0.01f, -2.0f, 2.0f);
        
        ImGui::Text("Line End Point (2D)");
        ImGui::DragFloat("X2", &x2, 0.01f, -2.0f, 2.0f);
        ImGui::DragFloat("Y2", &y2, 0.01f, -2.0f, 2.0f);
        
        if (ImGui::Button("Draw 2D Line"))
        {   
           lineVertices.clear();
           lineIndices.clear();
           if (lineVAO != 0) {
            glDeleteVertexArrays(1, &lineVAO);
            lineVAO = 0;
        }
        if (lineVBO != 0) {
            glDeleteBuffers(1, &lineVBO);
            lineVBO = 0;
        }
        if (lineIBO != 0) {
            glDeleteBuffers(1, &lineIBO);
            lineIBO = 0;
        }    
            // rotation = 0;
            showLine = true;
            drawLine2DBresenham(x1, y1, x2, y2);       

            
        }
        
        if (ImGui::Button("Clear Line"))
        {
            showLine = false;
        }
        
        // Add presets for common line positions
        if (ImGui::Button("Horizontal Line"))
        {
            x1 = -1.0f; y1 = 0.0f;
            x2 = 1.0f; y2 = 0.0f;
            drawLine2DBresenham(x1, y1, x2, y2);
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Vertical Line"))
        {
            x1 = 0.0f; y1 = -1.0f;
            x2 = 0.0f; y2 = 1.0f;
            drawLine2DBresenham(x1, y1, x2, y2);
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Diagonal Line"))
        {
            x1 = -1.0f; y1 = -1.0f;
            x2 = 1.0f; y2 = 1.0f;
            drawLine2DBresenham(x1, y1, x2, y2);
        }
    }
    
    if (ImGui::CollapsingHeader("Custom Polygon & ScanlineFill")) {
        if (ImGui::Button("Enter Custom Polygon Mode")) {
            customPolygonMode = true;
            showLine = false;
            meshVertices.clear();
            meshNormals.clear();
            meshColors.clear();
            meshIndices.clear();
            customVertices.clear();
            CreateMeshBuffers(); // Update the buffers
        }
        
        if (ImGui::Button("Return to Mesh Mode")) {
            customPolygonMode = false;
            prepareMeshData();
            CreateMeshBuffers();
        }
        if (customPolygonMode) {
            static float newX = 0.0f, newY = 0.0f, newZ = 0.0f;
            ImGui::Text("Add Vertex:");
            ImGui::DragFloat("X", &newX, 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat("Y", &newY, 0.1f, -10.0f, 10.0f);
            
            if (ImGui::Button("Add Vertex")) {
                customVertices.emplace_back(newX, newY, newZ);
            }
            
            if (ImGui::Button("Clear Vertices")) {
                customVertices.clear();
            }
            

        if (ImGui::Button("Create Polygon")) {
            if (customVertices.size() >= 3) {
                meshVertices.clear();
                meshNormals.clear();
                meshColors.clear();
                meshIndices.clear();
                
                for (const auto& v : customVertices) {
                    meshVertices.push_back(v.x);
                    meshVertices.push_back(v.y);
                    meshVertices.push_back(0.0f); // Flat on Z plane
                    
                    meshNormals.push_back(0.0f);
                    meshNormals.push_back(0.0f);
                    meshNormals.push_back(1.0f);
                    
                    meshColors.push_back(1.0f);
                    meshColors.push_back(0.0f);
                    meshColors.push_back(0.0f);
                }
                
                for (size_t i = 0; i < customVertices.size(); i++) {
                    meshIndices.push_back(i);
                }
                
                initializeEdgeTable();
                scanlineFill();
                
                CreateMeshBuffers();
            }
        }
        }
    }
}
    
    ImGui::End();
}

void cleanup()
{
    if (model)
    {
        FreeOffModel(model);
    }
    
    if (lineVAO != 0)
    {
        glDeleteVertexArrays(1, &lineVAO);
    }
    if (lineVBO != 0)
    {
        glDeleteBuffers(1, &lineVBO);
    }
    if (lineIBO != 0)
    {
        glDeleteBuffers(1, &lineIBO);
    }
}

// Add a helper function for ray-triangle intersection


int main(int argc, char *argv[])
{
	glfwInit();

	// Define version and compatibility settings
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create OpenGL window and context
	GLFWwindow *window = glfwCreateWindow(theWindowWidth, theWindowHeight, "OpenGL", NULL, NULL);
	glfwMakeContextCurrent(window);

	// Check for window creation failure
	if (!window)
	{
		fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return 0;
	}

	// Initialize GLEW
	glewExperimental = GL_TRUE;
	glewInit();
	printf("GL version: %s\n", glGetString(GL_VERSION));


	onInit(argc, argv);
	InitImGui(window);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		rotation += 0.01f; // Increment rotation for animation
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		onDisplay();
		RenderImGui();

		// Render ImGui
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        
		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwTerminate();
	
	return 0;
}
