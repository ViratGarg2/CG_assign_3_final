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
#include <algorithm>


#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "file_utils.h"
#include "math_utils.h"
#include "graphics_utils.h"
#include "light.h"
#include "scanline.h"
#include "mesh.h"
#include "OFFReader.h"
#include "geometry/Ray.h"
#include "geometry/Sphere.h"
#include "geometry/Cube.h"
#include "geometry/Plane.h"
#include "geometry/Point3D.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

glm::vec3 cameraPos   = glm::vec3(0.0f, 2.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float yaw   = -90.0f;
float pitch =  0.0f;
float lastX =  700.0f / 2.0;
float lastY =  700.0f / 2.0;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool mouseCameraControlEnabled = false;

int theWindowWidth = 700, theWindowHeight = 700;
int theWindowPositionX = 40, theWindowPositionY = 40;

vector<Cube> cubes;

std::vector<Plane> slicingPlanes = {
    Plane(glm::vec3(1.0f, 0.0f, 0.0f), 0.0f, 0.0f, glm::vec3(1.0f, 0.0f, 0.0f)),
    Plane(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, glm::vec3(0.0f, 1.0f, 0.0f)),
    Plane(glm::vec3(0.0f, 0.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f, 0.0f, 1.0f)),
    Plane(glm::vec3(1.0f, 1.0f, 1.0f), 0.5f, 0.0f, glm::vec3(1.0f, 1.0f, 0.0f))
};

bool rayTracingMode = false;
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
    if (fabs(a) < EPSILON) return false;

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
    glm::vec3 cameraRight = glm::vec3(viewMat[0]);  // Right direction

    glm::vec3 origin = cameraPos;
    glm::vec3 horizontal = viewportWidth * cameraRight;
    glm::vec3 vertical = viewportHeight * cameraUp;
    glm::vec3 lowerLeftCorner = origin + (cameraFront * focalLength) - (horizontal / 2.0f) - (vertical / 2.0f);

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


char theProgramTitle[] = "Sample";

bool isFullScreen = false;
bool isAnimating = true;
float rotation = 0.0f;
GLuint VBO, VAO, IBO;
GLuint normalVBO, colorVBO;
GLuint gWorldLocation, gViewLocation, gProjectionLocation;
GLuint ShaderProgram;

const int ANIMATION_DELAY = 20;
const char *pVSFileName = "shaders/shader.vs";
const char *pFSFileName = "shaders/shader.fs";

OffModel* model = nullptr;
std::vector<float> meshVertices;
std::vector<float> meshNormals;
std::vector<float> meshColors;
std::vector<unsigned int> meshIndices;
std::vector<float> originalVertices;



bool showScanlineFill = false;
bool customPolygonMode = false;

std::vector<Point3D> customVertices;

std::vector<float> lineVertices;
std::vector<unsigned int> lineIndices;
GLuint lineVAO = 0, lineVBO = 0, lineIBO = 0;
bool showLine = false;

std::vector<std::vector<Edge>> edgeTable;
std::vector<ActiveEdge> activeEdgeTable;
int scanlineMinY = 0;
int scanlineMaxY = 0;


void resetCamera() {
    cameraPos   = glm::vec3(0.0f, 2.0f, 3.0f);
    cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);
    yaw   = -90.0f;
    pitch =  0.0f;
    firstMouse = true;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (rayTracingMode && mouseCameraControlEnabled) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    if (!rayTracingMode || !mouseCameraControlEnabled) return;

    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (!rayTracingMode || !mouseCameraControlEnabled) {
        firstMouse = true; 
        return;
    }

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw   += xoffset;
    pitch += yoffset;

    if(pitch > 89.0f)
        pitch = 89.0f;
    if(pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && rayTracingMode)
    {
        mouseCameraControlEnabled = !mouseCameraControlEnabled;
        if (!mouseCameraControlEnabled) {
            resetCamera();
        }
        firstMouse = true;
    }
}


void createProjectionMatrix(Matrix4f& Projection) {
    float aspectRatio = (float)theWindowWidth / (float)theWindowHeight;
    float scale = 1.0f / model->extent * 1.5f; // Scale based on model size
    
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
    glm::vec3 lookTarget = cameraPos + cameraFront;
    
    Vector3f position = {cameraPos.x, cameraPos.y, cameraPos.z};
    Vector3f target = {lookTarget.x, lookTarget.y, lookTarget.z};
    Vector3f up = {cameraUp.x, cameraUp.y, cameraUp.y};
    
 
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

void drawLine2DBresenham(float x1, float y1, float x2, float y2) {
    drawLine2DBresenham(x1, y1, x2, y2, lineVertices, lineIndices, lineVAO, lineVBO, lineIBO);
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
    char modelFilename[] = "models/1grm.off";
    model = readOffFile(modelFilename);
    if (!model) {
        fprintf(stderr, "Failed to load models/1grm.off\n");
        exit(1);
    }
    prepareMeshData(model, meshVertices, meshNormals, meshColors, meshIndices, originalVertices);
    CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices);

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
        float scaleValue = 2.0f; // Adjust scale as needed
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
        float lineScale = 2.0f;
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
    
    bool oldRayTracingMode = rayTracingMode;
    ImGui::Checkbox("Ray Tracing Mode", &rayTracingMode);

    if (oldRayTracingMode != rayTracingMode) {
        char isoFile[] = "models/isohedron.off";
        char grmFile[] = "models/1grm.off";
        char* modelFile = rayTracingMode ? isoFile : grmFile;
        if (model) {
            FreeOffModel(model);
        }
        model = readOffFile(modelFile);
        if (!model) {
            fprintf(stderr, "Failed to load %s\n", modelFile);
            exit(1);
        }
        prepareMeshData(model, meshVertices, meshNormals, meshColors, meshIndices, originalVertices);
        CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices);
        rayTracingModel = model;
    }
    
    if (rayTracingMode) {
        ImGui::Text("Camera control: %s", mouseCameraControlEnabled ? "Mouse" : "Default");
        ImGui::Text("Press SPACE to toggle camera control.");
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
                    sliceMesh(slicingPlanes[i], meshVertices, meshNormals, meshColors, meshIndices);
                    
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
                prepareMeshData(model, meshVertices, meshNormals, meshColors, meshIndices, originalVertices);
                CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices);
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
            CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices); // Update the buffers
        }
        
        if (ImGui::Button("Return to Mesh Mode")) {
            customPolygonMode = false;
            prepareMeshData(model, meshVertices, meshNormals, meshColors, meshIndices, originalVertices);
            CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices);
        }
        if (customPolygonMode) {
            static float newX = 0.0f, newY = 0.0f, newZ = 0.0f;
            ImGui::Text("Add Vertex:");
            ImGui::DragFloat("X", &newX, 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat("Y", &newY, 0.1f, -10.0f, 10.0f);
            
            if (ImGui::Button("Add Vertex")) {
                customVertices.emplace_back(newX, newY, newZ);
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
                
                initializeEdgeTable(customVertices, edgeTable, scanlineMinY, scanlineMaxY);
                scanlineFill(edgeTable, activeEdgeTable, scanlineMinY, scanlineMaxY, meshVertices, meshNormals, meshColors, meshIndices, [&](){ CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices); });
                
                CreateMeshBuffers(VAO, VBO, normalVBO, colorVBO, IBO, meshVertices, meshNormals, meshColors, meshIndices);
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



int main(int argc, char *argv[])
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create OpenGL window and context
    GLFWwindow *window = glfwCreateWindow(theWindowWidth, theWindowHeight, "OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);

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
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

		rotation += 0.01f; // Increment rotation for animation
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		onDisplay();
		RenderImGui();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        
		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwTerminate();
	
	return 0;
}
