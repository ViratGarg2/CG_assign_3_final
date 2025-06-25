#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec3 Color;

uniform mat4 gWorld;
uniform mat4 gView;
uniform mat4 gProjection;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Color;
} vs_out;

void main() {
    vs_out.FragPos = vec3(gWorld * vec4(Position, 1.0));
    vs_out.Normal = mat3(transpose(inverse(gWorld))) * Normal;
    vs_out.Color = Color;
    gl_Position = gProjection * gView * gWorld * vec4(Position, 1.0);
}
