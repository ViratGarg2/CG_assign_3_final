#version 330 core

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Color;
} fs_in;

out vec4 outColor;

void main()
{
    // Use the interpolated color from vertex shader
    outColor = vec4(fs_in.Color, 1.0);
}