#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPos, 1.0);
    gl_Position.y = -gl_Position.y;
    fragColor = inColor;
}
