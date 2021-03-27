#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(push_constant) uniform PushConstantData {
    float time;
} uniforms;

void main() {
    
    
    gl_Position = vec4(position, 0.0, 1.0);


    tex_coords = (position + 1) / 2;
}