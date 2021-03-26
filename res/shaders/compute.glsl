#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    float delta_time;
} uniforms;

layout(set = 0, binding = 0, rgba8) uniform image2D img;
layout(set = 0, binding = 1, rgba8) uniform image2D vimg;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0;

    vec4 to_write = imageLoad(vimg, ivec2(gl_GlobalInvocationID.xy));



    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}