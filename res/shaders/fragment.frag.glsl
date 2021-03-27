#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PushConstantData {
    vec3 color;
} uniforms;

layout(set = 0, binding = 0) uniform sampler2D tex;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

const float SCALE = 100;

void main() {
    vec4 pixel = texture(tex, tex_coords);


    f_color = vec4(hsv2rgb(vec3(0.5 + pixel.x / SCALE, 1.0, 0.3 * abs(pixel.x / SCALE * 2.))), 1);


    if(pixel.x < 0) {
        f_color.b += -pixel.x/SCALE;
        if(abs(pixel.x) > SCALE) {
            f_color.rg += vec2(tanh((abs(pixel.x) - SCALE)/SCALE));
        }
    } else {
        f_color.r += pixel.x/SCALE;
        if(abs(pixel.x) > SCALE) {
            f_color.gb += vec2(tanh((abs(pixel.x) - SCALE)/SCALE));
        }
    }



}