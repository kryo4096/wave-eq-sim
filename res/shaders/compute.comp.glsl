#version 450

#define M_PI 3.1415926535897932384626433832795

#ifndef retirePhase
void retirePhase() { memoryBarrierShared(); barrier(); }
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    float delta_time;
    bool init_image;
    vec2 touch_coords;
    float touch_force;
    float wave_speed;
    float damping;

} u;

layout(set = 0, binding = 0, rg32f) uniform image2D back_buf;
layout(set = 0, binding = 1, rg32f) uniform image2D render_buf;

const mat3 laplacian = mat3(0,1,0,1,-4,1,0,1,0);

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 bounds = imageSize(back_buf);

    if(u.init_image) {
        imageStore(render_buf, pos, vec4(0));
        imageStore(back_buf, pos, vec4(0));
    } else if(pos.x > 0 && pos.y > 0 && pos.x < bounds.x-1 && pos.y < bounds.y-1) {

        float l = 0;

        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                l += imageLoad(back_buf, ivec2(pos.x + i, pos.y + j)).x * laplacian[i+1][j+1]; 
            }
        }

        vec4 pixel = imageLoad(back_buf, pos);

        pixel.y += (u.wave_speed*u.wave_speed*l - u.damping * pixel.y) * u.delta_time;
        
        if(ivec2(u.touch_coords * bounds) == pos)  {
            pixel.y += u.touch_force * u.delta_time;
        } 

        pixel.x += pixel.y * u.delta_time;

        imageStore(render_buf, pos, pixel);

    }
}