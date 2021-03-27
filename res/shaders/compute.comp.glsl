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
} uniforms;

layout(set = 0, binding = 0, rg32f) uniform image2D img;

const mat3 laplacian = mat3(1,1,1,1,-8,1,1,1,1);

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 bounds = imageSize(img);

    if(uniforms.init_image) {
        imageStore(img, pos, vec4(0));
    }


    if(pos.x > 0 && pos.y > 0 && pos.x < bounds.x-1 && pos.y < bounds.y-1) {
        float l = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                l += imageLoad(img, ivec2(pos.x + i, pos.y + j)).x * laplacian[i+1][j+1]; 
            }
        }
        vec4 pixel = imageLoad(img, pos);
        
        pixel.x += (4*l) * uniforms.delta_time;
        
        if(ivec2(uniforms.touch_coords * bounds) == pos) {
            pixel.x += uniforms.touch_force * uniforms.delta_time;
        } 

        //pixel.x += pixel.y * uniforms.delta_time;


        retirePhase();

        imageStore(img, pos, pixel);
    }
}