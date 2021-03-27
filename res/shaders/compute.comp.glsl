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

const mat3 laplacian = mat3(0,1,0,1,-4,1,0,1,0);

highp float random(vec2 co)
{
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 bounds = imageSize(img);

    if(uniforms.init_image) {
        imageStore(img, pos, 2 * vec4(random(pos),random(pos + 1000)*200-100,0,0));
    }

    retirePhase();

    

    if(pos.x > 0 && pos.y > 0 && pos.x < bounds.x-1 && pos.y < bounds.y-1) {


        int time_res = 50;

        float dt = uniforms.delta_time / time_res;

        for(int i = 0; i < time_res; i++) {
            float l = 0;
            for(int i = -1; i <= 1; i++) {
                for(int j = -1; j <= 1; j++) {
                    l += imageLoad(img, ivec2(pos.x + i, pos.y + j)).x * laplacian[i+1][j+1]; 
                }
            }
            vec4 pixel = imageLoad(img, pos);
            
        

            pixel.y += (500*l - 0.0 * pixel.y) * dt;
            
            if(ivec2(uniforms.touch_coords * bounds) == pos) {
                pixel.y += uniforms.touch_force * dt * 20;
            } 

            pixel.x += pixel.y * dt;

             retirePhase();

            imageStore(img, pos, pixel);

        }


       
    }
}