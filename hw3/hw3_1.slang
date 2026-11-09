import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // sample pixel center, given image plane z = -1
    float a = float(width)/float(height); // aspect ratio
    float3 o = float3(0.0, 0.0, 0.0); // camera origin
    float3 d = float3(-a + 2.0 * a * (float(tid.x)+0.5) / float(width),
                        1.0 - 2.0 * (float(tid.y) + 0.5) / float(height),
                        -1.0); // ray pointing from origin through pixel center on image plane at z=-1
    
    float t_closest = -1.0; // no sphere hit yet(default)
    float3 n = float3(0.0, 0.0, 0.0);
    bool hit = false;

    for (int i = 0; i < num_spheres; i++) {
        float3 xc = spheres[i].center; 
        float r = spheres[i].radius;

        // eq (1) At^2 + Bt + C = 0
        float3 oc = o - xc;
        float A = dot(d,d);
        float B = 2.0 * dot(d, oc);
        float C = dot(oc,oc) - r * r;

        float2 t; // will store the 2 quadratic roots 
        if (solve_quadratic(A, B, C, t)) {
            float t_sphere = -1.0;
            if (t.x > 0.0){
                t_sphere = t.x;
            } 
            if (t.y > 0.0 && (t_sphere < 0.0 || t.y < t_sphere)){ // replace first root if it's closer
                t_sphere = t.y;
            } 

            if (t_sphere > 0.0 && (t_closest < 0.0 || t_sphere < t_closest)) { // first hit or closer than curr
                t_closest = t_sphere;
                float3 p = o + t_closest * d; // hit point
                n = normalize(p - xc); // outward sphere normal
                hit = true;
            }
        }
    }

    float3 color = background;
    if (hit) {
        color = n * 0.5 + 0.5;
    }
    image[tid] = float4(color, 1.0);
}
