[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    
    
    float2 pixel_center = float2(tid.x + 0.5, (float)height - (tid.y + 0.5)); // RWTexture2D is indexed top-down; need to flip y
    float distance = length(pixel_center - center); // calculates Euclidean distance

    if(distance <= radius){ // distance from pixel center to circle center <= radius ?
        image[tid] = float4(color, 1.0);

    }else{
        image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    }
}
