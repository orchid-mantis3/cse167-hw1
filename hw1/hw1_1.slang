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
    // TODO: your code here
    center = float2(center.x, height - center.y);
    if (length(float2(tid) - center) <= radius) {
            image[tid] = float4(color.x, color.y, color.z, 1.0);
        } else{
            image[tid] = float4(0.5, 0.5, 0.5, 1.0);
        }
}
