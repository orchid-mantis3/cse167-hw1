[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

// helper to find cross prd of two vectors -> return signed area of 
// parallelogram that the vectors span
// + = sec vector counterclockwise(y-up); - = sec vector clockwise(y-down)
float cross2(float2 u, float2 v) {
    return u.x * v.y - u.y * v.x;
}
// helper to check if pixel ctr p is inside tri
bool eval_tri(float2 sp0, float2 sp1, float2 sp2, float inv_d0, float inv_d1,
                float inv_d2, float2 p, out float depth, out float3 b) {
                    // area of whole tri.
                    float area = cross2(sp1 - sp0, sp2 - sp0);
                    if (area == 0) {
                        depth = 0.0;
                        b = float3(0.0);
                        return false;
                    }
                    // barryc. coords for tri. ratios for proper perspective
                    // area of tri. p, p1, p2; opp. to p0
                    float bp0 = cross2(sp1 - p,  sp2 - p) / area;
                    // area of tri. p0, p, p2; opp. to p1
                    float bp1 = cross2(sp2 - p, sp0 - p) / area;
                    // area of tri. p0, p1, p; opp. to p2
                    float bp2 = cross2(sp0 - p, sp1 - p) / area;

                    if (bp0 < 0 || bp1 < 0 || bp2 < 0) {
                        depth = 0.0;
                        b = float3(0.0);
                        return false;
                    }

                    // Using the projected barycentric coordinates and the depth at the three vertices, we convert them
                    // to the original barycentric coordinates using Eq. 15
                    float num0 = bp0 * inv_d0;
                    float num1 = bp1 * inv_d1;
                    float num2 = bp2 * inv_d2;
                    float denom = num0 + num1 + num2;

                    // check if pt is behind camera (so <= 0)
                    if (denom <= 0) {
                        depth = 0.0;
                        b = float3(0.0);
                        return false;
                    }
                    b = float3(num0, num1, num2) / denom;
                    depth = b.x / inv_d0 + b.y / inv_d1 + b.z / inv_d2;
                    return true;
                }
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO
    if (tid >= num_triangles) return;
    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];
    
    // min and max of screen coords
    int x_min = max(0, (int)floor(min(v0.x, min(v1.x, v2.x))));
    int y_min = max(0, (int)floor(min(v0.y, min(v1.y, v2.y))));
    int x_max = min(int(width) - 1, (int)floor(max(v0.x, max(v1.x, v2.x))));
    int y_max = min(int(height)-1, (int)floor(max(v0.y, max(v1.y, v2.y))));
    
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            float2 p = float2(x + 0.5, y + 0.5);
            // cast x & y as uint for buffer idx to not mix signed/unsigned
            uint idx = uint(y) * width + uint(x);
            float depth;
            float3 b;
            // check if this tri covers this px, if so how far is it
            // via 3 screen posit., 3 inverse depths, p
            // if true fill in depth
            if (eval_tri(v0.xy, v1.xy, v2.xy, v0.z, v1.z, v2.z, p, depth, b)) {
                InterlockedMin(depth_buffer[idx], asuint(depth));
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO
    if (tid >= num_triangles) return;
    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];
    
    // min and max of screen coords
    int x_min = max(0, (int)floor(min(v0.x, min(v1.x, v2.x))));
    int y_min = max(0, (int)floor(min(v0.y, min(v1.y, v2.y))));
    int x_max = min(int(width) - 1, (int)floor(max(v0.x, max(v1.x, v2.x))));
    int y_max = min(int(height)-1, (int)floor(max(v0.y, max(v1.y, v2.y))));
    
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            float2 p = float2(x + 0.5, y + 0.5);
            // cast x & y as uint for buffer idx to not mix signed/unsigned
            uint idx = uint(y) * width + uint(x);
            float depth;
            float3 b;
            // check if this tri covers this px, if so how far is it
            if (eval_tri(v0.xy, v1.xy, v2.xy, v0.z, v1.z, v2.z, p, depth, b)) {
                // depth comparison; pick lowest #id tri. at that px
                if (asuint(depth) == depth_buffer[idx]) {
                    InterlockedMin(triangle_id_buffer[idx], tid);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // HW 2.5 TODO
    uint idx = tid.y * width + tid.x;
    uint id = triangle_id_buffer[idx];
    uint3 f = faces[id];

    // fetch new vertices
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    // fetch colors
    float3 c0 = vertex_colors[f.x];
    float3 c1 = vertex_colors[f.y];
    float3 c2 = vertex_colors[f.z];

    float2 p = float2(tid.x + 0.5, tid.y + 0.5);

    // computer barryc. coords
    float depth;
    float3 b;
    // check if this tri doesn't cover this px, if so return
    if (!eval_tri(v0.xy, v1.xy, v2.xy, v0.z, v1.z, v2.z, p, depth, b)) return;
    image[tid] = float4(b.x * c0 + b.y * c1 + b.z * c2, 1.0);
    
}
