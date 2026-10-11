import hw1_common;


[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    float2 q = float2(tid.x + 0.5, (float)height - (tid.y + 0.5));

    if (use_fill_color && is_closed) {
        // non-zero winding rule
        int count = 0;
        for (int i = 0; i < num_polyline; i++) {
            float2 p0 = polyline[i];
            float2 p1 = polyline[(i + 1) % num_polyline];
            bool is_upward = p1.y > p0.y;
            bool crosses = (p0.y <= q.y && p1.y > q.y) || (p1.y <= q.y && p0.y > q.y);
            if (crosses) {
                float t = (q.y - p0.y) / (p1.y - p0.y);
                float x_cross = p0.x + t * (p1.x - p0.x);
                if (x_cross > q.x)
                    count += is_upward ? 1 : -1;
            }
        }
        if (count != 0)
            image[tid] = float4(fill_color, 1.0);
    }

    if (use_stroke_color) {
        int num_segments = is_closed ? num_polyline : num_polyline - 1;
        for (int i = 0; i < num_segments; i++) {
            float2 p0 = polyline[i];
            float2 p1 = polyline[(i + 1) % num_polyline];

            bool start = (!is_closed && i == 0);
            bool end   = (!is_closed && i == (num_segments - 1));

            float2 d = p1 - p0;
            float len = length(d);

            float l = dot(d, q - p0) / len;
            if ((!start || l >= 0) && (!end || l <= len)) {
                float t = clamp(l, 0.0, len);
                float2 q_prime = p0 + (d / len) * t;
                if (length(q - q_prime) <= stroke_width * 0.5) {
                    image[tid] = float4(stroke_color, 1.0);
                    break;
                }
            }
        }
    }
}
