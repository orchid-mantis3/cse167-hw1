import hw1_common;

float2 to_obj_space(float2 q, float3x3 world_to_obj) {
    float3 p = mul(world_to_obj, float3(q, 1.0));
    return p.xy;
}

// returns the color at world-space sample point q, given scene data
float3 sample_color(float2 q,
                    float3 background,
                    StructuredBuffer<Shape> shapes,
                    int num_shapes,
                    StructuredBuffer<Circle> circles,
                    StructuredBuffer<Polyline> polylines,
                    StructuredBuffer<float2> points) {
    float3 color = background;

    for (int s = 0; s < num_shapes; s++) {
        Shape shape = shapes[s];
        float2 p = to_obj_space(q, shape.world_to_obj);
        bool hit = false;

        if (shape.type == ShapeType::Circle) {
            Circle c = circles[shape.index];
            float d = length(p - c.center);

            if (shape.use_stroke && (abs(d - c.radius) <= shape.stroke_width * 0.5)) {
                color = shape.stroke_rgba.rgb;
                hit = true;
            } else if (shape.use_fill && d <= c.radius) {
                color = shape.fill_rgba.rgb;
                hit = true;
            }

        } else if (shape.type == ShapeType::Polyline) {
            Polyline pl = polylines[shape.index];
            int n = pl.pts_end - pl.pts_begin;

            // stroke test
            if (shape.use_stroke) {
                int num_segments = pl.closed ? n : n - 1;
                for (int i = 0; i < num_segments; i++) {
                    float2 p0 = points[pl.pts_begin + i];
                    float2 p1 = points[pl.pts_begin + (i + 1) % n];

                    bool start = (!pl.closed && i == 0);
                    bool end = (!pl.closed && i == (num_segments - 1));

                    float2 d = p1 - p0;
                    float len = length(d);
                    if (len < 1e-6) continue;

                    float l = dot(d, p - p0) / len;
                    if ((!start || l >= 0) && (!end || l <= len)) {
                        float t = clamp(l, 0.0, len);
                        float2 q_prime = p0 + (d / len) * t;
                        if (length(p - q_prime) <= shape.stroke_width * 0.5) {
                            color = shape.stroke_rgba.rgb;
                            hit = true;
                            break;
                        }
                    }
                }
            }

            // fill test
            if (!hit && shape.use_fill && pl.closed) {
                int count = 0;
                for (int i = 0; i < n; i++) {
                    float2 p0 = points[pl.pts_begin + i];
                    float2 p1 = points[pl.pts_begin + (i + 1) % n];
                    bool is_upward = p1.y > p0.y;
                    bool crosses = (p0.y <= p.y && p1.y > p.y) || (p1.y <= p.y && p0.y > p.y);
                    if (crosses) {
                        float t = (p.y - p0.y) / (p1.y - p0.y);
                        float x_cross = p0.x + t * (p1.x - p0.x);
                        if (x_cross > p.x)
                            count += is_upward ? 1 : -1;
                    }
                }
                if (count != 0) {
                    color = shape.fill_rgba.rgb;
                    hit = true;
                }
            }
        }

        if (hit) break; // front-to-back
    }

    return color;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    

    // average over NxN grid of subpixels where N=samples_per_axis
    float3 color_sum = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < samples_per_axis; i++) {
        for (int j = 0; j < samples_per_axis; j++) {
            // center of subpixel (i, j) within this pixel
            float2 q = float2(
                float(tid.x) + (float(i) + 0.5) / float(samples_per_axis),
                float(height) - float(tid.y) - (float(j) + 0.5) / float(samples_per_axis)
            );
            color_sum += sample_color(q, background, shapes, num_shapes, circles, polylines, points);
        }
    }

    float n = float(samples_per_axis * samples_per_axis); // total number of samples
    image[tid] = float4(color_sum / n, 1.0); // divides by total number of samples, maps avg color to that pixel
}