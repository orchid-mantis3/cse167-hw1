import hw1_common;

float2 to_obj_space(float2 q, float3x3 world_to_obj) {
    float3 p = mul(world_to_obj, float3(q, 1.0));
    return p.xy;
}

// helper func to handle winding of one root
int bezier_crossing(float t, float2 p0, float2 p1, float2 p2, float2 p) {
    // if t is not in range [0,1)
    if (!(0 <= t && t < 1)) return 0;
    // x-position at t 
    float x = ((1-t)*(1-t)) * p0.x + (2 * (1-t) * t) * p1.x + t*t * p2.x;

    // if x isnt to the right of p.x
    if (!(x > p.x)) return 0;

    // slope of y at t
    float dy = 2 * (1-t) * (p1.y - p0.y) + 2 * t * (p2.y - p1.y);

    // return 1 if slope is up and -1 if slope is down
    return dy > 0 ? 1 : -1;
}

// returns the color at world-space sample point q, given scene data
float3 sample_color(float2 q,
                    float3 background,
                    StructuredBuffer<Shape> shapes,
                    int num_shapes,
                    StructuredBuffer<Circle> circles,
                    StructuredBuffer<Polyline> polylines,
                    StructuredBuffer<QuadraticBezier> quadratic_beziers,
                    StructuredBuffer<float2> points) {
    float3 color = background;

    for (int s = num_shapes  - 1; s >= 0; s--) {
        Shape shape = shapes[s];
        float2 p = to_obj_space(q, shape.world_to_obj);

        if (shape.type == ShapeType::Circle) {
            Circle c = circles[shape.index];
            float d = length(p - c.center);

            // create blend so that the color of the layers blend with the one behind it since they're all semi-transparent
            if (shape.use_fill && d <= c.radius) {
                        // 1 - w * a + w * b
                color = (1 - shape.fill_rgba.w) * color + shape.fill_rgba.w * shape.fill_rgba.rgb;
            }
            if (shape.use_stroke && (abs(d - c.radius) <= shape.stroke_width * 0.5)) {
                color = (1 - shape.stroke_rgba.w) * color + shape.stroke_rgba.w * shape.stroke_rgba.rgb;
            } 

        }
        
        if (shape.type == ShapeType::Polyline) {
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
                            color = (1 - shape.stroke_rgba.w) * color + shape.stroke_rgba.w * shape.stroke_rgba.rgb;
                            break;
                        }
                    }
                }
            }
            // fill test
            if (shape.use_fill && pl.closed) {
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
                    color = (1 - shape.fill_rgba.w) * color + shape.fill_rgba.w * shape.fill_rgba.rgb;
                }
            }
    }
    // quad beziers
    if (shape.type == ShapeType::QuadraticBezier) {
        QuadraticBezier qb = quadratic_beziers[shape.index];
        int n = qb.pts_end - qb.pts_begin;
                                    //closed : open
        int num_segments = qb.closed ? n / 2 : (n - 1) / 2 ;

        // fill test
        if (shape.use_fill && qb.closed) {
            int ctr = 0;
            for (int i = 0; i < num_segments; i++) {
                // start, ctrl, and end of segment
                float2 p0 = points[qb.pts_begin + 2 * i];
                float2 p1 = points[qb.pts_begin + 2 * i + 1];
                float2 p2 = points[qb.pts_begin + ((2 * i + 2) % n)];

                float a = p0.y - 2.0 * p1.y + p2.y;
                float b = 2.0 * (p1.y - p0.y);
                float c = p0.y - p.y;

                // if a(makes quad eq bt + c = 0 --> root will be t = -c/b) and b(makes y() basically a const in whole seg) are basically 0 so then itll never cross the ray
                // bezier_crossing takes the sole root and checks if its in range([0,1)), to the right of the sample, and its gen. dir
                if (abs(a) < 1e-6) {
                    if (abs(b) > 1e-6) {
                        ctr += bezier_crossing(-c / b, p0, p1, p2, p);
                    }
                } else {
                    float2 roots; 
                    if (solve_quadratic(a, b, c, roots)) {
                        ctr += bezier_crossing(roots.x, p0, p1, p2, p);
                        ctr += bezier_crossing(roots.y, p0, p1, p2, p);
                    }
                }
                
            }
            if (ctr != 0) {
                color = (1 - shape.fill_rgba.w) * color + shape.fill_rgba.w * shape.fill_rgba.rgb;
            }
        }

        // stroke test
        if (shape.use_stroke) {
            float min_dist = 1e10;
            for (int i = 0; i < num_segments; i++) {
                // start, ctrl, and end of segment
                float2 p0 = points[qb.pts_begin + 2 * i];
                float2 p1 = points[qb.pts_begin + 2 * i + 1];
                float2 p2 = points[qb.pts_begin + ((2 * i + 2) % n)];

                float2 A = p0 - 2.0 * p1 + p2;
                float2 B = 2.0 * (p1 - p0);
                float2 C = p0 - p;
                float3 roots;
                bool start = (!qb.closed && i == 0);
                bool end = (!qb.closed && i == (num_segments - 1));
                int num_roots = solve_cubic(2.0 * dot(A, A), 3.0 * dot(A, B), 2.0 * dot(A, C) + dot(B,B), dot(B, C), roots);
                for (int k = 0; k < num_roots; k++) {
                    // flat-cap check
                    float raw_t = roots[k];
                    if ((start && raw_t < 0.0) || (end&& raw_t > 1.0)) continue;
                    // clamp roots[k] to [0,1] into t
                    float t = clamp(roots[k], 0.0, 1.0);
                    // pt on curve
                    float2 q = A * t * t + B * t + p0;
                    min_dist = min(min_dist, length(p - q));
                }
                if (!(start && dot(p - p0, B) < 0.0)) { min_dist = min(min_dist, length(p - p0)); }

                if (!(end && dot(p - p2, 2.0 * A + B) > 0.0)) { min_dist = min(min_dist, length(p - p2)); }

            }
            if (min_dist <= shape.stroke_width * 0.5) {
                // blend
                color = (1 - shape.stroke_rgba.w) * color + shape.stroke_rgba.w * shape.stroke_rgba.rgb;
            }
        }
    }
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
                  StructuredBuffer<QuadraticBezier> quadratic_beziers, 
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
            color_sum += sample_color(q, background, shapes, num_shapes, circles, polylines, quadratic_beziers, points);
        }
    }

    float n = float(samples_per_axis * samples_per_axis); // total number of samples
    image[tid] = float4(color_sum / n, 1.0); // divides by total number of samples, maps avg color to that pixel
}