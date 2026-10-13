import hw1_common;


//from linear transformation slides: Each shape is defined in its own simple object space
// and a transformation F places that shape into the canvas
// where world_to_obj is F_inverse; compute p = F_inverse x q for each pixel
float2 to_obj_space(float2 q, float3x3 world_to_obj){
    float3 p = mul(world_to_obj, float3(q, 1.0)); // translation can't be written as 2x2 matrix, so use homogeneous coordinates
    // append a 1 to the point and use a 3x3 matrix
    return p.xy;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
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

    // TODO: your code here

    float2 q = float2(tid.x + 0.5, (float)height - (tid.y + 0.5));
    float3 color = background;
    
    for (int s = 0; s < num_shapes; s++) { // index 0 is the front-most shape; for each pixel go front to back and the first shape that covers the pixel claims it
        Shape shape = shapes[s];
        float2 p = to_obj_space(q, shape.world_to_obj);
        bool hit = false;

        if (shape.type == ShapeType::Circle) {
            Circle c = circles[shape.index];
            float d = length(p - c.center);

            // test stroke first because stroke's inner half overlaps with fill region, so stroke should draw on top 
            if (shape.use_stroke && (abs(d - c.radius) <= shape.stroke_width * 0.5)) {
                // |d-r| <= w/2 selects the ring of points whose distance from center is within w/2 of the radius
                color = shape.stroke_rgba.rgb;
                hit = true;
            } else if (shape.use_fill && d <= c.radius) {  // point is inside a circle if its distance to the center is at most the radius
                color = shape.fill_rgba.rgb;
                hit = true;
            }

        } else if (shape.type == ShapeType::Polyline) {
            Polyline pl = polylines[shape.index];
            int n = pl.pts_end - pl.pts_begin; // get number of points in polyline

            // stroke test using same algorithm as 1.2
            if (shape.use_stroke) {
                int num_segments = pl.closed ? n : n - 1; // since closed polyline has one extra edge
                for (int i = 0; i < num_segments; i++) {
                    float2 p0 = points[pl.pts_begin + i];
                    float2 p1 = points[pl.pts_begin + (i + 1) % n];

                    bool start = (!pl.closed && i == 0);
                    bool end = (!pl.closed && i == (num_segments - 1));

                    float2 d = p1 - p0;
                    float len = length(d);

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

            // fill test; same as 1.2
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
                        if (x_cross > p.x) {
                            count += is_upward ? 1 : -1;
                        }
                    }
                }
                if (count != 0) {
                    color = shape.fill_rgba.rgb;
                    hit = true;
                }
            }
        }

        if (hit) break;   // nothing behind a claimed pixel can show, so if pixel is already claimed, exit loop
    }

    image[tid] = float4(color, 1.0);
}
