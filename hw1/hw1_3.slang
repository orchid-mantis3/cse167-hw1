import hw1_common;

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

    image[tid] = float4(1.0);
    // TODO: your code here
    image[tid] = float4(background, 1.0);
    float2 start_pt,end_pt,pixel;
    pixel = float2(tid);
    pixel.x += 0.5;
    // fix y-orientation
    pixel.y = float(height) - 1 - pixel.y + 0.5;

    for (int i = num_shapes - 1; i >= 0; i--) {
        Shape curr_shape = shapes[i];
        if (shapes[i].type == ShapeType.Circle) {
            Circle circ = circles[curr_shape.index];
            float dist_to_center = length(pixel - circ.center);
            // if filled and in bounds color shape
            if (curr_shape.use_fill && dist_to_center <= circ.radius) {
                image[tid] = curr_shape.fill_rgba;
            }
            // test if using stroke and within bounds of edge
            if (curr_shape.use_stroke && (abs(dist_to_center - circ.radius) <= (curr_shape.stroke_width / 2.0))) {
                    image[tid] = curr_shape.stroke_rgba;
            }
        } else if (shapes[i].type == ShapeType.Polyline) {
            Polyline polyline = polylines[curr_shape.index];
            int ctr = 0, next, num_pts = polyline.pts_end - polyline.pts_begin;

            // if polyline is closed fill it
            if (polyline.closed){
                for (int j = polyline.pts_begin; j < polyline.pts_end; j++) {
                    next = polyline.pts_begin + ((j - polyline.pts_begin + 1) % num_pts);
                    start_pt = points[j];
                    end_pt = points[next];

                    // verify if edge crosses through pixel's ray, as in above or below
                    if ((start_pt.y > pixel.y) != (end_pt.y > pixel.y)) {
                        // where edge reaches pixel.y
                        float t = (pixel.y - start_pt.y) / (end_pt.y - start_pt.y);
                                                            //portion of distance traveled so far via start_pt.x
                        float cross_x = start_pt.x  + (t * (end_pt.x - start_pt.x));
                        if (cross_x > pixel.x) {
                            // edge dir; ++ for upward dir and -- for downward dir
                            if (end_pt.y > start_pt.y) {
                                ctr++;
                            } else {
                                ctr--;
                            }
                        }
                    }
                }
                // non-zero rule
                if (ctr != 0 && curr_shape.use_fill && polyline.closed) {
                    image[tid] = curr_shape.fill_rgba;
                }
            }
            // deals with strokes
            if (curr_shape.use_stroke) {
                float2 seg_vector, start_to_px;
                int num_segments;
                bool is_on_stroke = false;
                // -1 segment if its open since its not connecting to the start
                if (polyline.closed) {
                    num_segments = num_pts;
                } else {
                    num_segments = num_pts - 1;
                }
                // k = segment index; iter thru to determine how to draw strokes
                for (int k = polyline.pts_begin; k < polyline.pts_begin + num_segments; k++) {
                    // so it wraps prop.
                    next = polyline.pts_begin + ((k - polyline.pts_begin + 1) % num_pts);
                    start_pt = points[k];
                    end_pt = points[next];                
                    seg_vector = end_pt - start_pt;
                    start_to_px = pixel - start_pt;
                    // coord on segment axis; how much start_to_px vector points along segment w/o segment length scaling
                    float proj_frac = (dot(start_to_px, seg_vector) / dot(seg_vector, seg_vector));
                    
                    // flat cap checks
                    bool beyond_start = !polyline.closed && (k == polyline.pts_begin) && proj_frac < 0.0;
                    bool beyond_end = !polyline.closed && (k == polyline.pts_begin + num_segments - 1) && proj_frac > 1.0;

                    if (!(beyond_end || beyond_start)) {
                        // constrains to [0,1] so we get closest landing pt of projection
                        float proj_frac_clamped = clamp(proj_frac, 0.0, 1.0);
                        float2 closest_pt = start_pt + (proj_frac_clamped *seg_vector);
                        float dist_to_seg = length(pixel - closest_pt);
                        if (dist_to_seg < curr_shape.stroke_width / 2) {
                            is_on_stroke = true;
                        }
                    }
                }
                if (is_on_stroke) {
                    image[tid] = curr_shape.stroke_rgba;
                }
            }
        }
    }
}
