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

    image[tid] = float4(1.0);
    // TODO: your code here
    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    float2 start_pt,end_pt,pixel;
    pixel = float2(tid);
    pixel.x += 0.5;
    // fix y-orientation
    pixel.y = height - 1 - pixel.y + 0.5;
    
    // handles fills
    if (is_closed && use_fill_color ) {
        int ctr = 0, next;
            for (int i = 0; i < num_polyline; i++) {
                next = (i + 1) % num_polyline;
                start_pt = polyline[i];
                end_pt = polyline[next];
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
            if (ctr != 0) {
                image[tid] = float4(fill_color.x, fill_color.y, fill_color.z, 1.0);
            }
    }
    // handles stroke lines
    if (use_stroke_color) {
        int next, num_segments = num_polyline;
        float2 seg_vector, start_to_px;
        bool is_on_stroke = false;
        if (!is_closed) num_segments = num_polyline - 1;
        for (int i = 0; i < num_segments; i++) {
                next = (i + 1) % num_polyline;
                start_pt = polyline[i];
                end_pt = polyline[next];
                seg_vector = end_pt - start_pt;
                start_to_px = pixel - start_pt;

                // coord on segment axis; how much start_to_px vector points along segment w/o segment length scaling
                float proj_frac = (dot(start_to_px, seg_vector) / dot(seg_vector, seg_vector));
                
                // flat cap checks
                bool beyond_start =  !is_closed && i == 0 && proj_frac < 0;
                bool beyond_end = !is_closed && i == num_segments - 1 && proj_frac > 1;

                if (!(beyond_end || beyond_start)) {
                    // constrains to [0,1] so we get closest landing pt of projection
                    float proj_frac_clamped = clamp(proj_frac, 0.0, 1.0);
                    float2 closest_pt = start_pt + (proj_frac_clamped *seg_vector);
                    float dist_to_seg = length(pixel - closest_pt);
                    if (dist_to_seg < stroke_width / 2) {
                        is_on_stroke = true;
                    }
                }
        }
        if (is_on_stroke) {
            image[tid] = float4(stroke_color.x, stroke_color.y, stroke_color.z, 1.0);
        }
    }
} 
