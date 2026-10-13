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
            float2 p0 = polyline[i]; // e.g. first point on polyline is point 0
            float2 p1 = polyline[(i + 1) % num_polyline]; // e.g. next point on polyline is point 1
            bool is_upward = p1.y > p0.y;
            
            // check if q.y lies between the endpoints' y-values(no need to check t separately)
            // check when ray passes exactly through a vertex shared by two edges, it only counts once(corner case)
            bool crosses = (p0.y <= q.y && p1.y > q.y) || (p1.y <= q.y && p0.y > q.y);
            if (crosses) {
                // ray-edge math from slides
                // ray point is q + (1,0)*s; every point on the line(edge) is p0 + t(p1-p0)
                float t = (q.y - p0.y) / (p1.y - p0.y); // how far along the edge we are when the edge reaches the same height as q
                float x_cross = p0.x + t * (p1.x - p0.x); // x component of the point where the ray crosses with the edge
                if (x_cross > q.x){ // if that crossing is to the right of q(where the ray starts), count it as a winding crossing
                    count += is_upward ? 1 : -1; // if edge crosses upward, +1
                }
            }
        }
        if (count != 0){
            image[tid] = float4(fill_color, 1.0);
        }
    }

    if (use_stroke_color) {
        int num_segments = is_closed ? num_polyline : num_polyline - 1; // closed polylines have one more edge
        for (int i = 0; i < num_segments; i++) {
            float2 p0 = polyline[i]; // start from polyline's first point e.g. point 0
            float2 p1 = polyline[(i + 1) % num_polyline]; // e.g. point 1 (next one)

            // detect whether curr segment is first or last segment of open polyline
            // if polyline is closed, there's no start/end, so both are false
            bool start = (!is_closed && i == 0);
            bool end   = (!is_closed && i == (num_segments - 1));

            float2 d = p1 - p0; // direction vector from p0 to p1
            float len = length(d); // segment length

            float l = dot(d, q - p0) / len; // projects point q onto the line through the segment
            // l says how far along the segment direction is the closest point to q
            
            // handles the caps of an open polyline
            // for first segment, requires l>=0 so stroke doesn't extend backward past the first endpoint
            // for last segment, requires l<= len so stroke doesn't extend forward past last endpoint
            if ((!start || l >= 0) && (!end || l <= len)) {
                float t = clamp(l, 0.0, len); // constrains projected distance to actual segment
                float2 q_prime = p0 + (d / len) * t; // closest point on the segment to q
                if (length(q - q_prime) <= stroke_width * 0.5) { // if distance from q to that closest point is less than half the stroke width, the pixel is inside the stroke -> color it
                    image[tid] = float4(stroke_color, 1.0);
                    break; // stops checking more segments once this pixel is already known to be part of the stroke
                }
            }
        }
    }
}
