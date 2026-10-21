[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // pixel center
    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // edges
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    // We can compute the 2D normals of the edges as nij = (eij,y, −eij,x)
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    // For a 2D point p, the edge equation is Eij (p) = (p − vi) · nij
    float E01 = dot(p-sp0, n01);
    float E12 = dot(p-sp1, n12);
    float E20 = dot(p-sp2, n20);

    bool inside = (E01 <= 0 && E12 <= 0 && E20 <= 0) || (E01 >= 0 && E12 >= 0 && E20 >= 0);
    if (!inside) {
        return;
    }

    // projected barycentric coordinates
    float area = cross2(sp1-sp0, sp2-sp0);
    float bp0 = cross2(sp1-p, sp2-p) / area;
    float bp1 = cross2(sp2-p, sp0-p) / area;
    float bp2 = cross2(sp0-p, sp1-p) / area;

    // Using the projected barycentric coordinates and the depth at the three vertices, we convert them
    // to the original barycentric coordinates using Eq. 15
    float num0 = bp0 * inv_depth0;
    float num1 = bp1 * inv_depth1;
    float num2 = bp2 * inv_depth2;
    float denom = num0 + num1 + num2;
    float b0 = num0 / denom;
    float b1 = num1 / denom;
    float b2 = num2 / denom;

    // obtain the Z values/depth using Eq. (16), compare the depth with the depth image at the
    // same pixel, and update the depth if the new depth is smaller
    float depth = (b0/inv_depth0) + (b1/inv_depth1) + (b2/inv_depth2);
    if(depth < depth_image[ipixel]){
        depth_image[ipixel] = depth;

        //update the color image using barycentric coordinates interpolated vertex color
        float3 c = b0*c0 + b1*c1 + b2*c2;
        color_image[ipixel] = float4(c, 1.0);
    }
}

float cross2(float2 u, float2 v) {
    return u.x * v.y - u.y * v.x;
}