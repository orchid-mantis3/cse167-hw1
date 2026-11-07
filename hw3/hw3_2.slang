import hw3_common;

bool intersect_sphere(float3 o, float3 d, Sphere s, float t_min, float t_max, out float t) {
    t = t_max;
    float3 xc = s.center;
    float r = s.radius;

    float3 oc = o - xc;
    float A = dot(d, d);
    float B = 2.0 * dot(d, oc);
    float C = dot(oc, oc) - r * r;

    float2 t_roots;
    if (!solve_quadratic(A, B, C, t_roots)) {
        return false;
    }

    float t_sphere = -1.0;
    if (t_roots.x > t_min && (t_max < 0.0 || t_roots.x < t_max)) {
        t_sphere = t_roots.x;
    }
    if (t_roots.y > t_min && (t_max < 0.0 || t_roots.y < t_max) &&
        (t_sphere < 0.0 || t_roots.y < t_sphere)) {
        t_sphere = t_roots.y;
    }

    if (t_sphere > 0.0) {
        t = t_sphere;
        return true;
    }
    t = t_max;
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // sample pixel center, given image plane z = -1
    float a = float(width) / float(height); // aspect ratio
    float3 o = float3(0.0, 0.0, 0.0); // camera origin
    float3 d = float3(-a + 2.0 * a * (float(tid.x) + 0.5) / float(width),
                      1.0 - 2.0 * (float(tid.y) + 0.5) / float(height),
                      -1.0);

    // closest hit along the camera ray
    float t_closest = -1.0; // no sphere hit yet
    int hit_shape = -1;
    for (int i = 0; i < num_shapes; i++) {
        float t;
        if (intersect_sphere(o, d, spheres[shapes[i].index], EPSILON, t_closest, t)) {
            t_closest = t;
            hit_shape = i;
        }
    }

    float3 color = background;
    if (hit_shape >= 0) {
        Shape shape = shapes[hit_shape];
        float3 xc = spheres[shape.index].center;
        float3 p = o + t_closest * d;  // shaded point
        float3 n = normalize(p - xc);  // normal

        color = float3(0.0, 0.0, 0.0);
        for (int k = 0; k < num_lights; k++) {
            float3 l = lights[k].position - p;  // unnormalized, ends at the light
            float dist2 = dot(l, l);

            // shadow ray 
            bool shadowed = false;
            for (int j = 0; j < num_shapes; j++) {
                float t;
                if (intersect_sphere(p, l, spheres[shapes[j].index], EPSILON, 1.0, t)) {
                    shadowed = true;
                    break;
                }
            }
            if (shadowed) {
                continue;
            }

            float cos_theta = max(dot(n, normalize(l)), 0.0);
            color += shape.Kd * lights[k].intensity / dist2 * cos_theta;
        }
    }

    image[tid] = float4(color, 1.0);
}
