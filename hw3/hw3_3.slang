import hw3_common;

bool intersect_sphere(float3 o, float3 d, Sphere s, float t_min, float t_max, out float t) {
    t = t_max;
    float3 oc = o - s.center;
    float A = dot(d, d);
    float B = 2.0 * dot(d, oc);
    float C = dot(oc, oc) - s.radius * s.radius;

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
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float a = float(width) / float(height); // aspect ratio
    float3 o = float3(0.0, 0.0, 0.0); // camera origin
    float3 d = float3(-a + 2.0 * a * (float(tid.x) + 0.5) / float(width),
                      1.0 - 2.0 * (float(tid.y) + 0.5) / float(height),
                      -1.0);

    float3 color = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {
        float t_closest = -1.0;
        int hit_shape = -1;
        for (int i = 0; i < num_shapes; i++) {
            float t;
            if (intersect_sphere(o, d, spheres[shapes[i].index], EPSILON, t_closest, t)) {
                t_closest = t;
                hit_shape = i;
            }
        }

        if (hit_shape < 0) {
            color += throughput * background;
            break;
        }

        Shape shape = shapes[hit_shape];
        float3 xc = spheres[shape.index].center;
        float3 p = o + t_closest * d;
        float3 n = normalize(p - xc);

        float3 contribution = float3(0.0, 0.0, 0.0);
        for (int k = 0; k < num_lights; k++) {
            float3 l = lights[k].position - p;
            float dist2 = dot(l, l);

            bool shadowed = false;
            for (int j = 0; j < num_shapes; j++) {
                float t;
                if (intersect_sphere(p, l, spheres[shapes[j].index], EPSILON, 1.0, t)) {
                    shadowed = true;
                    break;
                }
            }
            if (!shadowed) {
                float cos_theta = max(dot(n, normalize(l)), 0.0);
                contribution += shape.Kd * lights[k].intensity / dist2 * cos_theta;
            }
        }

        color += throughput * contribution;
        throughput *= shape.Kr;
        o = p;
        d = reflect(d, n);
    }

    image[tid] = float4(color, 1.0);
}