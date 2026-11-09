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
    if (t_roots.x > t_min && t_roots.x < t_max) {
        t_sphere = t_roots.x;
    }
    // compare against t_sphere for t_roots.x < t_roots_y
    if (t_roots.y > t_min && t_roots.y < t_max &&
        (t_sphere < 0.0 || t_roots.y < t_sphere)) {
        t_sphere = t_roots.y;
    }

    if (t_sphere > 0.0) {
        t = t_sphere;
        return true;
    }
    return false;
}

struct Hit {
    bool hit;
    float t;
    int shape;
    float3 n;
    uint tri;// global tri idx j of winner
    float3 b; // baryc. from intersect_tri
};

Hit closest_hit(float3 o, float3 d, float t_min, float t_max,
                StructuredBuffer<Shape> shapes, int num_shapes,
                StructuredBuffer<Sphere> spheres, 
                StructuredBuffer<TriangleMesh> meshes,
                StructuredBuffer<float3> vertices,
                StructuredBuffer<float3> vertex_normals,
                StructuredBuffer<uint3> triangles,
                StructuredBuffer<BVHNode> bvh_nodes) {
    Hit h; 
    h.hit = false; h.t = t_max; h.shape = -1; h.n = float3(0);
    h.tri = 0; h.b = float3(0);
    for (int i = 0; i < num_shapes; i++) {
        Shape sh = shapes[i];
        // pt affected by transl. for origin
        float3 o_obj = xform_pt(sh.world_to_obj, o); // pt w = 1
        // dir not affected by transl.
        float3 d_obj = xform_vec(sh.world_to_obj, d); // dir: w = 0 but not norm yet
        float t;
        if (sh.type == ShapeType.Sphere){
            if (intersect_sphere(o_obj, d_obj, spheres[sh.index], t_min, h.t, t)) {
                h.hit = true;
                // makes each ltr shape beat the curr best || t_max
                h.t = t;
                h.shape = i;
            }
        } else {
            // finds the mesh
            TriangleMesh mesh = meshes[sh.index];
            int node_idx = int(mesh.bvh_root_index);
            // esc idx used for skipping a subtree
            int end_idx = bvh_nodes[node_idx].next_index;
            while (node_idx < end_idx) {
                BVHNode node = bvh_nodes[node_idx];
                                                                                    // h.t as t_max ensures boxes further than curr best hit will count as a miss
                if (!intersect_aabb(node.bbox_min, node.bbox_max, o_obj, d_obj, t_min, h.t)) {
                    // miss so skip the subtree
                    node_idx = node.next_index; 
                } else if (node.is_leaf) {
                    // test only leaf's triangles
                    for (uint j = node.triangle_begin; j < node.triangle_end; j++) {
                        uint3 tri = triangles[j] + mesh.vertex_offset;
                        float3 p0 = vertices[tri.x]; 
                        float3 p1 = vertices[tri.y];
                        float3 p2 = vertices[tri.z];
                        float t; float3 b;
                        if (intersect_tri(p0, p1, p2, o_obj, d_obj, t_min, h.t, b, t)) {
                            h.hit = true;
                            h.t = t;
                            h.shape = i;
                            h.tri = j;
                            h.b = b;
                        }
                    }
                    // done with leaf --> skip curr node subtree and move to next node not in subtree
                    node_idx = node.next_index;
                } else {
                    // go to first child
                    node_idx = node_idx + 1;
                }
            }
        }
    }
    if (h.hit) {
        // obj-space hit pt and norm
        Shape sh = shapes[h.shape];
        float3 n_obj;
        if (sh.type == ShapeType.Sphere) {
            float3 o_obj = xform_pt(sh.world_to_obj, o);
            float3 d_obj = xform_vec(sh.world_to_obj, d); // dir: w = 0 but not norm yet
            float3 p_obj = o_obj + h.t * d_obj;
            n_obj = normalize(p_obj - spheres[sh.index].center);
        } else {
            TriangleMesh mesh = meshes[sh.index];
            uint3 tri = triangles[h.tri] + mesh.vertex_offset;
            // baryc. interp of tri vertex normals using the coords
            float3 n_interp = 
                h.b.x * vertex_normals[tri.x] +
                h.b.y * vertex_normals[tri.y] + 
                h.b.z * vertex_normals[tri.z];
            n_obj = normalize(n_interp);
        }
        // world normal = normalize(inverse transpose, w = 0)
        // norm because non-uniform scale changes len.
        h.n = normalize(xform_vec(transpose(sh.world_to_obj), n_obj));
    }
    return h;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // calc aspect
    float a = float(width) / float(height);
    float half_w = tan(radians(camera.fov_x) / 2.0);
    float half_h = half_w / a;
    // cam space dir toward px center
    float3 cam_d = float3(
        -half_w + 2.0 * half_w * (float(tid.x) + 0.5) / float(width),
        half_h - 2.0 * half_h * (float(tid.y) + 0.5) / float(height),
        -1.0);  
    // move ray to world space
    float3 o = xform_pt(camera.cam_to_world, float3(0.0));
    float3 d = xform_vec(camera.cam_to_world, cam_d);

    float3 color = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {
        Hit h = closest_hit(o, d, EPSILON, 1e30, shapes, num_shapes, spheres,
                            meshes, vertices, vertex_normals, triangles, bvh_nodes);
        if(!h.hit){
            color += throughput * background;
            break;
        }
        Shape shape = shapes[h.shape];
        // hit pt uses h.t
        float3 p = o + h.t * d;

        float3 contribution = float3(0.0, 0.0, 0.0);
        for (int k = 0; k < num_lights; k++) {
            float3 l = lights[k].position - p;
            float dist2 = dot(l, l);
            Hit sh = closest_hit(p, l, EPSILON, 1.0, shapes, num_shapes, spheres,
                                 meshes, vertices, vertex_normals, triangles, bvh_nodes);
            bool shadowed = sh.hit;
            if (!shadowed) {
                                        // h.n = world norm
                float cos_theta = max(dot(h.n, normalize(l)), 0.0);
                contribution += shape.Kd * lights[k].intensity / dist2 * cos_theta;
            }
        }

        color += throughput * contribution;
        throughput *= shape.Kr;
        
        // if every component of throughput(r,g,b) = 0 stop bounce loop
        if (all(throughput == 0.0)) break;
        o = p;
        d = reflect(d, h.n);
    }


    image[tid] = float4(color, 1.0);
}
