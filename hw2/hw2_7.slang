StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;
uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;
Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.7 TODO: implement

    float3 p = vertices[vid];
    float3 n = vertex_normals[vid];

    //clip space position
    output.position = mul(MVP, float4(p, 1.0));
    //world space position
    output.world_position = mul(model_matrix, float4(p, 1.0)).xyz;
    output.world_normal = mul(normal_matrix, n);
    output.uv = vertex_uvs[vid];
    
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.7 TODO: implement

    float3 n = normalize(input.world_normal); //unit surface normal
    float3 l = normalize(light_dir); //unit vector pointing towards light
    float3 v = normalize(camera_position - input.world_position); //unit vector towards camera
    float3 E = light_color;

    //diffuse contribution (Eq 31)
    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;
    float n_dot_l = dot(n, l);

    //ambient
    float3 Ca = 0.2 * Kd;
    //diffuse
    float3 Cd = Kd * E * max(0.0, n_dot_l);
    //specular 
    float3 Cs = float3(0.0);
    if(n_dot_l>0.0){
        float3 r = reflect(-l, n);
        Cs = Ks * E * pow(max(0.0, dot(r, v)), exponent);
    }

    //final model
    float3 C = Ca + Cd + Cs;

    return float4(C, 1.0);
}