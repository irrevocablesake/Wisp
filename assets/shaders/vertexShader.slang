struct Particle
{
    float4 position;
    float4 velocity;
    float density;
    float pressure;
    float padding[2];
    float4 pressureForces;
    float4 viscosictyForces;
};

struct VSInput
{
    float3 position : POSITION;
};

struct VSOutput
{
    float4 position : SV_Position;
    float density : TEXCOORD0;
};

struct UniformData {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
}

[[vk::binding(0, 0)]]
StructuredBuffer<Particle> particles;

[[vk::binding(0, 1)]]
ConstantBuffer<UniformData> uniformData;

[shader("vertex")]
VSOutput main(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    float4 localPosition = float4(input.position, 1.0);

    float4 worldPosition = mul(uniformData.model, localPosition);
    worldPosition = worldPosition + particles[ instanceID ].position;

    float4 viewPosition = mul(uniformData.view, worldPosition);
    float4 clipPosition = mul(uniformData.projection, viewPosition);

    output.position = clipPosition;
    output.density = particles[instanceID].density;

    return output;
}