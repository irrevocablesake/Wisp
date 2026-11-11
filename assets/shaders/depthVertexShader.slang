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

struct UniformData
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Particle> particles;

[[vk::binding(0, 1)]]
ConstantBuffer<UniformData> uniformData;

[shader("vertex")]
float4 main(VSInput input, uint instanceID : SV_InstanceID) : SV_Position
{
    float4 localPosition = float4(input.position, 1.0);

    float4 worldPosition = mul(uniformData.model, localPosition);
    worldPosition.xyz += particles[instanceID].position.xyz;

    float4 viewPosition = mul(uniformData.view, worldPosition);
    float4 clipPosition = mul(uniformData.projection, viewPosition);

    return clipPosition;
}