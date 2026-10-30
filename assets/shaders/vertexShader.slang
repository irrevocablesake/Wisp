struct Particle
{
    float4 position;
    float4 velocity;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
};

struct VSOutput
{
    float4 position : SV_Position;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Particle> particles;

[shader("vertex")]
VSOutput main(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    float3 particlePosition = particles[instanceID].position.xyz;

    float sphereScale = 0.02;

    float3 worldPosition =
        input.position * sphereScale + particlePosition;

    output.position = float4(worldPosition, 1.0);

    return output;
}