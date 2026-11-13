struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput {
    float4 position : SV_Position;
};

struct UniformData {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4x4 inverseProjection;
    float2 screenSize;
};

[[vk::binding(0, 0)]] ConstantBuffer<UniformData> uniformData;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(uniformData.model, float4(input.position, 1.0));
    output.position = mul(uniformData.projection, mul(uniformData.view, worldPos));
    return output;
}