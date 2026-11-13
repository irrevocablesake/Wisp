struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct UniformData {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4x4 inverseProjection;
    float2 screenSize;
};

[[vk::binding(0, 0)]]
Sampler2D<float> depthTexture;

[[vk::binding(0, 1)]] 
ConstantBuffer<UniformData> uniformData;

float3 getViewSpacePosition(float2 uv) {
    float depth = depthTexture.Sample(uv);
    float2 ndc = uv - 1.0;

    float4 clipPosition = float4(ndc, depth, 1.0);

    float4 viewPosition = mul(uniformData.inverseProjection, clipPosition);

    return viewPosition.xyz / viewPosition.w;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float depth = depthTexture.Sample(input.uv);
    float visualization = 1.0 - depth;
    return float4(visualization * 20.0, visualization * 20.0, visualization * 20.0, 1.0 );
}