struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]]
Sampler2D<float> depthTexture;

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float depth = depthTexture.Sample(input.uv);
    float visualization = 1.0 - depth;
    return float4(visualization * 20.0, visualization * 20.0, visualization * 20.0, 1.0 );
}