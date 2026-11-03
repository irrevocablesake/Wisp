struct PSInput
{
    float4 position : SV_Position;
    float density : TEXCOORD0;
};

[shader("fragment")]
float4 main(PSInput input) : SV_Target
{
    float normalizedDensity = saturate(input.density / 12000.0f );

    float3 color = lerp(
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        normalizedDensity
    );

    return float4( color, 1.0);
}