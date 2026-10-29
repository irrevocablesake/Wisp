struct PSInput
{
    float4 position : SV_Position;
};

[shader("fragment")]
float4 main(PSInput input) : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}