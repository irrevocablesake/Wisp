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

static const int R = 5;
static const float sigmaSpatial = 2.5;
static const float sigmaDepth = 0.30;

static const float BG_THRESHHOLD = 0.999999;

float3 getViewSpacePosition(float2 uv) {
    float depth = depthTexture.Sample(uv);
    float2 ndc = uv - 1.0;

    float4 clipPosition = float4(ndc, depth, 1.0);

    float4 viewPosition = mul(uniformData.inverseProjection, clipPosition);

    return viewPosition.xyz / viewPosition.w;
}

float3 reconstructNormal(float2 uv)
{
    float2 texel = 1.0 / uniformData.screenSize;

    //Normal Basis
    float3 center = getFilteredViewPosition(uv);
    float3 right = getFilteredViewPosition(uv + float2(texel.x, 0.0));
    float3 up = getFilteredViewPosition(uv - float2(0.0, texel.y));

    float3 dx = right - center;
    float3 dy = up - center;

    float3 normal = normalize(cross(dy, dx));

    // Orient the normal towards the camera.
    if (dot(normal, normalize(-center)) < 0.0)
        normal = -normal;

    return normal;
}

float3 getFilteredViewPosition(float2 uv)
{
    float centerDepth = depthTexture.Sample(uv);

    // Background pixels have no valid surface position.
    if (centerDepth >= BG_THRESHHOLD)
        return float3(0.0, 0.0, 0.0);

    float3 center = getViewSpacePosition(uv);

    float2 texel = 1.0 / uniformData.screenSize;

    float3 accumulatedPosition = float3(0.0, 0.0, 0.0);
    float accumulatedWeight = 0.0;

    for (int y = -R; y <= R; ++y)
    {
        for (int x = -R; x <= R; ++x)
        {
            float2 sampleUV = uv + float2(x, y) * texel;

            // Reject samples outside the texture.
            if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
                sampleUV.y < 0.0 || sampleUV.y > 1.0)
                continue;

            float sampleDepth = depthTexture.Sample(sampleUV);

            // Don't include background pixels.
            if (sampleDepth >= BG_THRESHHOLD)
                continue;

            float3 samplePosition = getViewSpacePosition(sampleUV);

            float spatialDistanceSquared = float(x * x + y * y);

            float depthDifference = samplePosition.z - center.z;

            float spatialWeight = exp( -spatialDistanceSquared / (2.0 * sigmaSpatial * sigmaSpatial) );

            float depthWeight = exp( -(depthDifference * depthDifference) / (2.0 * sigmaDepth * sigmaDepth) );

            float weight = spatialWeight * depthWeight;

            accumulatedPosition += samplePosition * weight;
            accumulatedWeight += weight;
        }
    }

    if (accumulatedWeight > 1e-6)
        return accumulatedPosition / accumulatedWeight;

    return center;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float depth = depthTexture.Sample(input.uv);

    if (depth >= BG_THRESHHOLD)
        return float4(0.0, 0.0, 0.0, 1.0);

    float3 normal = reconstructNormal(input.uv);

    float3 visualization = normal * 0.5 + 0.5;

    return float4(visualization, 1.0);
}