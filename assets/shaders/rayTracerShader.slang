struct UniformData {
    float3 cameraPosition;
    float pad0;

    float3 viewportU;
    float pad1;

    float3 viewportV;
    float pad2;

    float3 pixelDeltaU;
    float pad3;

    float3 pixelDeltaV;
    float pad4;

    float3 viewportUPL;
    float pad5;

    float3 pixel00Location;
    float pad6;

	float imageWidth;
	float imageHeight;

	float aspectRatio;

	float viewportWidth;
	float viewportHeight;

	float focalLength;
};

[[vk::binding(0, 0)]]
RWTexture2D<float4> outputImage;

[[vk::binding(0, 1)]]
ConstantBuffer<UniformData> uniformBuffer;

struct Ray
{
    float3 origin;
    float3 direction;

    __init(float3 org, float3 dir) {
        origin = org;
        direction = dir;
    }

    float3 at(float t)
    {
        return origin + t * direction;
    }
};

bool hitSphere(const float3 center, float radius, Ray ray) {
    float3 oc = center - ray.origin;
    float a = dot(ray.direction, ray.direction);
    float b = -2.0 * dot(ray.direction, oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    return (discriminant >= 0);
}

float3 computeColor(Ray ray) {
    if (hitSphere(float3(0, 0, -1), 0.5, ray)) return float3(1.0, 0, 0);

    float3 unitDirection = normalize(ray.direction);
    float a = 0.5 * ( unitDirection.y + 1.0 );

    return (1.0 - a) * float3(1.0, 1.0, 1.0) + a * float3(0.5, 0.7, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    float3 pixelCenter = uniformBuffer.pixel00Location + (id.x * uniformBuffer.pixelDeltaU) + (id.y * uniformBuffer.pixelDeltaV);
    float3 rayDirection = pixelCenter - uniformBuffer.cameraPosition;

    Ray ray = Ray( uniformBuffer.cameraPosition, rayDirection);
    float3 pixelColor = computeColor(ray);

    outputImage[id.xy] = float4( pixelColor.z, pixelColor.y, pixelColor.x, 1.0);
}