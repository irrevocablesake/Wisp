#include "ParticlesKernel.h"

void ParticlesKernel::initialize(CudaContext& _cudaContext) {
	cudaContext = &_cudaContext;
}

__global__ void updateKernel( ParticlesKernel::CudaParticle* particles, unsigned int count, float deltaTime){
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= count)
		return;

	//CHANGE LATER TO READ FROM BUFFER INTEROP
	constexpr float gravity = -9.81f;
	constexpr float radius = 0.02f;
	constexpr float restitution = 0.8f;

	constexpr float3 boundsMin = { -2.0f, -2.0f, -2.0f };
	constexpr float3 boundsMax = { 2.0f,  2.0f,  2.0f };

	ParticlesKernel::CudaParticle& particle = particles[index];
	particle.velocity.y += gravity * deltaTime;

	particle.position.x += particle.velocity.x * deltaTime;
	particle.position.y += particle.velocity.y * deltaTime;
	particle.position.z += particle.velocity.z * deltaTime;

    if (particle.position.x < boundsMin.x + radius) {
        particle.position.x = boundsMin.x + radius;
        particle.velocity.x = fabsf(particle.velocity.x) * restitution;
    }
    else if (particle.position.x > boundsMax.x - radius) {
        particle.position.x = boundsMax.x - radius;
        particle.velocity.x = -fabsf(particle.velocity.x) * restitution;
    }

    if (particle.position.y < boundsMin.y + radius) {
        particle.position.y = boundsMin.y + radius;
        particle.velocity.y = fabsf(particle.velocity.y) * restitution;
    }
    else if (particle.position.y > boundsMax.y - radius) {
        particle.position.y = boundsMax.y - radius;
        particle.velocity.y = -fabsf(particle.velocity.y) * restitution;
    }

    if (particle.position.z < boundsMin.z + radius) {
        particle.position.z = boundsMin.z + radius;
        particle.velocity.z = fabsf(particle.velocity.z) * restitution;
    }
    else if (particle.position.z > boundsMax.z - radius) {
        particle.position.z = boundsMax.z - radius;
        particle.velocity.z = -fabsf(particle.velocity.z) * restitution;
    }
}

void ParticlesKernel::launchParticleUpdate(void* buffer, unsigned int count, float deltaTime) {
	constexpr int blockSize = 256;
	unsigned int gridSize = (count + blockSize - 1) / blockSize;
	updateKernel << <gridSize, blockSize, 0, cudaContext->stream >> > ( static_cast<CudaParticle*>(buffer), count, deltaTime );
}

void ParticlesKernel::updateParticles(unsigned int count, float deltaTime, bool waitForVulkan) {
	if (waitForVulkan) {
		cudaExternalSemaphoreWaitParams waitParams{};
		cudaWaitExternalSemaphoresAsync(&vkToCuda, &waitParams, 1, cudaContext->stream);
	}

	launchParticleUpdate(cudaBuffer, count, deltaTime);

	cudaExternalSemaphoreSignalParams signalParams{};
	cudaSignalExternalSemaphoresAsync( &cudaToVk, &signalParams, 1, cudaContext->stream );
}