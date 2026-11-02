#include "ParticlesKernel.h"

void ParticlesKernel::initialize(CudaContext& _cudaContext) {
	cudaContext = &_cudaContext;
}

__global__ void updateKernel( ParticlesKernel::CudaParticle* particles, unsigned int count, float deltaTime){
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= count)
		return;

	ParticlesKernel::CudaParticle& particle = particles[index];
	particle.velocity.y += -9.81f * deltaTime;

	particle.position.x += particle.velocity.x * deltaTime;
	particle.position.y += particle.velocity.y * deltaTime;
	particle.position.z += particle.velocity.z * deltaTime;
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