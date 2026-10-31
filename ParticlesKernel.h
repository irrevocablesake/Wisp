#pragma once

#include <cuda_runtime.h>
#include "CudaContext.h"

class ParticlesKernel {
	public:
		CudaContext* cudaContext;
	public:
		cudaExternalSemaphore_t vkToCuda{};
		cudaExternalSemaphore_t cudaToVk{};

		cudaExternalMemory_t externalMemory{};
		void* cudaBuffer = nullptr;

		struct CudaParticle {
			float4 position;
			float4 velocity;
		};

		void initialize( CudaContext &_cudaContext );

		void updateParticles(unsigned int count, float deltaTime, bool waitForVulkan = true);
		void launchParticleUpdate(void* buffer, unsigned int count, float deltaTime );
};