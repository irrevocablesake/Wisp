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

		uint32_t* particleCellIds;
		uint32_t* sortedCellIds = nullptr;
		uint32_t* particleIndices = nullptr;
		uint32_t* sortedParticleIndices = nullptr;

		void* sortTempStorage = nullptr;
		size_t sortTempStorageBytes = 0;

		struct CudaParticle {
			float4 position;
			float4 velocity;
		};

		struct SimulationParams {
			float gravity;

			float particleRadius;
			float restitution;

			int particlesCount;
			float cellSize;

			struct SimulationBounds {
				float3 min;
				float3 max;
				float3 dimensions;
			} bounds;
		} simulationParams;

		void initialize( CudaContext &_cudaContext );

		void updateParticles(  float deltaTime, bool waitForVulkan = true);
		void generateIDs();
		void sortIDs();

		void launchParticleUpdate(void* buffer, float deltaTime );
};