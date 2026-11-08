#pragma once

#include <cuda_runtime.h>
#include "CudaContext.h"
#include "GPUSimulationParams.h"

class ParticlesKernel {
	public:
		CudaContext* cudaContext;
	public:
		cudaExternalSemaphore_t vkToCuda{};
		cudaExternalSemaphore_t cudaToVk{};

		cudaExternalMemory_t externalMemory{};
		void* cudaBuffer = nullptr;

		cudaExternalMemory_t BPexternalMemory{};
		void* BPcudaBuffer = nullptr;

		uint32_t* particleCellIds;
		uint32_t* sortedCellIds = nullptr;
		uint32_t* particleIndices = nullptr;
		uint32_t* sortedParticleIndices = nullptr;

		void* sortTempStorage = nullptr;
		size_t sortTempStorageBytes = 0;

		uint32_t* cellStart = nullptr;
		uint32_t* cellEnd = nullptr;

		uint32_t* neighborCounts = nullptr;

		void calculateDensity();

		void findNeighbors();

		struct CudaParticle {
			float4 position;
			float4 velocity;
			float density;
			float pressure;
			float padding[2]{};
			float4 pressureForces;
			float4 viscosityForces;
		};

		struct BoundaryParticle {
			float4 position{};
		};

		const GPUSimulationParams *simulationParams = nullptr;

		uint32_t* boundaryParticleCellIds = nullptr;
		uint32_t* sortedBoundaryCellIds = nullptr;
		uint32_t* boundaryParticleIndices = nullptr;
		uint32_t* sortedBoundaryParticleIndices = nullptr;

		uint32_t* boundaryCellStart = nullptr;
		uint32_t* boundaryCellEnd = nullptr;

		void* boundarySortTempStorage = nullptr;
		size_t boundarySortTempStorageBytes = 0;

		float3 boundaryGridOrigin{};
		uint3 boundaryGridDimensions{};
		void initializeBoundaryGrid();

		void initialize( CudaContext& _cudaContext, GPUSimulationParams& _simulationParams );

		void updateParticles( float deltaTime );
		void generateIDs();
		void sortIDs();
		void buildCellRanges();
		void calculatePressure();
		void calculatePressureForces();
		void calculateViscosityForces();

		void cudaWaitingForVulkan();
		void signalCudaFinished();

};