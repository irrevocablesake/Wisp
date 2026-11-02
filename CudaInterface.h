#pragma once

#include "CudaContext.h"
#include "ParticlesKernel.h"

class CudaInterface {
	public:
		CudaContext cudaContext;
		ParticlesKernel particlesKernel;
		
		using CudaBuffer = CudaContext::CudaBuffer;

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

	public:
		void initialize(const std::array< uint8_t, 16 >& deviceUUID );

		cudaExternalSemaphore_t importSemaphore(HANDLE vkToCudaHandle );
};