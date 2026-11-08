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
			float viscosityCoefficient;
			float particleRadius;
			float restitution;
			float smoothingRadius;

			uint32_t boundaryParticlesCount;
			float boundaryParticleMass;

			int particlesCount;
			float mass;
			float cellSize;

			float restDensity;
			float gasConstant;

			struct SimulationBounds {
				float3 min;
				float3 max;
				float3 dimensions;
			} bounds;

			uint3 gridDimensions;
		} simulationParams;

	public:
		void initialize(const std::array< uint8_t, 16 >& deviceUUID );

		cudaExternalSemaphore_t importSemaphore(HANDLE vkToCudaHandle );
};