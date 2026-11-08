#pragma once

#include "CudaContext.h"
#include "ParticlesKernel.h"
#include "GPUSimulationParams.h"
#include "CPUSimulationParams.h"

class CudaInterface {
	public:
		CudaContext cudaContext;
		ParticlesKernel particlesKernel;
		
		using CudaBuffer = CudaContext::CudaBuffer;
		GPUSimulationParams simulationParams;

	public:
		void initialize(const std::array< uint8_t, 16 >& deviceUUID, const CPUSimulationParams& _simulationParams );

		cudaExternalSemaphore_t importSemaphore(HANDLE vkToCudaHandle );
};