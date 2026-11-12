#pragma once

#include "CudaContext.cuh"
#include "ParticlesKernel.cuh"
#include "AdaptiveTimeStep.cuh"
#include "GPUSimulationParams.h"
#include "CPUSimulationParams.h"

class CudaInterface {
	public:
		CudaContext cudaContext;
		ParticlesKernel particlesKernel;
		TimeStep timeStep;
		
		using CudaBuffer = CudaContext::CudaBuffer;
		GPUSimulationParams simulationParams;

		cudaExternalSemaphore_t vkToCuda{};
		cudaExternalSemaphore_t cudaToVk{};

	public:
		void simulate();
		void initialize(const std::array< uint8_t, 16 >& deviceUUID, const CPUSimulationParams& _simulationParams );
		cudaExternalSemaphore_t importSemaphore(HANDLE vkToCudaHandle );
		void cudaWaitingForVulkan();
		void signalCudaFinished();
};