#pragma once

#include "CudaContext.h"
#include "ParticlesKernel.h"

class CudaInterface {
	public:
		CudaContext cudaContext;
		ParticlesKernel particlesKernel;
		
		using CudaBuffer = CudaContext::CudaBuffer;
	public:
		void initialize(const std::array< uint8_t, 16 >& deviceUUID);

		cudaExternalSemaphore_t importSemaphore(HANDLE vkToCudaHandle );
};