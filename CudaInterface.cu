#include "CudaInterface.h"

void CudaInterface::initialize(const std::array< uint8_t, 16 >& deviceUUID, const CPUSimulationParams &_simulationParams  ) {
	cudaContext.initialize(deviceUUID);

	simulationParams = toCudaParams(_simulationParams);

	particlesKernel.initialize( cudaContext, simulationParams );
}

cudaExternalSemaphore_t CudaInterface::importSemaphore(HANDLE vkToCudaHandle) {
	return cudaContext.importSemaphores(vkToCudaHandle );
}