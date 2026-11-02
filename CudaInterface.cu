#include "CudaInterface.h"

void CudaInterface::initialize(const std::array< uint8_t, 16 >& deviceUUID  ) {
	cudaContext.initialize(deviceUUID);
	particlesKernel.initialize(cudaContext );
}

cudaExternalSemaphore_t CudaInterface::importSemaphore(HANDLE vkToCudaHandle) {
	return cudaContext.importSemaphores(vkToCudaHandle );
}