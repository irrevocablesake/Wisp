#include "CudaInterface.h"

void CudaInterface::initialize(const std::array< uint8_t, 16 >& deviceUUID, const CPUSimulationParams &_simulationParams  ) {
	cudaContext.initialize(deviceUUID);

	simulationParams = toCudaParams(_simulationParams);

	particlesKernel.initialize( cudaContext, simulationParams );
}

cudaExternalSemaphore_t CudaInterface::importSemaphore(HANDLE vkToCudaHandle) {
	return cudaContext.importSemaphores(vkToCudaHandle );
}

void CudaInterface::simulate() {
	particlesKernel.cudaWaitingForVulkan();

	particlesKernel.generateIDs();
	particlesKernel.sortIDs();
	particlesKernel.buildCellRanges();
	particlesKernel.calculateDensity();
	particlesKernel.calculatePressure();
	particlesKernel.calculatePressureForces();
	particlesKernel.calculateViscosityForces();
	particlesKernel.updateParticles(0.005);

	particlesKernel.signalCudaFinished();
}