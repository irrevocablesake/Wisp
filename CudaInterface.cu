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
    cudaWaitingForVulkan();

    particlesKernel.generateIDs();
    particlesKernel.sortIDs();
    particlesKernel.launchReorder();

    particlesKernel.buildCellRanges();
    particlesKernel.calculateDensityAndPressure();
    particlesKernel.calculatePressureAndViscosityForces();
    particlesKernel.updateParticles(0.005f);

    signalCudaFinished();
}

void CudaInterface::cudaWaitingForVulkan() {
	cudaExternalSemaphoreWaitParams waitParams{};
	cudaWaitExternalSemaphoresAsync(&vkToCuda, &waitParams, 1, cudaContext.stream);
}

void CudaInterface::signalCudaFinished() {
	cudaExternalSemaphoreSignalParams signalParams{};
	cudaSignalExternalSemaphoresAsync(&cudaToVk, &signalParams, 1, cudaContext.stream);
}