#include "CudaInterface.cuh"

void CudaInterface::initialize(const std::array< uint8_t, 16 >& deviceUUID, const CPUSimulationParams &_simulationParams  ) {
	cudaContext.initialize(deviceUUID);

	simulationParams = toCudaParams(_simulationParams);

	particlesKernel.initialize( cudaContext, simulationParams );
}

cudaExternalSemaphore_t CudaInterface::importSemaphore(HANDLE vkToCudaHandle) {
	return cudaContext.importSemaphores(vkToCudaHandle );
}
void CudaInterface::simulate(float vortexStrength ){
    cudaWaitingForVulkan();

    const float speedOfSound = sqrtf(simulationParams.gasConstant);
    float value = simulationParams.cflCoefficient * simulationParams.smoothingRadius / speedOfSound;
    const float dt = fminf( value, simulationParams.maxDeltaTime );

    constexpr int SUBSTEPS = 7;
    for (int s = 0; s < SUBSTEPS; ++s) {
        particlesKernel.generateIDs();
        particlesKernel.sortIDs();
        particlesKernel.launchReorder();
        particlesKernel.buildCellRanges();
        particlesKernel.calculateDensityAndPressure();
        particlesKernel.calculatePressureAndViscosityForces();
        particlesKernel.updateParticles(dt, vortexStrength );

        std::swap(
            particlesKernel.particleBufferA,
            particlesKernel.particleBufferB
        );

        particlesKernel.currentBufferIndex ^= 1;
    }

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