#include "AdaptiveTimeStep.cuh"

void TimeStep::initialize( const GPUSimulationParams &_simulationParams, cudaStream_t &_stream ) {
    simulationParams = &_simulationParams;
    stream = _stream;

    stepBuffer.MAX_BLOCKS = ( simulationParams -> particlesCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    cudaMalloc(&stepBuffer.partialStepBuffer, stepBuffer.MAX_BLOCKS * sizeof(TimeStep::PartialStep));
    cudaMalloc(&stepBuffer.stepBuffer, sizeof(TimeStep::Step));

    cudaMemset(stepBuffer.partialStepBuffer, 0, stepBuffer.MAX_BLOCKS * sizeof(TimeStep::PartialStep));
    cudaMemset(stepBuffer.stepBuffer, 0, sizeof(TimeStep::Step));

}

__global__ void accumulateFrameTimeKernel( TimeStep::Step* state, float frameDeltaTime) {
    if (frameDeltaTime && frameDeltaTime > 0.0f)
        state->accumulator += frameDeltaTime;
}

void TimeStep::addFrameTimeToAccumulator(float frameDeltaTime){
    accumulateFrameTimeKernel << <1, 1, 0, stream >> > (
        stepBuffer.stepBuffer,
        frameDeltaTime
    );
}

__global__ void calculateFinalAdaptiveTimeStepKernel(
    const TimeStep::PartialStep* partialResults,
    unsigned int blockCount,
    float smoothingRadius,
    float speedOfSound,
    float cflCoefficient,
    float accelerationCoefficient,
    float maxDeltaTime,

    TimeStep::Step* finalResult){

    __shared__ float sharedMaxVelocity[THREADS_PER_BLOCK];
    __shared__ float sharedMaxAcceleration[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;

    if (finalResult->accumulator <= 1e-6f) {
        if (tid == 0) {
            finalResult->deltaTime = 0.0f;
            finalResult->maxVelocity = 0.0f;
            finalResult->maxAcceleration = 0.0f;
            finalResult->accumulator = 0.0f;
        }

        return;
    }

    float localMaxVelocity = 0.0f;
    float localMaxAcceleration = 0.0f;

    for (unsigned int i = tid; i < blockCount; i += blockDim.x) {
        localMaxVelocity = fmaxf( localMaxVelocity, partialResults[i].maxVelocity );
        localMaxAcceleration = fmaxf( localMaxAcceleration, partialResults[i].maxAcceleration );
    }

    sharedMaxVelocity[tid] = localMaxVelocity;
    sharedMaxAcceleration[tid] = localMaxAcceleration;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride = stride / 2){
        if (tid < stride) {
            sharedMaxVelocity[tid] = fmaxf( sharedMaxVelocity[tid], sharedMaxVelocity[tid + stride] );
            sharedMaxAcceleration[tid] = fmaxf( sharedMaxAcceleration[tid], sharedMaxAcceleration[tid + stride] );
        }

        __syncthreads();
    }

    if (tid == 0) {
        const float maxVelocity = sharedMaxVelocity[0];
        const float maxAcceleration = sharedMaxAcceleration[0];

        const float cflDeltaTime = cflCoefficient * smoothingRadius / fmaxf(speedOfSound + maxVelocity, 1e-6f);
        float accelerationDeltaTime = CUDART_INF_F;

        if (maxAcceleration > 1e-6f){
            accelerationDeltaTime = accelerationCoefficient * sqrtf(smoothingRadius / maxAcceleration);
        }

        float deltaTime = fminf( cflDeltaTime, accelerationDeltaTime );
        deltaTime = fminf(deltaTime, maxDeltaTime);
        deltaTime = fminf(deltaTime, finalResult->accumulator);

        if (!isfinite(deltaTime) || deltaTime < 0.0f)
            deltaTime = 0.0f;

        finalResult->deltaTime = deltaTime;
        finalResult->maxVelocity = maxVelocity;
        finalResult->maxAcceleration = maxAcceleration;
    }
}

void TimeStep::calculateAdaptiveTimeStep(){
    const uint32_t blocks = (simulationParams->particlesCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    calculateFinalAdaptiveTimeStepKernel << < 1, 256, 0, stream >> > (
        stepBuffer.partialStepBuffer,
        blocks,
        simulationParams->smoothingRadius,
        simulationParams->speedOfSound,
        simulationParams->cflCoefficient,
        simulationParams->accelerationCoefficient,
        simulationParams->maxDeltaTime,
        stepBuffer.stepBuffer
    );
}