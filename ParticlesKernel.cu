#include "ParticlesKernel.h"

#include<cccl/cub/cub.cuh>

__global__ void initializeIndicesKernel( uint32_t* particleIndices, unsigned int count ){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    particleIndices[index] = index;
}

__global__ void updateKernel( ParticlesKernel::CudaParticle* particles, unsigned int count, float deltaTime){
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= count )
		return;

	//CHANGE LATER TO READ FROM BUFFER INTEROP
	constexpr float gravity = -9.81f;
	constexpr float radius = 0.02f;
	constexpr float restitution = 0.8f;

	constexpr float3 boundsMin = { -2.0f, -2.0f, -2.0f };
	constexpr float3 boundsMax = { 2.0f,  2.0f,  2.0f };

	ParticlesKernel::CudaParticle& particle = particles[index];
	particle.velocity.y += gravity * deltaTime;

	particle.position.x += particle.velocity.x * deltaTime;
	particle.position.y += particle.velocity.y * deltaTime;
	particle.position.z += particle.velocity.z * deltaTime;

    if (particle.position.x < boundsMin.x + radius) {
        particle.position.x = boundsMin.x + radius;
        particle.velocity.x = fabsf(particle.velocity.x) * restitution;
    }
    else if (particle.position.x > boundsMax.x - radius) {
        particle.position.x = boundsMax.x - radius;
        particle.velocity.x = -fabsf(particle.velocity.x) * restitution;
    }

    if (particle.position.y < boundsMin.y + radius) {
        particle.position.y = boundsMin.y + radius;
        particle.velocity.y = fabsf(particle.velocity.y) * restitution;
    }
    else if (particle.position.y > boundsMax.y - radius) {
        particle.position.y = boundsMax.y - radius;
        particle.velocity.y = -fabsf(particle.velocity.y) * restitution;
    }

    if (particle.position.z < boundsMin.z + radius) {
        particle.position.z = boundsMin.z + radius;
        particle.velocity.z = fabsf(particle.velocity.z) * restitution;
    }
    else if (particle.position.z > boundsMax.z - radius) {
        particle.position.z = boundsMax.z - radius;
        particle.velocity.z = -fabsf(particle.velocity.z) * restitution;
    }
}

__global__ void generateIDsKernel( const ParticlesKernel::CudaParticle* particles, unsigned int count, float3 gridOrigin, float3 gridDimensions, float cellSize,  uint32_t* particleCellIds ){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count )
        return;

    float3 position = make_float3( particles[index].position.x, particles[index].position.y, particles[index].position.z );

    int3 cell{};
    cell.x = static_cast<int>( floorf((position.x - gridOrigin.x) / cellSize));

    cell.y = static_cast<int>( floorf((position.y - gridOrigin.y) / cellSize));

    cell.z = static_cast<int>( floorf((position.z - gridOrigin.z) / cellSize));

    cell.x = max(0, min(cell.x, (int)gridDimensions.x - 1));
    cell.y = max(0, min(cell.y, (int)gridDimensions.y - 1));
    cell.z = max(0, min(cell.z, (int)gridDimensions.z - 1));

    particleCellIds[index] = cell.x + gridDimensions.x * ( cell.y + gridDimensions.y * cell.z);
}

__global__ void buildCellRangesKernel( const uint32_t* sortedCellIds, uint32_t* cellStart, uint32_t* cellEnd, uint32_t particleCount){
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    uint32_t currentCell = sortedCellIds[index];

    if (index == 0 || sortedCellIds[index - 1] != currentCell){
        cellStart[currentCell] = index;
    }

    if (index == particleCount - 1 || sortedCellIds[index + 1] != currentCell){
        cellEnd[currentCell] = index + 1;
    }
}

void ParticlesKernel::buildCellRanges()
{
    const uint32_t count = simulationParams.particlesCount;

    if (count == 0)
        return;

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    size_t totalCells = static_cast<size_t>(simulationParams.bounds.dimensions.x) * static_cast<size_t>(simulationParams.bounds.dimensions.y) * static_cast<size_t>(simulationParams.bounds.dimensions.z);
    size_t arraySize = totalCells * sizeof(uint32_t);

    cudaMemsetAsync(cellStart, 0xFF, arraySize, cudaContext->stream);
    cudaMemsetAsync(cellEnd, 0xFF, arraySize, cudaContext->stream);

    buildCellRangesKernel << <blocks, threadsPerBlock, 0, cudaContext->stream >> > (
        sortedCellIds,
        cellStart,
        cellEnd,
        count
    );
}

void ParticlesKernel::generateIDs() {
    constexpr unsigned int blockSize = 256;

    unsigned int gridSize = ( simulationParams.particlesCount + blockSize - 1) / blockSize;
    generateIDsKernel << <gridSize, blockSize, 0, cudaContext->stream >> > (
        static_cast<CudaParticle*>(cudaBuffer),
        simulationParams.particlesCount,
        simulationParams.bounds.min,
        simulationParams.bounds.dimensions,
        simulationParams.cellSize,
        particleCellIds
    );
}

void ParticlesKernel::initialize(CudaContext& _cudaContext) {
    cudaContext = &_cudaContext;

    const unsigned int count = static_cast<unsigned int>(simulationParams.particlesCount);
    const size_t bytes = count * sizeof(uint32_t);

    cudaMalloc(reinterpret_cast<void**>(&particleCellIds), bytes);
    cudaMalloc(reinterpret_cast<void**>(&sortedCellIds), bytes);
    cudaMalloc(reinterpret_cast<void**>(&particleIndices), bytes);
    cudaMalloc(reinterpret_cast<void**>(&sortedParticleIndices), bytes);

    constexpr unsigned int blockSize = 256;
    const unsigned int gridSize = (count + blockSize - 1) / blockSize;

    initializeIndicesKernel << <gridSize, blockSize, 0, cudaContext->stream >> > ( particleIndices, count );

    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sortTempStorageBytes,
        particleCellIds,
        sortedCellIds,
        particleIndices,
        sortedParticleIndices,
        count,
        0,
        sizeof(uint32_t) * 8,
        cudaContext->stream
    );

    cudaMalloc(&sortTempStorage, sortTempStorageBytes);

    size_t totalCells = static_cast<size_t>(simulationParams.bounds.dimensions.x) * static_cast<size_t>(simulationParams.bounds.dimensions.y) * static_cast<size_t>(simulationParams.bounds.dimensions.z);
    size_t rangeArraySize = totalCells * sizeof(uint32_t);

    cudaMalloc(&cellStart, rangeArraySize);
    cudaMalloc(&cellEnd, rangeArraySize);

    //An asynchronous NVIDIA CUDA Runtime API function that fills device memory with a given value on a specified stream without blocking the host CPU
    cudaMemsetAsync(cellStart, 0xFF, rangeArraySize, cudaContext->stream);
    cudaMemsetAsync(cellEnd, 0xFF, rangeArraySize, cudaContext->stream);
}

void ParticlesKernel::sortIDs() {
    const unsigned int count = static_cast<unsigned int>(simulationParams.particlesCount);

    cub::DeviceRadixSort::SortPairs(
        sortTempStorage,
        sortTempStorageBytes,
        particleCellIds,
        sortedCellIds,
        particleIndices,
        sortedParticleIndices,
        count,
        0,
        sizeof(uint32_t) * 8,
        cudaContext->stream
    );
}

void ParticlesKernel::launchParticleUpdate(void* buffer, float deltaTime) {
	constexpr int blockSize = 256;
	unsigned int gridSize = ( simulationParams.particlesCount + blockSize - 1) / blockSize;
	updateKernel << <gridSize, blockSize, 0, cudaContext->stream >> > ( static_cast<CudaParticle*>(buffer), simulationParams.particlesCount, deltaTime );
}

void ParticlesKernel::updateParticles( float deltaTime, bool waitForVulkan) {
	if (waitForVulkan) {
		cudaExternalSemaphoreWaitParams waitParams{};
		cudaWaitExternalSemaphoresAsync(&vkToCuda, &waitParams, 1, cudaContext->stream);
	}

	launchParticleUpdate(cudaBuffer, deltaTime);

	cudaExternalSemaphoreSignalParams signalParams{};
	cudaSignalExternalSemaphoresAsync( &cudaToVk, &signalParams, 1, cudaContext->stream );
}