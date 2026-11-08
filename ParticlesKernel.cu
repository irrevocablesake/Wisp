#include "ParticlesKernel.h"

#include<math_constants.h>
#include<cccl/cub/cub.cuh>

__global__ void initializeIndicesKernel( uint32_t* particleIndices, unsigned int count ){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    particleIndices[index] = index;
}

template < typename ParticleType >
__global__ void generateIDsKernel( const ParticleType* particles, unsigned int count, float3 gridOrigin, uint3 gridDimensions, float cellSize,  uint32_t* particleCellIds ){
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

void ParticlesKernel::buildCellRanges() {
    const uint32_t count = simulationParams->particlesCount;

    if (count == 0)
        return;

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    size_t totalCells = static_cast<size_t>(simulationParams->gridDimensions.x) * static_cast<size_t>(simulationParams->gridDimensions.y) * static_cast<size_t>(simulationParams->gridDimensions.z);

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

    unsigned int gridSize = (simulationParams->particlesCount + blockSize - 1) / blockSize;
    generateIDsKernel << <gridSize, blockSize, 0, cudaContext->stream >> > (
        static_cast<CudaParticle*>(cudaBuffer),
        simulationParams->particlesCount,
        simulationParams->bounds.min,
        simulationParams->gridDimensions,
        simulationParams->cellSize,
        particleCellIds
        );
}

__global__ void findNeighborsKernel(
    const ParticlesKernel::CudaParticle* particles,
    const uint32_t* sortedParticleIndices,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    uint32_t* neighborCounts,
    uint32_t particleCount,
    float3 gridOrigin,
    uint3 gridDimensions,
    float cellSize,
    float smoothingRadius)
{
    uint32_t particleID = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleID >= particleCount)
        return;

    float3 position = make_float3(particles[particleID].position.x, particles[particleID].position.y, particles[particleID].position.z);

    int3 cell = make_int3(
        floorf((position.x - gridOrigin.x) / cellSize),
        floorf((position.y - gridOrigin.y) / cellSize),
        floorf((position.z - gridOrigin.z) / cellSize)
    );

    uint32_t neighborCount = 0;
    float radiusSquared = smoothingRadius * smoothingRadius;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int cx = cell.x + x;
                int cy = cell.y + y;
                int cz = cell.z + z;

                if (cx < 0 || cy < 0 || cz < 0 ||
                    cx >= static_cast<int>(gridDimensions.x) ||
                    cy >= static_cast<int>(gridDimensions.y) ||
                    cz >= static_cast<int>(gridDimensions.z))
                    continue;

                uint32_t cellID = cx + gridDimensions.x * (cy + gridDimensions.y * cz);

                if (cellStart[cellID] == UINT_MAX)
                    continue;

                for (uint32_t i = cellStart[cellID]; i < cellEnd[cellID]; ++i) {
                    uint32_t otherID = sortedParticleIndices[i];

                    if (otherID == particleID)
                        continue;

                    float3 otherPosition = make_float3(
                        particles[otherID].position.x,
                        particles[otherID].position.y,
                        particles[otherID].position.z
                    );

                    float3 difference = make_float3(
                        position.x - otherPosition.x,
                        position.y - otherPosition.y,
                        position.z - otherPosition.z
                    );

                    float distanceSquared = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;

                    if (distanceSquared < radiusSquared) {
                        ++neighborCount;
                    }
                }
            }
        }
    }

    neighborCounts[particleID] = neighborCount;
}

void ParticlesKernel::findNeighbors()
{
    uint32_t count = simulationParams->particlesCount;

    if (count == 0)
        return;

    constexpr uint32_t threadsPerBlock = 256;

    uint32_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    findNeighborsKernel << <blocks, threadsPerBlock, 0, cudaContext->stream >> > (
        static_cast<CudaParticle*>(cudaBuffer),
        sortedParticleIndices,
        cellStart,
        cellEnd,
        neighborCounts,
        count,
        simulationParams->bounds.min,
        make_uint3(
            static_cast<uint32_t>(simulationParams->gridDimensions.x),
            static_cast<uint32_t>(simulationParams->gridDimensions.y),
            static_cast<uint32_t>(simulationParams->gridDimensions.z)
        ),
        simulationParams->cellSize,
        simulationParams->smoothingRadius
        );
}

void ParticlesKernel::sortIDs() {
    const unsigned int count = static_cast<unsigned int>(simulationParams->particlesCount);

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

__global__ void updateKernel(
    ParticlesKernel::CudaParticle* particles,
    unsigned int count,
    float deltaTime,
    float gravity,
    float particleMass,
    float3 boundsMin,
    float3 boundsMax,
    float radius,
    float restitution)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    ParticlesKernel::CudaParticle& particle = particles[index];

    float3 totalForce = make_float3(
        particle.pressureForces.x + particle.viscosityForces.x,
        particle.pressureForces.y + particle.viscosityForces.y,
        particle.pressureForces.z + particle.viscosityForces.z
    );

    //POSSIBLY SURFACE FRICTION?

    particle.velocity.x += (totalForce.x / particleMass) * deltaTime;
    particle.velocity.y += (totalForce.y / particleMass + gravity) * deltaTime;
    particle.velocity.z += (totalForce.z / particleMass) * deltaTime;

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
//NOTE: Later i think we either can remove this function or modify it to make it work with calculateDensityKernel
//Note: Function from the paper, just use as an abstraction - will understand later, basically says: How much influcence P2 will have on P1 based on r and h, returns a weight
__device__ float poly6Kernel(float distanceSquared, float smoothingRadius)
{
    float h2 = smoothingRadius * smoothingRadius;

    if (distanceSquared >= h2)
        return 0.0f;

    float h9 = powf(smoothingRadius, 9.0f);

    float coefficient = 315.0f / (64.0f * CUDART_PI_F * h9);

    float difference = h2 - distanceSquared;

    return coefficient * difference * difference * difference;
}

__device__ float calculateBoundaryDensityContribution(
    float3 position,
    const ParticlesKernel::BoundaryParticle* boundaryParticles,
    const uint32_t* sortedBoundaryParticleIndices,
    const uint32_t* boundaryCellStart,
    const uint32_t* boundaryCellEnd,
    float boundaryParticleMass,
    float smoothingRadius,
    float3 gridOrigin,
    uint3 gridDimensions,
    float cellSize)
{
    float density = 0.0f;

    const float h2 = smoothingRadius * smoothingRadius;
    const float h9 = powf(smoothingRadius, 9.0f);

    const float poly6Coefficient =
        315.0f / (64.0f * CUDART_PI_F * h9);

    int3 cell = make_int3(
        static_cast<int>(floorf((position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf((position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf((position.z - gridOrigin.z) / cellSize))
    );

    cell.x = max(0, min(cell.x, static_cast<int>(gridDimensions.x) - 1));
    cell.y = max(0, min(cell.y, static_cast<int>(gridDimensions.y) - 1));
    cell.z = max(0, min(cell.z, static_cast<int>(gridDimensions.z) - 1));

    const int searchRadius =
        static_cast<int>(ceilf(smoothingRadius / cellSize));

    for (int z = -searchRadius; z <= searchRadius; ++z)
    {
        for (int y = -searchRadius; y <= searchRadius; ++y)
        {
            for (int x = -searchRadius; x <= searchRadius; ++x)
            {
                int3 neighbourCell = make_int3(
                    cell.x + x,
                    cell.y + y,
                    cell.z + z
                );

                if (neighbourCell.x < 0 ||
                    neighbourCell.y < 0 ||
                    neighbourCell.z < 0 ||
                    neighbourCell.x >= static_cast<int>(gridDimensions.x) ||
                    neighbourCell.y >= static_cast<int>(gridDimensions.y) ||
                    neighbourCell.z >= static_cast<int>(gridDimensions.z))
                {
                    continue;
                }

                const uint32_t cellID =
                    neighbourCell.x +
                    gridDimensions.x *
                    (neighbourCell.y +
                        gridDimensions.y * neighbourCell.z);

                const uint32_t start = boundaryCellStart[cellID];

                if (start == UINT_MAX)
                    continue;

                const uint32_t end = boundaryCellEnd[cellID];

                for (uint32_t i = start; i < end; ++i)
                {
                    const uint32_t boundaryID =
                        sortedBoundaryParticleIndices[i];

                    const float3 boundaryPosition = make_float3(
                        boundaryParticles[boundaryID].position.x,
                        boundaryParticles[boundaryID].position.y,
                        boundaryParticles[boundaryID].position.z
                    );

                    const float dx = position.x - boundaryPosition.x;
                    const float dy = position.y - boundaryPosition.y;
                    const float dz = position.z - boundaryPosition.z;

                    const float distanceSquared =
                        dx * dx + dy * dy + dz * dz;

                    if (distanceSquared >= h2)
                        continue;

                    const float differenceH = h2 - distanceSquared;

                    const float weight =
                        poly6Coefficient *
                        differenceH * differenceH * differenceH;

                    density += boundaryParticleMass * weight;
                }
            }
        }
    }

    return density;
}

//NOTE: We can modify this function to use findNeighbours - just quite a repeattion of code, will fix later - bored for this now
__global__ void calculateDensityKernel(
    ParticlesKernel::CudaParticle* particles,

    const ParticlesKernel::BoundaryParticle* boundaryParticles,
    uint32_t boundaryParticlesCount,

    const uint32_t* sortedParticleIndices,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,

    const uint32_t* sortedBoundaryParticleIndices,
    const uint32_t* boundaryCellStart,
    const uint32_t* boundaryCellEnd,

    uint32_t particleCount,

    float particleMass,
    float boundaryParticleMass,

    float smoothingRadius,
    float restDensity,
    float gasConstant,

    float3 gridOrigin,
    uint3 gridDimensions,
    float cellSize,

    float3 boundaryGridOrigin,
    uint3 boundaryGridDimensions)
{
    uint32_t particleID = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleID >= particleCount)
        return;

    float3 position = make_float3(
        particles[particleID].position.x,
        particles[particleID].position.y,
        particles[particleID].position.z
    );

    float density = 0.0f;

    float smoothingRadiusSquared = smoothingRadius * smoothingRadius;
    float smoothingRadius9 = powf(smoothingRadius, 9.0f);
    float poly6Coefficient =
        315.0f / (64.0f * CUDART_PI_F * smoothingRadius9);

    int3 cell = make_int3(
        static_cast<int>(floorf((position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf((position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf((position.z - gridOrigin.z) / cellSize))
    );


    //NEDD TO FIX THIS -1 TO 1 RANGE
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighborCell = make_int3(
                    cell.x + x,
                    cell.y + y,
                    cell.z + z
                );

                if (neighborCell.x < 0 || neighborCell.x >= gridDimensions.x ||
                    neighborCell.y < 0 || neighborCell.y >= gridDimensions.y ||
                    neighborCell.z < 0 || neighborCell.z >= gridDimensions.z)
                {
                    continue;
                }

                uint32_t cellID = neighborCell.x + gridDimensions.x * (neighborCell.y + gridDimensions.y * neighborCell.z);

                if (cellStart[cellID] == UINT_MAX)
                    continue;

                for (uint32_t i = cellStart[cellID]; i < cellEnd[cellID]; ++i) {
                    uint32_t otherID = sortedParticleIndices[i];

                    float3 otherPosition = make_float3(
                        particles[otherID].position.x,
                        particles[otherID].position.y,
                        particles[otherID].position.z
                    );

                    float3 difference = make_float3(
                        position.x - otherPosition.x,
                        position.y - otherPosition.y,
                        position.z - otherPosition.z
                    );

                    float distanceSquared =
                        difference.x * difference.x +
                        difference.y * difference.y +
                        difference.z * difference.z;

                    if (distanceSquared < smoothingRadiusSquared) {
                        float differenceH = smoothingRadiusSquared - distanceSquared;

                        float weight = poly6Coefficient *
                            differenceH * differenceH * differenceH;

                        density += particleMass * weight;
                    }
                }
            }
        }
    }

    if (boundaryParticlesCount > 0)
    {
        density += calculateBoundaryDensityContribution(
            position,
            boundaryParticles,
            sortedBoundaryParticleIndices,
            boundaryCellStart,
            boundaryCellEnd,
            boundaryParticleMass,
            smoothingRadius,
            boundaryGridOrigin,
            boundaryGridDimensions,
            cellSize
        );
    }

    particles[particleID].density = density;
}

void ParticlesKernel::calculateDensity() {

    constexpr uint32_t BLOCK_SIZE = 256;

    const uint32_t count = simulationParams->particlesCount;

    if (count == 0)
        return;

    const uint32_t blocks =
        (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculateDensityKernel << <
        blocks,
        BLOCK_SIZE,
        0,
        cudaContext->stream
        >> > (
            static_cast<CudaParticle*>(cudaBuffer),

            static_cast<BoundaryParticle*>(BPcudaBuffer),
            simulationParams->boundaryParticlesCount,

            sortedParticleIndices,
            cellStart,
            cellEnd,

            sortedBoundaryParticleIndices,
            boundaryCellStart,
            boundaryCellEnd,

            count,

            simulationParams->mass,
            simulationParams->boundaryParticleMass,

            simulationParams->smoothingRadius,
            simulationParams->restDensity,
            simulationParams->gasConstant,

            simulationParams->bounds.min,
            simulationParams->gridDimensions,
            simulationParams->cellSize,

            boundaryGridOrigin,
            boundaryGridDimensions
            );
}

__global__ void calculatePressureKernel(
    ParticlesKernel::CudaParticle* particles,
    int particleCount,
    float restDensity,
    float gasConstant){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= particleCount)
        return;

    float density = particles[i].density;

    particles[i].pressure = gasConstant * fmaxf(density - restDensity, 0.0f);
}


void ParticlesKernel::initializeBoundaryGrid()
{
    const uint32_t count = simulationParams -> boundaryParticlesCount;

    if (count == 0)
        return;

    constexpr uint32_t BLOCK_SIZE = 256;
    constexpr float BOUNDARY_SPACING = 0.04f;

    // The outermost boundary layer extends two spacings beyond the room.
    // Add one extra spacing as padding for safe grid indexing.
    constexpr float PADDING = 3.0f * BOUNDARY_SPACING;

    float3 boundsMin = simulationParams -> bounds.min;
    float3 boundsMax = simulationParams -> bounds.max;

    boundaryGridOrigin = make_float3(
        boundsMin.x - PADDING,
        boundsMin.y - PADDING,
        boundsMin.z - PADDING
    );

    boundaryGridDimensions = make_uint3(
        static_cast<uint32_t>(ceilf(
            (boundsMax.x - boundsMin.x + 2.0f * PADDING) /
            simulationParams -> cellSize)) + 1,

        static_cast<uint32_t>(ceilf(
            (boundsMax.y - boundsMin.y + 2.0f * PADDING) /
            simulationParams -> cellSize)) + 1,

        static_cast<uint32_t>(ceilf(
            (boundsMax.z - boundsMin.z + 2.0f * PADDING) /
            simulationParams -> cellSize)) + 1
    );

    const size_t particleBytes = count * sizeof(uint32_t);

    const size_t totalCells =
        static_cast<size_t>(boundaryGridDimensions.x) *
        boundaryGridDimensions.y *
        boundaryGridDimensions.z;

    const size_t cellBytes = totalCells * sizeof(uint32_t);

    // Allocate boundary particle sorting arrays.
    cudaMalloc(&boundaryParticleCellIds, particleBytes);
    cudaMalloc(&sortedBoundaryCellIds, particleBytes);
    cudaMalloc(&boundaryParticleIndices, particleBytes);
    cudaMalloc(&sortedBoundaryParticleIndices, particleBytes);

    // Allocate boundary cell ranges.
    cudaMalloc(&boundaryCellStart, cellBytes);
    cudaMalloc(&boundaryCellEnd, cellBytes);

    // Initialize original boundary particle indices.
    const uint32_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initializeIndicesKernel << <blocks, BLOCK_SIZE, 0, cudaContext->stream >> > (
        boundaryParticleIndices,
        count
        );

    generateIDsKernel << <blocks, BLOCK_SIZE, 0, cudaContext->stream >> > (
        static_cast<const BoundaryParticle*>(BPcudaBuffer),
        count,
        boundaryGridOrigin,
        boundaryGridDimensions,
        simulationParams -> cellSize,
        boundaryParticleCellIds
        );

    // Determine CUB temporary storage requirements.
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        boundarySortTempStorageBytes,
        boundaryParticleCellIds,
        sortedBoundaryCellIds,
        boundaryParticleIndices,
        sortedBoundaryParticleIndices,
        count,
        0,
        sizeof(uint32_t) * 8,
        cudaContext->stream
    );

    cudaMalloc(
        &boundarySortTempStorage,
        boundarySortTempStorageBytes
    );

    // Sort boundary particles by cell ID.
    cub::DeviceRadixSort::SortPairs(
        boundarySortTempStorage,
        boundarySortTempStorageBytes,
        boundaryParticleCellIds,
        sortedBoundaryCellIds,
        boundaryParticleIndices,
        sortedBoundaryParticleIndices,
        count,
        0,
        sizeof(uint32_t) * 8,
        cudaContext->stream
    );

    // Initialize empty cell ranges.
    cudaMemsetAsync(
        boundaryCellStart,
        0xFF,
        cellBytes,
        cudaContext->stream
    );

    cudaMemsetAsync(
        boundaryCellEnd,
        0xFF,
        cellBytes,
        cudaContext->stream
    );

    // Build ranges for occupied cells.
    buildCellRangesKernel << <blocks, BLOCK_SIZE, 0, cudaContext->stream >> > (
        sortedBoundaryCellIds,
        boundaryCellStart,
        boundaryCellEnd,
        count
        );
}

void ParticlesKernel::calculatePressure(){
    constexpr uint32_t BLOCK_SIZE = 256;

    uint32_t gridSize = ( simulationParams -> particlesCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calculatePressureKernel << <
        gridSize,
        BLOCK_SIZE,
        0,
        cudaContext->stream
     >> > (
            static_cast<CudaParticle*>(cudaBuffer),
            simulationParams -> particlesCount,
            simulationParams -> restDensity,
            simulationParams -> gasConstant
         );
}

__device__ float3 spikyGradient( float3 displacement, float distance, float smoothingRadius )
{
    if (distance <= 0.0f || distance > smoothingRadius)
        return make_float3(0.0f, 0.0f, 0.0f);

    float h2 = smoothingRadius * smoothingRadius;
    float h3 = h2 * smoothingRadius;
    float h6 = h3 * h3;

    float coefficient = -45.0f / (CUDART_PI_F * h6);
    float difference = smoothingRadius - distance;
    float magnitude = coefficient * difference * difference;

    return make_float3(
        magnitude * displacement.x / distance,
        magnitude * displacement.y / distance,
        magnitude * displacement.z / distance
    );
}

__global__ void calculatePressureForcesKernel(
    ParticlesKernel::CudaParticle* particles,
    uint32_t particleCount,
    float particleMass,
    float smoothingRadius,

    float3 gridOrigin,
    float cellSize,
    uint3 gridDimensions,

    const uint32_t* sortedParticleIndices,
    const uint32_t* cellStart,
    const uint32_t* cellEnd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= particleCount)
        return;

    const ParticlesKernel::CudaParticle particle = particles[i];

    float3 position = make_float3(
        particle.position.x,
        particle.position.y,
        particle.position.z
    );

    float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);

    float smoothingRadiusSquared = smoothingRadius * smoothingRadius;

    int3 cell = make_int3(
        static_cast<int>(floorf(
            (position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf(
            (position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf(
            (position.z - gridOrigin.z) / cellSize))
    );

    int searchRadius = static_cast<int>(
        ceilf(smoothingRadius / cellSize)
    );

    float h2 = smoothingRadius * smoothingRadius;
    float h3 = h2 * smoothingRadius;
    float h6 = h3 * h3;

    float spikyCoefficient = -45.0f / (CUDART_PI_F * h6);

    for (int z = -searchRadius; z <= searchRadius; ++z)
    {
        for (int y = -searchRadius; y <= searchRadius; ++y)
        {
            for (int x = -searchRadius; x <= searchRadius; ++x)
            {
                int3 neighbourCell = make_int3(
                    cell.x + x,
                    cell.y + y,
                    cell.z + z
                );

                if (neighbourCell.x < 0 ||
                    neighbourCell.y < 0 ||
                    neighbourCell.z < 0 ||
                    neighbourCell.x >= static_cast<int>(gridDimensions.x) ||
                    neighbourCell.y >= static_cast<int>(gridDimensions.y) ||
                    neighbourCell.z >= static_cast<int>(gridDimensions.z)) {
                    continue;
                }

                uint32_t cellId =
                    neighbourCell.x +
                    gridDimensions.x *
                    (neighbourCell.y +
                        gridDimensions.y * neighbourCell.z);

                uint32_t start = cellStart[cellId];
                uint32_t end = cellEnd[cellId];

                if (start == UINT_MAX)
                    continue;

                for (uint32_t k = start; k < end; ++k)
                {
                    uint32_t j = sortedParticleIndices[k];

                    if (i == j)
                        continue;

                    const ParticlesKernel::CudaParticle neighbour = particles[j];

                    float3 neighbourPosition = make_float3(
                        neighbour.position.x,
                        neighbour.position.y,
                        neighbour.position.z
                    );

                    float3 displacement = make_float3(
                        position.x - neighbourPosition.x,
                        position.y - neighbourPosition.y,
                        position.z - neighbourPosition.z
                    );

                    float distanceSquared =
                        displacement.x * displacement.x +
                        displacement.y * displacement.y +
                        displacement.z * displacement.z;

                    if (distanceSquared <= 0.0f ||
                        distanceSquared >= smoothingRadiusSquared)
                    {
                        continue;
                    }

                    float distance = sqrtf(distanceSquared);

                    float difference = smoothingRadius - distance;

                    float gradientMagnitude = spikyCoefficient * difference * difference;

                    float3 gradient = make_float3(
                        gradientMagnitude * displacement.x / distance,
                        gradientMagnitude * displacement.y / distance,
                        gradientMagnitude * displacement.z / distance
                    );

                    float averagePressure = 0.5f * (particle.pressure + neighbour.pressure);

                    float neighbourDensity = fmaxf(neighbour.density, 1e-6f);

                    float scale = -particleMass * averagePressure / neighbourDensity;

                    pressureForce.x += scale * gradient.x;
                    pressureForce.y += scale * gradient.y;
                    pressureForce.z += scale * gradient.z;
                }
            }
        }
    }

    particles[i].pressureForces = make_float4(
        pressureForce.x,
        pressureForce.y,
        pressureForce.z,
        0.0f
    );
}

//Note: Look up the formula, there are multiple ways of doing this
//NOTE: Roughly similar to the function that's used for density computation for each particle!, so i guess we can change it later + both the helper functions! 
//Note: Better if we add MD format formula for both desngity and pressure force ocmputation
void ParticlesKernel::calculatePressureForces()
{
    constexpr uint32_t BLOCK_SIZE = 256;

    uint32_t gridSize = ( simulationParams -> particlesCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calculatePressureForcesKernel << <
        gridSize,
        BLOCK_SIZE,
        0,
        cudaContext->stream
        >> > (
            static_cast<CudaParticle*>(cudaBuffer),
            simulationParams -> particlesCount,
            simulationParams -> mass,
            simulationParams -> smoothingRadius,

            simulationParams -> bounds.min,
            simulationParams -> cellSize,
            simulationParams -> gridDimensions,

            sortedParticleIndices,
            cellStart,
            cellEnd
        );

}

//NOTE: The process of this is again same as the process for density, pressure forces, viscosity
//NOTE: Helper function, similar to others this is for reference, later we have to work on it - redundancy
__device__ float viscosityKernel( float distance, float smoothingRadius ) {
    if (distance < 0.0f || distance > smoothingRadius)
        return 0.0f;

    float h2 = smoothingRadius * smoothingRadius;
    float h3 = h2 * smoothingRadius;
    float h6 = h3 * h3;

    float coefficient = 45.0f / (CUDART_PI_F * h6);

    return coefficient * (smoothingRadius - distance);
}

__global__ void calculateViscosityForcesKernel(
    ParticlesKernel::CudaParticle* particles,
    uint32_t particleCount,
    float particleMass,
    float smoothingRadius,
    float viscosityCoefficient,

    float3 gridOrigin,
    float cellSize,
    uint3 gridDimensions,

    const uint32_t* sortedParticleIndices,
    const uint32_t* cellStart,
    const uint32_t* cellEnd)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= particleCount)
        return;

    const ParticlesKernel::CudaParticle particle = particles[i];

    float3 position = make_float3(
        particle.position.x,
        particle.position.y,
        particle.position.z
    );

    float3 velocity = make_float3(
        particle.velocity.x,
        particle.velocity.y,
        particle.velocity.z
    );

    float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);

    float smoothingRadiusSquared = smoothingRadius * smoothingRadius;

    int3 cell = make_int3(
        static_cast<int>(floorf(
            (position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf(
            (position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf(
            (position.z - gridOrigin.z) / cellSize))
    );

    int searchRadius = static_cast<int>(
        ceilf(smoothingRadius / cellSize)
        );

    float h2 = smoothingRadius * smoothingRadius;
    float h3 = h2 * smoothingRadius;
    float h6 = h3 * h3;

    float viscosityKernelCoefficient =
        45.0f / (CUDART_PI_F * h6);

    for (int z = -searchRadius; z <= searchRadius; ++z)
    {
        for (int y = -searchRadius; y <= searchRadius; ++y)
        {
            for (int x = -searchRadius; x <= searchRadius; ++x)
            {
                int3 neighbourCell = make_int3(
                    cell.x + x,
                    cell.y + y,
                    cell.z + z
                );

                if (neighbourCell.x < 0 ||
                    neighbourCell.y < 0 ||
                    neighbourCell.z < 0 ||
                    neighbourCell.x >= static_cast<int>(gridDimensions.x) ||
                    neighbourCell.y >= static_cast<int>(gridDimensions.y) ||
                    neighbourCell.z >= static_cast<int>(gridDimensions.z))
                {
                    continue;
                }

                uint32_t cellId =
                    neighbourCell.x +
                    gridDimensions.x *
                    (neighbourCell.y +
                        gridDimensions.y * neighbourCell.z);

                uint32_t start = cellStart[cellId];
                uint32_t end = cellEnd[cellId];

                if (start == UINT_MAX)
                    continue;

                for (uint32_t k = start; k < end; ++k)
                {
                    uint32_t j = sortedParticleIndices[k];

                    if (i == j)
                        continue;

                    const ParticlesKernel::CudaParticle neighbour =
                        particles[j];

                    float3 neighbourPosition = make_float3(
                        neighbour.position.x,
                        neighbour.position.y,
                        neighbour.position.z
                    );

                    float3 displacement = make_float3(
                        position.x - neighbourPosition.x,
                        position.y - neighbourPosition.y,
                        position.z - neighbourPosition.z
                    );

                    float distanceSquared =
                        displacement.x * displacement.x +
                        displacement.y * displacement.y +
                        displacement.z * displacement.z;

                    if (distanceSquared <= 0.0f ||
                        distanceSquared >= smoothingRadiusSquared)
                    {
                        continue;
                    }

                    float distance = sqrtf(distanceSquared);

                    float laplacian =
                        viscosityKernelCoefficient *
                        (smoothingRadius - distance);

                    float neighbourDensity =
                        fmaxf(neighbour.density, 1e-6f);

                    float3 relativeVelocity = make_float3(
                        neighbour.velocity.x - velocity.x,
                        neighbour.velocity.y - velocity.y,
                        neighbour.velocity.z - velocity.z
                    );

                    float scale =
                        viscosityCoefficient *
                        particleMass * particleMass *
                        laplacian / neighbourDensity;

                    viscosityForce.x += relativeVelocity.x * scale;
                    viscosityForce.y += relativeVelocity.y * scale;
                    viscosityForce.z += relativeVelocity.z * scale;
                }
            }
        }
    }

    particles[i].viscosityForces = make_float4(
        viscosityForce.x,
        viscosityForce.y,
        viscosityForce.z,
        0.0f
    );
}

void ParticlesKernel::calculateViscosityForces()
{
    constexpr uint32_t BLOCK_SIZE = 256;

    uint32_t gridSize = (simulationParams -> particlesCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calculateViscosityForcesKernel << <
        gridSize,
        BLOCK_SIZE,
        0,
        cudaContext->stream
        >> > (
            static_cast<CudaParticle*>(cudaBuffer),
            simulationParams -> particlesCount,
            simulationParams -> mass,
            simulationParams -> smoothingRadius,
            simulationParams -> viscosityCoefficient,

            simulationParams -> bounds.min,
            simulationParams -> cellSize,
            simulationParams -> gridDimensions,

            sortedParticleIndices,
            cellStart,
            cellEnd
            );

}


void ParticlesKernel::initialize(CudaContext& _cudaContext, GPUSimulationParams &_simulationParams ) {
    cudaContext = &_cudaContext;
    simulationParams = &_simulationParams;

    const unsigned int count = static_cast<unsigned int>(simulationParams -> particlesCount);
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

    size_t totalCells = static_cast<size_t>(simulationParams -> gridDimensions.x) * static_cast<size_t>(simulationParams -> gridDimensions.y) * static_cast<size_t>(simulationParams -> gridDimensions.z);
    size_t rangeArraySize = totalCells * sizeof(uint32_t);

    cudaMalloc(&cellStart, rangeArraySize);
    cudaMalloc(&cellEnd, rangeArraySize);

    //An asynchronous NVIDIA CUDA Runtime API function that fills device memory with a given value on a specified stream without blocking the host CPU
    cudaMemsetAsync(cellStart, 0xFF, rangeArraySize, cudaContext->stream);
    cudaMemsetAsync(cellEnd, 0xFF, rangeArraySize, cudaContext->stream);

    cudaMalloc( &neighborCounts, simulationParams -> particlesCount * sizeof(uint32_t) );

    initializeBoundaryGrid();
}


void ParticlesKernel::updateParticles( float deltaTime) {

    constexpr int blockSize = 256;
    unsigned int gridSize = (simulationParams -> particlesCount + blockSize - 1) / blockSize;
    updateKernel << <gridSize, blockSize, 0, cudaContext->stream >> > (
        static_cast<CudaParticle*>(cudaBuffer),
        simulationParams->particlesCount,
        deltaTime,
        simulationParams->gravity,
        simulationParams->mass, 
        simulationParams->bounds.min,
        simulationParams->bounds.max,
        simulationParams->particleRadius,
        simulationParams->restitution
        );
}

void ParticlesKernel::cudaWaitingForVulkan() {
    cudaExternalSemaphoreWaitParams waitParams{};
    cudaWaitExternalSemaphoresAsync(&vkToCuda, &waitParams, 1, cudaContext->stream);
}

void ParticlesKernel::signalCudaFinished() {
    cudaExternalSemaphoreSignalParams signalParams{};
    cudaSignalExternalSemaphoresAsync(&cudaToVk, &signalParams, 1, cudaContext->stream);
}