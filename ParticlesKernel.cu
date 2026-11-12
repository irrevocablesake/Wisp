#include "ParticlesKernel.cuh"

#include<math_constants.h>
#include<cccl/cub/cub.cuh>

#define MAX_SEARCH_RADIUS 1

__device__ float calculateBoundaryDensityContribution(
    float3 position,
    const ParticlesKernel::BoundaryParticle* boundaryParticles,
    const uint32_t* __restrict__ sortedBoundaryParticleIndices,
    const uint32_t* __restrict__ boundaryCellStart,
    const uint32_t* __restrict__ boundaryCellEnd,
    float boundaryParticleMass,
    float smoothingRadius,
    float3 gridOrigin,
    uint3 gridDimensions,
    float cellSize,
    float poly6Coefficient
    )
{
    float density = 0.0f;

    const float smoothingRadiusSquared = smoothingRadius * smoothingRadius;

    int3 cell = make_int3(
        static_cast<int>(floorf((position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf((position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf((position.z - gridOrigin.z) / cellSize))
    );

    cell.x = max(0, min(cell.x, static_cast<int>(gridDimensions.x) - 1));
    cell.y = max(0, min(cell.y, static_cast<int>(gridDimensions.y) - 1));
    cell.z = max(0, min(cell.z, static_cast<int>(gridDimensions.z) - 1));

    #pragma unroll
    for (int z = -MAX_SEARCH_RADIUS; z <= MAX_SEARCH_RADIUS; ++z)
    {
        #pragma unroll
        for (int y = -MAX_SEARCH_RADIUS; y <= MAX_SEARCH_RADIUS; ++y)
        {
            #pragma unroll
            for (int x = -MAX_SEARCH_RADIUS; x <= MAX_SEARCH_RADIUS; ++x)
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

                const uint32_t start = __ldg(&boundaryCellStart[cellID]);

                if (start == UINT_MAX)
                    continue;

                const uint32_t end = __ldg(&boundaryCellEnd[cellID]);

                for (uint32_t i = start; i < end; ++i)
                {
                    const uint32_t boundaryID = __ldg(&sortedBoundaryParticleIndices[i]);
                    const ParticlesKernel::BoundaryParticle* bPtr = &boundaryParticles[boundaryID];

                    const float3 boundaryPosition = make_float3(
                        __ldg(&bPtr->position.x),
                        __ldg(&bPtr->position.y),
                        __ldg(&bPtr->position.z)
                    );

                    const float dx = position.x - boundaryPosition.x;
                    const float dy = position.y - boundaryPosition.y;
                    const float dz = position.z - boundaryPosition.z;

                    const float distanceSquared =
                        dx * dx + dy * dy + dz * dz;

                    if (distanceSquared >= smoothingRadiusSquared)
                        continue;

                    const float differenceH = smoothingRadiusSquared -distanceSquared;

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

//NOTE: We can modify this function to use findNeighbours - just quite a repeattion of code, will fix later - bored for this now
__global__ void __launch_bounds__(512, 2) calculateDensityAndPressureKernel(
    ParticlesKernel::CudaParticle* particles,

    const ParticlesKernel::BoundaryParticle* boundaryParticles,
    uint32_t boundaryParticlesCount,

    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,

    const uint32_t* __restrict__ sortedBoundaryParticleIndices,
    const uint32_t* __restrict__ boundaryCellStart,
    const uint32_t* __restrict__ boundaryCellEnd,

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
    uint3 boundaryGridDimensions,
    float poly6Coefficient
    )
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

    int3 cell = make_int3(
        static_cast<int>(floorf((position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf((position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf((position.z - gridOrigin.z) / cellSize))
    );

    #pragma unroll
    for (int z = -MAX_SEARCH_RADIUS; z <= MAX_SEARCH_RADIUS; ++z)
    {
        #pragma unroll
        for (int y = -MAX_SEARCH_RADIUS; y <= MAX_SEARCH_RADIUS; ++y)
        {
            #pragma unroll
            for (int x = -MAX_SEARCH_RADIUS; x <= MAX_SEARCH_RADIUS; ++x)
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

                uint32_t start = __ldg(&cellStart[cellID]);
                uint32_t end = __ldg(&cellEnd[cellID]);

                if (start == UINT_MAX)
                    continue;

                for (uint32_t i = start; i < end; ++i) {
                    const ParticlesKernel::CudaParticle* otherPtr = &particles[i];

                    float3 otherPosition = make_float3(
                        __ldg(&otherPtr->position.x),
                        __ldg(&otherPtr->position.y),
                        __ldg(&otherPtr->position.z)
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
            cellSize,
            poly6Coefficient
        );
    }

    particles[particleID].density = density;
    particles[particleID].pressure = gasConstant * fmaxf(density - restDensity, 0.0f);

}

__global__ void __launch_bounds__(512, 2) calculatePressureAndViscosityForcesKernel(
    ParticlesKernel::CudaParticle* particles,
    uint32_t particleCount,
    float particleMass,
    float smoothingRadius,
    float viscosityCoefficient,
    float viscosityKernelCoefficient,
    float spikyCoefficient,

    float3 gridOrigin,
    float cellSize,
    uint3 gridDimensions,

    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd
    )
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

    float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
    float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);

    float smoothingRadiusSquared = smoothingRadius * smoothingRadius;

    int3 cell = make_int3(
        static_cast<int>(floorf((position.x - gridOrigin.x) / cellSize)),
        static_cast<int>(floorf((position.y - gridOrigin.y) / cellSize)),
        static_cast<int>(floorf((position.z - gridOrigin.z) / cellSize))
    );

    #pragma unroll
    for (int z = -MAX_SEARCH_RADIUS; z <= MAX_SEARCH_RADIUS; ++z)
    {
        #pragma unroll
        for (int y = -MAX_SEARCH_RADIUS; y <= MAX_SEARCH_RADIUS; ++y)
        {
            #pragma unroll
            for (int x = -MAX_SEARCH_RADIUS; x <= MAX_SEARCH_RADIUS; ++x)
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

                uint32_t cellId = neighbourCell.x + gridDimensions.x * (neighbourCell.y + gridDimensions.y * neighbourCell.z);

                uint32_t start = __ldg(&cellStart[cellId]);
                uint32_t end = __ldg(&cellEnd[cellId]);

                if (start == UINT_MAX)
                    continue;

                for (uint32_t k = start; k < end; ++k) {
                    if (i == k) continue;

                    const ParticlesKernel::CudaParticle* nPtr = &particles[k];
                    float3 nPos = make_float3(__ldg(&nPtr->position.x), __ldg(&nPtr->position.y), __ldg(&nPtr->position.z));
                    float3 nVel = make_float3(__ldg(&nPtr->velocity.x), __ldg(&nPtr->velocity.y), __ldg(&nPtr->velocity.z));
                    float nDensity = __ldg(&nPtr->density);
                    float nPress = __ldg(&nPtr->pressure);

                    float3 displacement = make_float3(
                        position.x - nPos.x,
                        position.y - nPos.y,
                        position.z - nPos.z
                    );

                    float distanceSquared = 
                        displacement.x * displacement.x +
                        displacement.y * displacement.y +
                        displacement.z * displacement.z;

                    if (distanceSquared > 0.0f && distanceSquared < smoothingRadiusSquared) {
                        float distance = sqrtf(distanceSquared);
                        float difference = smoothingRadius - distance;

                        float neighbourDensity = fmaxf(nDensity, 1e-6f);

                        // Pressure force
                        float gradientMagnitude = spikyCoefficient * difference * difference;

                        float3 gradient = make_float3(
                            gradientMagnitude * displacement.x / distance,
                            gradientMagnitude * displacement.y / distance,
                            gradientMagnitude * displacement.z / distance
                        );

                        float averagePressure = 0.5f * (particle.pressure + nPress);

                        float pressureScale = -particleMass * averagePressure / neighbourDensity;

                        pressureForce.x += pressureScale * gradient.x;
                        pressureForce.y += pressureScale * gradient.y;
                        pressureForce.z += pressureScale * gradient.z;

                        // Viscosity force
                        float laplacian = viscosityKernelCoefficient * difference;

                        float3 relativeVelocity = make_float3(
                            nVel.x - velocity.x,
                            nVel.y - velocity.y,
                            nVel.z - velocity.z
                        );

                        float viscosityScale = viscosityCoefficient * particleMass * particleMass * laplacian / neighbourDensity;

                        viscosityForce.x += relativeVelocity.x * viscosityScale;
                        viscosityForce.y += relativeVelocity.y * viscosityScale;
                        viscosityForce.z += relativeVelocity.z * viscosityScale;
                    }
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

    particles[i].viscosityForces = make_float4(
        viscosityForce.x,
        viscosityForce.y,
        viscosityForce.z,
        0.0f
    );
}

//Note: Look up the formula, there are multiple ways of doing this
//NOTE: Roughly similar to the function that's used for density computation for each particle!, so i guess we can change it later + both the helper functions! 
//Note: Better if we add MD format formula for both desngity and pressure force ocmputation


//NOTE: The process of this is again same as the process for density, pressure forces, viscosity
//NOTE: Helper function, similar to others this is for reference, later we have to work on it - redundancy




void ParticlesKernel::initializeBoundaryGrid()
{
    const size_t particleBytes = simulationParams->boundaryParticlesCount * sizeof(uint32_t);

    const uint3& boundaryGridDimensions = simulationParams->boundaryGridDimensions;
    const float3& boundaryGridOrigin = simulationParams->boundaryGridOrigin;

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


    initializeIndicesKernel << < BOUNDARY_BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > (
        boundaryParticleIndices,
        simulationParams->boundaryParticlesCount
        );

    generateIDsKernel << <BOUNDARY_BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > (
        static_cast<const BoundaryParticle*>(BPcudaBuffer),
        simulationParams->boundaryParticlesCount,
        boundaryGridOrigin,
        boundaryGridDimensions,
        simulationParams->cellSize,
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
        simulationParams->boundaryParticlesCount,
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
        simulationParams->boundaryParticlesCount,
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
    buildCellRangesKernel << <BOUNDARY_BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > (
        sortedBoundaryCellIds,
        boundaryCellStart,
        boundaryCellEnd,
        simulationParams->boundaryParticlesCount
        );
}


void ParticlesKernel::buildCellRanges() {
    size_t totalCells = static_cast<size_t>(simulationParams->gridDimensions.x) * static_cast<size_t>(simulationParams->gridDimensions.y) * static_cast<size_t>(simulationParams->gridDimensions.z);

    size_t arraySize = totalCells * sizeof(uint32_t);

    cudaMemsetAsync(cellStart, 0xFF, arraySize, cudaContext->stream);
    cudaMemsetAsync(cellEnd, 0xFF, arraySize, cudaContext->stream);

    buildCellRangesKernel << <BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > (
        sortedCellIds,
        cellStart,
        cellEnd,
        simulationParams->particlesCount
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

void ParticlesKernel::generateIDs() {
    generateIDsKernel << <BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > (
        particleBufferA,
        simulationParams->particlesCount,
        simulationParams->bounds.min,
        simulationParams->gridDimensions,
        simulationParams->cellSize,
        particleCellIds
    );
}

void ParticlesKernel::calculateDensityAndPressure() {

    const uint32_t count = simulationParams->particlesCount;

    if (count == 0)
        return;

    calculateDensityAndPressureKernel << <
        BLOCKS_COUNT,
        THREADS_PER_BLOCK,
        0,
        cudaContext->stream
        >> > (
            particleBufferB,

            static_cast<BoundaryParticle*>(BPcudaBuffer),
            simulationParams->boundaryParticlesCount,

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

            simulationParams->boundaryGridOrigin,
            simulationParams->boundaryGridDimensions,

            POLY6_COEFFICIENT
      );
}

void ParticlesKernel::calculatePressureAndViscosityForces()
{

    calculatePressureAndViscosityForcesKernel << <
        BLOCKS_COUNT,
        THREADS_PER_BLOCK,
        0,
        cudaContext->stream
        >> > (
            particleBufferB,
            simulationParams -> particlesCount,
            simulationParams -> mass,
            simulationParams -> smoothingRadius,
            simulationParams -> viscosityCoefficient,
            VISCOSITY_COEFFICIENT,
            SPIKY_COEFFICIENT,

            simulationParams -> bounds.min,
            simulationParams -> cellSize,
            simulationParams -> gridDimensions,

            cellStart,
            cellEnd
        );
}

void ParticlesKernel::initialize(CudaContext& _cudaContext, GPUSimulationParams &_simulationParams ) {
    cudaContext = &_cudaContext;
    simulationParams = &_simulationParams;

    THREADS_PER_BLOCK = 256;
    BLOCKS_COUNT = (simulationParams->particlesCount + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    BOUNDARY_BLOCKS_COUNT = (simulationParams->boundaryParticlesCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const unsigned int count = static_cast<unsigned int>(simulationParams -> particlesCount);
    const size_t bytes = count * sizeof(uint32_t);

    cudaMalloc(reinterpret_cast<void**>(&particleCellIds), bytes);
    cudaMalloc(reinterpret_cast<void**>(&sortedCellIds), bytes);
    cudaMalloc(reinterpret_cast<void**>(&particleIndices), bytes);
    cudaMalloc(reinterpret_cast<void**>(&sortedParticleIndices), bytes);

    initializeIndicesKernel << <BLOCKS_COUNT, THREADS_PER_BLOCK, 0, cudaContext->stream >> > ( particleIndices, count );

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

    float h = simulationParams->smoothingRadius;
    float h2 = h * h;
    float h3 = h2 * h;
    float h6 = h3 * h3;
    float h9 = h6 * h3;

    POLY6_COEFFICIENT = 315.0f / (64.0f * CUDART_PI_F * h9 );

    SPIKY_COEFFICIENT = -45.0f / (CUDART_PI_F * h6 );
    VISCOSITY_COEFFICIENT = 45.0f / (CUDART_PI_F * h6 );

    particleBufferA = static_cast<ParticlesKernel::CudaParticle*>(cudaBuffer);
    particleBufferB = static_cast<ParticlesKernel::CudaParticle*>(cudaBufferB);
}

__global__ void reorderParticlesKernel(
    const ParticlesKernel::CudaParticle* particles,
    ParticlesKernel::CudaParticle* reorderedParticles,
    const uint32_t* sortedParticleIndices,
    uint32_t particleCount)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= particleCount)
        return;

    uint32_t originalIndex = sortedParticleIndices[i];

    reorderedParticles[i] = particles[originalIndex];
}

void ParticlesKernel::launchReorder()
{
    reorderParticlesKernel << <
        BLOCKS_COUNT,
        THREADS_PER_BLOCK,
        0,
        cudaContext->stream
        >> > (
            particleBufferA,
            particleBufferB,
            sortedParticleIndices,
            simulationParams->particlesCount
            );
}

void ParticlesKernel::updateParticles(float deltaTime)
{
    updateKernel << <
        BLOCKS_COUNT,
        THREADS_PER_BLOCK,
        0,
        cudaContext->stream
        >> > (
            particleBufferB,
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