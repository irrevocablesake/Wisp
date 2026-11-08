#pragma once

#include "CPUSimulationParams.h"
#include <cuda_runtime.h>

struct GPUSimulationParams {
	float gravity;

	uint32_t boundaryParticlesCount;
	float boundaryParticleMass;

	float particleRadius;
	float restitution;
	float smoothingRadius;
	float mass;

	float viscosityCoefficient;
	float restDensity;
	float gasConstant;

	int particlesCount;
	float cellSize;

	struct SimulationBounds {
		float3 min;
		float3 max;
		float3 dimensions;
	} bounds;

	uint3 gridDimensions;
};

inline GPUSimulationParams toCudaParams(
	const CPUSimulationParams& source )
{
	GPUSimulationParams result{};

	result.gravity = source.gravity;
	result.boundaryParticlesCount = source.boundaryParticlesCount;
	result.boundaryParticleMass = source.boundaryParticleMass;
	result.particleRadius = source.particleRadius;
	result.restitution = source.restitution;
	result.smoothingRadius = source.smoothingRadius;
	result.mass = source.mass;
	result.viscosityCoefficient = source.viscosityCoefficient;
	result.restDensity = source.restDensity;
	result.gasConstant = source.gasConstant;
	result.particlesCount = source.particlesCount;
	result.cellSize = source.cellSize;
	result.bounds.max = make_float3( source.bounds.max.x, source.bounds.max.y, source.bounds.max.z );
	result.bounds.min = make_float3(source.bounds.min.x, source.bounds.min.y, source.bounds.min.z);
	result.bounds.dimensions = make_float3(source.bounds.dimensions.x, source.bounds.dimensions.y, source.bounds.dimensions.z);
	result.gridDimensions = make_uint3( source.gridDimensions.x, source.gridDimensions.y, source.gridDimensions.z );

	return result;
}