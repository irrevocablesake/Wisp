#pragma once

#include<glm/glm.hpp>

struct CPUSimulationParams {

	float latticeSpacing = 0.04f;
	float boundarySpacing = 0.04f;
	float latticeJitter = 0.04f;
	
	float gravity = -9.81f;

	int boundaryLayers = 3;

	float particleRadius = 0.02f;
	float restitution = 0.8f;
	float smoothingRadius = 0.08f;
	float mass = 1.0f;
	float viscosityCoefficient = 0.1f;
	float restDensity = 1000.0f;

	int particlesCount = 40960;

	float gasConstant = 0.006f;

	float cellSize = 0.04f;

	uint32_t boundaryParticlesCount = 0;
	float boundaryParticleMass  = restDensity * boundarySpacing * boundarySpacing * boundarySpacing;

	struct SimulationBounds {
		glm::vec3 min = glm::vec3(-2.0f);
		glm::vec3 max = glm::vec3(2.0f);
		glm::vec3 dimensions = max - min;
	} bounds;

	glm::uvec3 gridDimensions = glm::uvec3(
		glm::ceil(bounds.dimensions / cellSize)
	);

	glm::vec3 boundaryGridOrigin{};
	glm::uvec3 boundaryGridDimensions{};
};