#pragma once

#include<glm/glm.hpp>

//Note: Ensure cellsize >= Smoothing Radius
struct CPUSimulationParams {

	float latticeSpacing = 0.08f;
	float boundarySpacing = 0.08f;
	float latticeJitter = 0.01f;
	
	float gravity = -9.81f;
	int boundaryLayers = 3;

	float particleRadius = 0.12f;
	float restitution = 0.8f;
	float smoothingRadius = 0.16f;
	float mass = 0.512f;
	float viscosityCoefficient = 0.01f;
	float restDensity = 1000.0f;

	float cflCoefficient = 0.4f;
	float accelerationCoefficient = 0.2f;
	float maxDeltaTime = 0.001f;

	//int particlesCount = 40960;
	//int particlesCount = 131072;
	 int particlesCount = 102400;
	//int particlesCount = 204800;
	//int particlesCount = 409600;
	//int particlesCount = 1024000;
	//int particlesCount = 2048000;

	float gasConstant = 1600.0f;

	float cellSize = 0.16f;

	uint32_t boundaryParticlesCount = 0;
	float boundaryParticleMass = 0.512;

	struct SimulationBounds {
		glm::vec3 min = glm::vec3(-6.0f, -2.0f, -3.0f);
		glm::vec3 max = glm::vec3(6.0f, 2.0f, 3.0f);
		glm::vec3 dimensions = max - min;
	} bounds;

	glm::uvec3 gridDimensions = glm::uvec3(
		glm::ceil(bounds.dimensions / cellSize)
	);

	glm::vec3 boundaryGridOrigin{};
	glm::uvec3 boundaryGridDimensions{};
};