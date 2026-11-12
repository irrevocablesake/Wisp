#pragma once

#include<glm/glm.hpp>

//will be useful to scale particles count
constexpr int THREADS_PER_BLOCK = 256;

//NOTE: Do not do particlesCount as 0

//Note: Ensure cellsize >= Smoothing Radius
struct CPUSimulationParams {

	float latticeSpacing = 0.04f;
	float boundarySpacing = 0.04f;
	float latticeJitter = 0.04f;
	
	float gravity = -9.81f;

	int boundaryLayers = 3;

	float particleRadius = 0.06f;
	float restitution = 0.8f;
	float smoothingRadius = 0.08f;
	float mass = 1.0f;
	float viscosityCoefficient = 0.01f;
	float restDensity = 1000.0f;

	float speedOfSound = 0.7746f;
	float cflCoefficient = 0.4f;
	float accelerationCoefficient = 0.2f;
	float maxDeltaTime = 0.04f;

	//int particlesCount = 40960;
	//int particlesCount = 131072;
	 //int particlesCount = 102400;
	//int particlesCount = 204800;
	//int particlesCount = 409600;
	//int particlesCount = 1024000;
	int particlesCount = 2048000;
	//int particlesCount = 10240000;

	float gasConstant = 0.006f;

	float cellSize = 0.08f;

	uint32_t boundaryParticlesCount = 0;
	float boundaryParticleMass  = restDensity * boundarySpacing * boundarySpacing * boundarySpacing;

	struct SimulationBounds {
		glm::vec3 min = glm::vec3(-4.0f);
		glm::vec3 max = glm::vec3(4.0f);
		glm::vec3 dimensions = max - min;
	} bounds;

	glm::uvec3 gridDimensions = glm::uvec3(
		glm::ceil(bounds.dimensions / cellSize)
	);

	glm::vec3 boundaryGridOrigin{};
	glm::uvec3 boundaryGridDimensions{};
};