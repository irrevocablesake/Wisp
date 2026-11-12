#pragma once

#include<glm/glm.hpp>

//will be useful to scale particles count
constexpr int THREADS_PER_BLOCK = 256;

//NOTE: Do not do particlesCount as 0

//Note: Ensure cellsize >= Smoothing Radius
struct CPUSimulationParams {
	float latticeSpacing = 0.04f;
	float latticeJitter = 0.1f * latticeSpacing;

	int boundaryLayers = 3;
	float boundarySpacing = latticeSpacing;
	
	float gravity = -9.81f;

	float particleRadius = 0.06f;
	float restitution = 0.0f;
	float smoothingRadius = 2.0f * particleRadius;
	float mass = 0.064f;
	float viscosityCoefficient = 0.00001f;
	float speedOfSound = 10.0f;
	float gasConstant = 0.006;
	float cellSize = smoothingRadius;
	float restDensity = ( mass / ( latticeSpacing * latticeSpacing * latticeSpacing ));

	
	float cflCoefficient = 0.4f;
	float accelerationCoefficient = 0.2f;
	float maxDeltaTime = 0.04f;

	//int particlesCount = 40960;
	//int particlesCount = 131072;
	 //int particlesCount = 102400;
	//int particlesCount = 204800;
	//int particlesCount = 409600;
	int particlesCount = 1024000;
	//int particlesCount = 2048000;
	//int particlesCount = 10240000;

	uint32_t boundaryParticlesCount; //DO NOT CONFIGURE
	float boundaryParticleMass  = restDensity * boundarySpacing * boundarySpacing * boundarySpacing;

	struct SimulationBounds {
		glm::vec3 min = glm::vec3(-3.0f);
		glm::vec3 max = glm::vec3(3.0f);
		glm::vec3 dimensions = max - min;
	} bounds;

	glm::uvec3 gridDimensions = glm::uvec3(
		glm::ceil(bounds.dimensions / cellSize)
	);

	glm::vec3 boundaryGridOrigin{};//DO NOT CONFIGURE
	glm::uvec3 boundaryGridDimensions{};//DO NOT CONFIGURE
};