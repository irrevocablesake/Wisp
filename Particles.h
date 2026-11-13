#pragma once

#include "engine.h"
#include "CPUSimulationParams.h"

#include <cmath>      
#include <random>      
#include <glm/glm.hpp> 

class Particles {
	private:
		Engine* engine;

	public:

		Engine::Geometry particleGeometry;
		Engine::Geometry wireGeometry;

		struct Particle {
			glm::vec4 position{};
			glm::vec4 velocity{};
			float density;
			float pressure;
			float padding[2]{};
			float4 pressureForces;
			float4 viscosityForces;
		};

		std::vector< Particle > particles;

		Engine::BufferHandle PBHandle;
		VkDescriptorSet PBDescriptorSet;

		Engine::BufferHandle PBHandleB;
		VkDescriptorSet PBDescriptorSetB;

		VkDescriptorSetLayout PBDescriptorSetLayout;

		struct BoundaryParticle {
			glm::vec4 position{};
		};

		std::vector< BoundaryParticle > boundaryParticles;

		Engine::BufferHandle BPHandle;
		VkDescriptorSet BPDescriptorSet;
		VkDescriptorSetLayout BPDescriptorSetLayout;

		CPUSimulationParams* simulationParams = nullptr;

	public:
		void memoryAllocator( Engine::BufferHandle &BufferHandle, size_t size, VkDescriptorSet &descriptorSet, VkDescriptorSetLayout &descriptorSetLayout ) {
			BufferHandle.size = size;
			engine->resourceManager.setupBuffer(BufferHandle, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, false, true );

			if (descriptorSetLayout == VK_NULL_HANDLE) {
				VkDescriptorSetLayoutBinding binding{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
				};

				descriptorSetLayout = engine->resourceManager.createDescriptorSetLayout({ binding });
			}

			descriptorSet = engine->resourceManager.allocateDescriptorSets({ descriptorSetLayout });

			VkDescriptorBufferInfo bufferInfo{
				.buffer = BufferHandle.buffer,
				.offset = 0,
				.range = BufferHandle.size
			};

			std::vector< VkWriteDescriptorSet > bufferUpdate = { {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptorSet,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &bufferInfo
			} };

			engine->resourceManager.updateDescriptorSet(bufferUpdate);
		}

		void generateWireframe() {
			wireGeometry = engine->solidShapes.generateWireframe(simulationParams->bounds.min, simulationParams->bounds.max);
		}

		void generateBoundaryParticles() {

			const glm::vec3 boundsMin = simulationParams->bounds.min;
			const glm::vec3 boundsMax = simulationParams->bounds.max;
			const uint32_t boundsParticlesCound = simulationParams->boundaryParticlesCount;

			glm::vec3 dimensions = boundsMax - boundsMin;

			uint32_t countX = static_cast<uint32_t>( std::floor(dimensions.x / simulationParams->boundarySpacing)) + 1;
			uint32_t countY = static_cast<uint32_t>( std::floor(dimensions.y / simulationParams->boundarySpacing)) + 1;
			uint32_t countZ = static_cast<uint32_t>( std::floor(dimensions.z / simulationParams->boundarySpacing)) + 1;
		
			boundaryParticles.reserve( 2 * simulationParams->boundaryLayers * (countX * countY + countY * countZ + countX * countZ) );

			auto addBoundaryParticle = [&](float x, float y, float z) {
				BoundaryParticle particle{};
				particle.position = glm::vec4(x, y, z, 0.0f);
				boundaryParticles.push_back(particle);
			};

			// X-facing walls
			for (uint32_t layer = 0; layer < simulationParams->boundaryLayers; ++layer)
			{
				float offset = layer * simulationParams->boundarySpacing;

				for (uint32_t y = 0; y < countY; ++y)
				{
					for (uint32_t z = 0; z < countZ; ++z)
					{
						float py = boundsMin.y + y * simulationParams->boundarySpacing;
						float pz = boundsMin.z + z * simulationParams->boundarySpacing;

						addBoundaryParticle(boundsMin.x - offset, py, pz);
						addBoundaryParticle(boundsMax.x + offset, py, pz);
					}
				}
			}

			// Y-facing walls (exclude X-edge columns)
			for (uint32_t layer = 0; layer < simulationParams->boundaryLayers; ++layer)
			{
				float offset = layer * simulationParams->boundarySpacing;

				for (uint32_t x = 1; x + 1 < countX; ++x)
				{
					for (uint32_t z = 0; z < countZ; ++z)
					{
						float px = boundsMin.x + x * simulationParams->boundarySpacing;
						float pz = boundsMin.z + z * simulationParams->boundarySpacing;

						addBoundaryParticle(px, boundsMin.y - offset, pz);
						addBoundaryParticle(px, boundsMax.y + offset, pz);
					}
				}
			}

			// Z-facing walls (exclude X and Y edges)
			for (uint32_t layer = 0; layer < simulationParams->boundaryLayers; ++layer)
			{
				float offset = layer * simulationParams->boundarySpacing;

				for (uint32_t x = 1; x + 1 < countX; ++x)
				{
					for (uint32_t y = 1; y + 1 < countY; ++y)
					{
						float px = boundsMin.x + x * simulationParams->boundarySpacing;
						float py = boundsMin.y + y * simulationParams->boundarySpacing;

						addBoundaryParticle(px, py, boundsMin.z - offset);
						addBoundaryParticle(px, py, boundsMax.z + offset);
					}
				}
			}

			memoryAllocator(BPHandle, boundaryParticles.size() * sizeof(BoundaryParticle), BPDescriptorSet, BPDescriptorSetLayout);
			engine->resourceManager.uploadData(BPHandle, boundaryParticles.data(), BPHandle.size);

			simulationParams->boundaryParticlesCount = static_cast< uint32_t > ( boundaryParticles.size() );
			float PADDING = 3.0f * simulationParams->boundarySpacing;

			simulationParams->boundaryGridOrigin = glm::vec3(
				boundsMin.x - PADDING,
				boundsMin.y - PADDING,
				boundsMin.z - PADDING
			);

			simulationParams->boundaryGridDimensions = glm::uvec3(
				static_cast<uint32_t>(ceilf(
					(boundsMax.x - boundsMin.x + 2.0f * PADDING) /
					simulationParams->cellSize)) + 1,

				static_cast<uint32_t>(ceilf(
					(boundsMax.y - boundsMin.y + 2.0f * PADDING) /
					simulationParams->cellSize)) + 1,

				static_cast<uint32_t>(ceilf(
					(boundsMax.z - boundsMin.z + 2.0f * PADDING) /
					simulationParams->cellSize)) + 1
			);
		}

		void generate() {
			particleGeometry = engine->solidShapes.generateSphere(4, 4, simulationParams->particleRadius );

			particles.resize( simulationParams->particlesCount );

			const uint32_t particlesPerAxis = std::ceil( std::cbrt(particles.size()));

			//spacing affects rest density, radius is just for visual / collision
			const float jitter = simulationParams->latticeSpacing * simulationParams->latticeJitter;
			std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<float> jitterDistribution(-jitter, jitter);

			const glm::vec3 latticeStart( -0.5f * simulationParams->latticeSpacing * (particlesPerAxis - 1) );

			unsigned int index = 0;

			for (unsigned int z = 0; z < particlesPerAxis; ++z)
			{
				for (unsigned int y = 0; y < particlesPerAxis; ++y)
				{
					for (unsigned int x = 0; x < particlesPerAxis; ++x)
					{
						if (index >= particles.size())
							goto GENERATION_DONE;

						Particle& p = particles[index];

						glm::vec3 position( x * simulationParams->latticeSpacing + jitterDistribution(rng), y * simulationParams->latticeSpacing + jitterDistribution(rng), z * simulationParams->latticeSpacing + jitterDistribution(rng) );
						position = latticeStart + position;

						p.position = glm::vec4(position, 0.0f);

						p.velocity = glm::vec4(0.0f);
						p.density = 0.0f;
						p.pressure = 0.0f;
						p.pressureForces = float4(0.0f);
						p.viscosityForces = float4(0.0f);

						index = index + 1;
					}
				}
			}

			GENERATION_DONE:
				memoryAllocator(PBHandle, particles.size() * sizeof(Particle), PBDescriptorSet, PBDescriptorSetLayout );
				engine->resourceManager.uploadData(PBHandle, particles.data(), PBHandle.size);

				memoryAllocator(PBHandleB, particles.size() * sizeof(Particle), PBDescriptorSetB, PBDescriptorSetLayout);
				engine->resourceManager.uploadData(PBHandleB, particles.data(), PBHandleB.size);
		}

		void generate(Engine& _engine, CPUSimulationParams& _simulationParams) {
			engine = &_engine;
			simulationParams = &_simulationParams;

			generate();
			generateBoundaryParticles();
			generateWireframe();
		}

		int count() {
			return particles.size() + boundaryParticles.size();
		}

		int PCount() {
			return particles.size();
		}

		int BPCount() {
			return boundaryParticles.size();
		}
};