#pragma once

#include "engine.h"

#include <cmath>      
#include <random>      
#include <glm/glm.hpp> 

class Particles {
	private:
		Engine* engine;

	public:

		Engine::Geometry particleGeometry;

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
		VkDescriptorSetLayout PBDescriptorSetLayout;

		struct BoundaryParticle {
			glm::vec4 position{};
		};

		std::vector< BoundaryParticle > boundaryParticles;

		Engine::BufferHandle BPHandle;
		VkDescriptorSet BPDescriptorSet;
		VkDescriptorSetLayout BPDescriptorSetLayout;

		float spacing = 0.04f;
		float jitterFactor = 0.05f;
		uint32_t layers = 3;

	public:
		void initialize(Engine& _engine) {
			engine = &_engine;
		}

		void memoryAllocator( Engine::BufferHandle &BufferHandle, size_t size, VkDescriptorSet &descriptorSet, VkDescriptorSetLayout &descriptorSetLayout ) {
			BufferHandle.size = size;
			engine->resourceManager.setupBuffer(BufferHandle, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

			VkDescriptorSetLayoutBinding binding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
			};

			descriptorSetLayout = engine->resourceManager.createDescriptorSetLayout({ binding });
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

		void generateBoundaryParticles(const glm::vec3& boundsMin, const glm::vec3& boundsMax) {
			glm::vec3 dimensions = boundsMax - boundsMin;

			uint32_t countX = static_cast<uint32_t>( std::floor(dimensions.x / spacing)) + 1;
			uint32_t countY = static_cast<uint32_t>( std::floor(dimensions.y / spacing)) + 1;
			uint32_t countZ = static_cast<uint32_t>( std::floor(dimensions.z / spacing)) + 1;
		
			boundaryParticles.reserve( 2 * layers * (countX * countY + countY * countZ + countX * countZ) );

			auto addBoundaryParticle = [&](float x, float y, float z) {
				BoundaryParticle particle{};
				particle.position = glm::vec4(x, y, z, 0.0f);
				boundaryParticles.push_back(particle);
			};

			// X-facing walls
			for (uint32_t layer = 0; layer < layers; ++layer)
			{
				float offset = layer * spacing;

				for (uint32_t y = 0; y < countY; ++y)
				{
					for (uint32_t z = 0; z < countZ; ++z)
					{
						float py = boundsMin.y + y * spacing;
						float pz = boundsMin.z + z * spacing;

						addBoundaryParticle(boundsMin.x - offset, py, pz);
						addBoundaryParticle(boundsMax.x + offset, py, pz);
					}
				}
			}

			// Y-facing walls (exclude X-edge columns)
			for (uint32_t layer = 0; layer < layers; ++layer)
			{
				float offset = layer * spacing;

				for (uint32_t x = 1; x + 1 < countX; ++x)
				{
					for (uint32_t z = 0; z < countZ; ++z)
					{
						float px = boundsMin.x + x * spacing;
						float pz = boundsMin.z + z * spacing;

						addBoundaryParticle(px, boundsMin.y - offset, pz);
						addBoundaryParticle(px, boundsMax.y + offset, pz);
					}
				}
			}

			// Z-facing walls (exclude X and Y edges)
			for (uint32_t layer = 0; layer < layers; ++layer)
			{
				float offset = layer * spacing;

				for (uint32_t x = 1; x + 1 < countX; ++x)
				{
					for (uint32_t y = 1; y + 1 < countY; ++y)
					{
						float px = boundsMin.x + x * spacing;
						float py = boundsMin.y + y * spacing;

						addBoundaryParticle(px, py, boundsMin.z - offset);
						addBoundaryParticle(px, py, boundsMax.z + offset);
					}
				}
			}

			memoryAllocator(BPHandle, boundaryParticles.size() * sizeof(BoundaryParticle), BPDescriptorSet, BPDescriptorSetLayout);
			engine->resourceManager.copyDataIntoBuffer(BPHandle.allocationInfo.pMappedData, boundaryParticles.data(), BPHandle.size);
		}

		void generate(int count, float radius ) {
			particleGeometry = engine->solidShapes.generateSphere(32, 32,radius );

			particles.resize(count);

			const uint32_t particlesPerAxis = std::ceil( std::cbrt(particles.size()));

			//spacing affects rest density, radius is just for visual / collision
			const float jitter = spacing * jitterFactor;
			std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<float> jitterDistribution(-jitter, jitter);

			const glm::vec3 latticeStart( -0.5f * spacing * (particlesPerAxis - 1) );

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

						glm::vec3 position( x * spacing + jitterDistribution(rng), y * spacing + jitterDistribution(rng), z * spacing + jitterDistribution(rng) );
						position = latticeStart + position;

						p.position = glm::vec4(position, 0.0f);

						p.velocity = glm::vec4(2.0f * position.y, 0.0, 0.0, 0.0);
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
				engine->resourceManager.copyDataIntoBuffer(PBHandle.allocationInfo.pMappedData, particles.data(), PBHandle.size);
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