#pragma once

#include "engine.h"

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
	};

	std::vector< Particle > particles;

	Engine::BufferHandle PBHandle;
	VkDescriptorSet PBDescriptorSet;
	VkDescriptorSetLayout PBDescriptorSetLayout;

public:
	void initialize(Engine& _engine) {
		engine = &_engine;
	}

	void generate(int count) {
		particleGeometry = engine->solidShapes.generateSphere(32, 32);

		particles.resize(count);

		for (Particle& p : particles) {
			p.position = glm::vec4(
				engine->generateRandomNumber(-0.8, 0.8),
				engine->generateRandomNumber(-0.8, 0.8),
				engine->generateRandomNumber(-0.8, 0.8),
				0.0f
			);

			p.velocity = glm::vec4(
				engine->generateRandomNumber(-0.2, 0.2),
				engine->generateRandomNumber(-0.2, 0.2),
				engine->generateRandomNumber(-0.2, 0.2),
				0.0f
			);

			p.density = 0.0f;
		}

		PBHandle.size = particles.size() * sizeof(Particle);
		engine->resourceManager.setupBuffer(PBHandle, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
		};

		PBDescriptorSetLayout = engine->resourceManager.createDescriptorSetLayout({ binding });
		PBDescriptorSet = engine->resourceManager.allocateDescriptorSets({ PBDescriptorSetLayout });

		VkDescriptorBufferInfo bufferInfo{
			.buffer = PBHandle.buffer,
			.offset = 0,
			.range = PBHandle.size
		};

		std::vector< VkWriteDescriptorSet > bufferUpdate = { {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = PBDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &bufferInfo
		} };

		engine->resourceManager.updateDescriptorSet(bufferUpdate);
		engine->resourceManager.copyDataIntoBuffer(PBHandle.allocationInfo.pMappedData, particles.data(), PBHandle.size);
	}

	int count() {
		return particles.size();
	}
};