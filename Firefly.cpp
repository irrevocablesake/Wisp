#include "engine.h"

#include<iostream>

class Firefly {
	Engine engine;
	Engine::TextureHandle RenderTargetTexture;
	Engine::BufferHandle UBHandle;

	VkPipeline RTXPipeline;
	VkPipelineLayout RTXPipelineLayout;

	VkDescriptorSet UBDescriptorSet;
	VkDescriptorSetLayout UBDescriptorSetLayout;

	VkDescriptorSet RenderTargetDescriptorSet;
	VkDescriptorSetLayout RenderTargetDescriptorSetLayout;

	struct UniformData {
		glm::vec3 cameraPosition;
		float pad0;

		glm::vec3 viewportU;
		float pad1;

		glm::vec3 viewportV;
		float pad2;

		glm::vec3 pixelDeltaU;
		float pad3;

		glm::vec3 pixelDeltaV;
		float pad4;

		glm::vec3 viewportUPL;
		float pad5;

		glm::vec3 pixel00Location;
		float pad6;

		float imageWidth;
		float imageHeight;

		float aspectRatio;

		float viewportWidth;
		float viewportHeight;

		float focalLength;
	} uniformData;

	void generateRenderTarget() {
		//Generate Texture
		RenderTargetTexture = engine.generateTextureHandle();
		RenderTargetTexture.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
		RenderTargetTexture.imageFilter = VK_FILTER_LINEAR;
		RenderTargetTexture.width = engine.windowManager.windowSize.x;
		RenderTargetTexture.height = engine.windowManager.windowSize.y;
		engine.generateTexture(RenderTargetTexture);

		//Expose Texture through Descriptor
		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		};

		RenderTargetDescriptorSetLayout = engine.resourceManager.createDescriptorSetLayout({ binding });
		RenderTargetDescriptorSet = engine.resourceManager.allocateDescriptorSets( RenderTargetDescriptorSetLayout );

		std::vector< VkWriteDescriptorSet > imageSet = { {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = RenderTargetDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.pImageInfo = &RenderTargetTexture.descriptor
		} };

		engine.resourceManager.updateDescriptorSet(imageSet);
	}

	void generateUBBuffer() {
		UBHandle.size = sizeof(UniformData);
		engine.resourceManager.setupBuffer(UBHandle, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		};

		UBDescriptorSetLayout = engine.resourceManager.createDescriptorSetLayout({ binding });

		UBDescriptorSet = engine.resourceManager.allocateDescriptorSets({ UBDescriptorSetLayout });

		VkDescriptorBufferInfo bufferInfo{
			.buffer = UBHandle.buffer,
			.offset = 0,
			.range = UBHandle.size
		};

		std::vector< VkWriteDescriptorSet > bufferUpdate = { {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = UBDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &bufferInfo
		} };

		engine.resourceManager.updateDescriptorSet(bufferUpdate);

		uniformData.imageWidth = engine.windowManager.windowSize.x;
		uniformData.imageHeight = engine.windowManager.windowSize.y;
		uniformData.aspectRatio = ( ( float ) uniformData.imageWidth / uniformData.imageHeight );
		uniformData.focalLength = 1.0;
		uniformData.viewportHeight = 2.0;
		uniformData.viewportWidth = uniformData.viewportHeight * uniformData.aspectRatio;
		uniformData.cameraPosition = glm::vec3(0, 0, 0);
		uniformData.viewportU = glm::vec3( uniformData.viewportWidth, 0, 0 );
		uniformData.viewportV = glm::vec3( 0, -uniformData.viewportHeight, 0 );
		uniformData.pixelDeltaU = uniformData.viewportU / uniformData.imageWidth;
		uniformData.pixelDeltaV = uniformData.viewportV / uniformData.imageHeight;
		uniformData.viewportUPL = uniformData.cameraPosition - glm::vec3(0, 0, uniformData.focalLength) - ( uniformData.viewportU / 2.0f ) - ( uniformData.viewportV / 2.0f );
		uniformData.pixel00Location = uniformData.viewportUPL + 0.5f * ( uniformData.pixelDeltaU + uniformData.pixelDeltaV );

		memcpy(UBHandle.allocationInfo.pMappedData, &uniformData, sizeof(UniformData));
	}

	void generatePipeline() {
		std::vector< VkDescriptorSetLayout > layouts = { RenderTargetDescriptorSetLayout, UBDescriptorSetLayout };

		//COMPUTE PIPELINE
		VkPipelineLayoutCreateInfo rayTracerLayoutCI{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data(),
		};

		vkCreatePipelineLayout(engine.vulkanContext.logicalDeviceIF.handle, &rayTracerLayoutCI, nullptr, &RTXPipelineLayout);

		VkPipelineShaderStageCreateInfo rayTracerShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = engine.compileShader("rayTracerShaderModule", "assets/shaders/rayTracerShader.slang"),
			.pName = "main"
		};

		engine.pipelineManager.createComputePipeline(RTXPipeline, rayTracerShaderStage, RTXPipelineLayout);
	}

	void setup() {
		engine.configure("Firefly", VK_API_VERSION_1_4);

		engine.resourceManager.setupDescriptorPool({
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
		});

		generateRenderTarget();
		generateUBBuffer();
		generatePipeline();
	}

	void animate() {
		bool quit{ false };
		uint64_t startFrameTime = SDL_GetPerformanceCounter();
		uint64_t previousFrameTime = startFrameTime;
		uint64_t frequency = SDL_GetPerformanceFrequency();
		float elapsedTime = 0.0f;

		while (!quit) {
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_EVENT_QUIT) {
					quit = true;
				}

				if (event.type == SDL_EVENT_WINDOW_RESIZED) {
					engine.updateSwapchain();
				}
			}

			uint64_t currentFrameTime = SDL_GetPerformanceCounter();

			float dt = (float)(currentFrameTime - previousFrameTime) / (float)frequency;
			previousFrameTime = currentFrameTime;

			elapsedTime = (float)(currentFrameTime - startFrameTime) / (float)frequency;

			vkWaitForFences( engine.vulkanContext.logicalDeviceIF.handle, 1, &engine.getCurrentFrame().frameFence, true, UINT64_MAX);
			vkResetFences(engine.vulkanContext.logicalDeviceIF.handle, 1, &engine.getCurrentFrame().frameFence);

			vkAcquireNextImageKHR(engine.vulkanContext.logicalDeviceIF.handle, engine.swapchainManager.swapchain, UINT64_MAX, engine.getCurrentFrame().imageAvailable, VK_NULL_HANDLE, &engine.swapchainManager.imageIndex);

			auto commandBuffer = engine.vulkanContext.getCurrentFrame().commandBuffer;
			vkResetCommandBuffer(commandBuffer, 0);

			VkCommandBufferBeginInfo commandBufferBeginInfo{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
			};

			vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

			VkViewport viewport{
				.width = float(engine.windowManager.windowSize.x),
				.height = float(engine.windowManager.windowSize.y),
				.minDepth = 0.0f,
				.maxDepth = 1.0f
			};

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			VkRect2D scissor{
				.extent {
					.width = static_cast<uint32_t>(engine.windowManager.windowSize.x),
					.height = static_cast<uint32_t>(engine.windowManager.windowSize.y),
				}
			};
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			uniformData.imageWidth = engine.windowManager.windowSize.x;
			uniformData.imageHeight = engine.windowManager.windowSize.y;

			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RTXPipelineLayout, 0, 1, &RenderTargetDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RTXPipelineLayout, 1, 1, &UBDescriptorSet, 0, nullptr);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RTXPipeline);
			vkCmdDispatch(commandBuffer, ( engine.windowManager.windowSize.x + 7) / 8, (engine.windowManager.windowSize.y + 7) / 8, 1);

			engine.barrier.transitionImageFromGeneralToTransferSrc(commandBuffer, RenderTargetTexture.image);
			engine.barrier.transitionImageFromUndefinedToTransferDst(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);

			VkImageCopy copyRegion{
				.srcSubresource{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.layerCount = 1
				},
				.dstSubresource{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.layerCount = 1
				},
				.extent{
					.width = engine.windowManager.surfaceCapabilites.currentExtent.width,
					.height = engine.windowManager.surfaceCapabilites.currentExtent.height,
					.depth = 1
				}
			};

			vkCmdCopyImage(
				commandBuffer,
				RenderTargetTexture.image,
				VK_IMAGE_LAYOUT_GENERAL,
				engine.swapchainManager.images[ engine.swapchainManager.imageIndex ],
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&copyRegion
			);

			engine.barrier.transitionImageTransferDstToPresent(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			engine.barrier.transitionImageFromTransferSrcToGeneral(commandBuffer, RenderTargetTexture.image);

			vkEndCommandBuffer(commandBuffer);

			VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo submitInfo{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &engine.getCurrentFrame().imageAvailable,
				.pWaitDstStageMask = &waitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &commandBuffer,
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &engine.swapchainManager.renderSemaphores[engine.swapchainManager.imageIndex],
			};
			vkQueueSubmit( engine.vulkanContext.queueIF.handle, 1, &submitInfo, engine.getCurrentFrame().frameFence );

			engine.vulkanContext.currentFrame = (engine.vulkanContext.currentFrame + 1) % engine.vulkanContext.MAX_FRAMES_IN_FLIGHT;

			VkPresentInfoKHR presentInfo{
				.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &engine.swapchainManager.renderSemaphores[engine.swapchainManager.imageIndex],
				.swapchainCount = 1,
				.pSwapchains = &engine.swapchainManager.swapchain,
				.pImageIndices = &engine.swapchainManager.imageIndex
			};
			vkQueuePresentKHR(engine.vulkanContext.queueIF.handle, &presentInfo);

			engine.updateSwapchain();
		}
	}

public:
	void simulate() {
		setup();
		animate();
	}
};

int main() {
	Firefly firefly;
	firefly.simulate();

	return 0;
}