#include "engine.h"

#include<iostream>

class Wisp {
	Engine engine;
	Engine::TextureHandle RenderTargetTexture;
	Engine::BufferHandle UBHandle;

	VkPipeline GraphicsPipeline;
	VkPipelineLayout GraphicsPipelineLayout;

	VkDescriptorSet UBDescriptorSet;
	VkDescriptorSetLayout UBDescriptorSetLayout;

	VkDescriptorSet RenderTargetDescriptorSet;
	VkDescriptorSetLayout RenderTargetDescriptorSetLayout;

	struct UniformData {
		float index;
	} uniformData;

	void generatePipeline() {
		//GRAPHICS PIPELINE
		VkPipelineShaderStageCreateInfo vertexShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = engine.compileShader("vertexModule", "assets/shaders/vertexShader.slang"),
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo fragmentShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = engine.compileShader("fragmentModule", "assets/shaders/fragmentShader.slang"),
			.pName = "main"
		};

		std::vector< VkPipelineShaderStageCreateInfo > shaderStages;
		shaderStages.push_back(vertexShaderStage);
		shaderStages.push_back(fragmentShaderStage);

		engine.pipelineManager.createGraphicsPipeline(GraphicsPipeline, GraphicsPipelineLayout, {}, VK_FORMAT_B8G8R8A8_SRGB, shaderStages, 0);
	}

	void setup() {
		engine.configure("Wisp", VK_API_VERSION_1_4);
		generatePipeline();
	}

	struct RenderingAttachment {
		VkRenderingAttachmentInfo attachmentInfo;
		VkRenderingInfo renderingInfo;
	};

	RenderingAttachment setRenderingAttachment(VkImageView& imageView) {
		RenderingAttachment attachment{};
		attachment.attachmentInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = imageView,
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue{
				engine.windowManager.clearColor
			}
		};

		attachment.renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea{
				.extent{
					.width = static_cast<uint32_t>( engine.windowManager.windowSize.x),
					.height = static_cast<uint32_t>(engine.windowManager.windowSize.y )
				},
			},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachment.attachmentInfo
		};

		return attachment;
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

			engine.barrier.transitionImageFromUndefinedToColorAttachment(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			RenderingAttachment renderingAttachment = setRenderingAttachment(engine.swapchainManager.imageViews[engine.swapchainManager.imageIndex] );
			vkCmdBeginRendering(commandBuffer, &renderingAttachment.renderingInfo );
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);
			vkCmdDraw(commandBuffer, 3, 1, 0, 0);
			vkCmdEndRendering(commandBuffer);
			engine.barrier.transitionImageFromColorAttachmentToPresent(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			
			vkEndCommandBuffer(commandBuffer);

			VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
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
	Wisp wisp;
	wisp.simulate();

	return 0;
}