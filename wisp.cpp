#include "CudaInterface.h"
#include "Particles.h"
#include "engine.h"

#include "CPUSimulationParams.h"

#include<glm/gtc/matrix_transform.hpp>

//NOTE: I think we should also add the ability to dump data from shader / cuda and read it in CPU 
// We may already be able to do that through interop but still

//NOTE: Should add the checks for density validation
class Wisp {

	//Engine Initialization
	Engine engine;

	//SEMAPHORE
	VkSemaphore cudaToVk;
	VkSemaphore vkToCuda;

	HANDLE cudaToVkHandle;
	HANDLE vkToCudaHandle;

	//CUDA Context
	CudaInterface cudaInterface;

	//UNIFORM BUFFER
	Engine::BufferHandle UBHandle;
	VkDescriptorSet UBDescriptorSet;
	VkDescriptorSetLayout UBDescriptorSetLayout;

	//Graphics Pipeline
	VkPipeline GraphicsPipeline;
	VkPipelineLayout GraphicsPipelineLayout;

	//Particles
	Particles particles;

	CPUSimulationParams simulationParams;

	struct UniformData {
		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
	} uniformData;

	struct Camera {
		glm::vec3 position{ 3.0f, 2.0f, 3.0f };
		glm::vec3 cameraUp{ 0, 1, 0 };
		glm::vec3 lookAt{ 0.0f, -2.0f, 0.0f };
	} camera;

	float deltaTime = 0.0f;
	void setupUB(){
		UBHandle.size = sizeof( UniformData );
		engine.resourceManager.setupBuffer(UBHandle, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);

		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
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
		engine.resourceManager.copyDataIntoBuffer(UBHandle.allocationInfo.pMappedData, &uniformData, UBHandle.size);
	}

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

		std::vector< VkPipelineShaderStageCreateInfo > shaderStages = { vertexShaderStage, fragmentShaderStage };

		std::vector< VkDescriptorSetLayout > layouts = { particles.PBDescriptorSetLayout, UBDescriptorSetLayout };

		VkVertexInputBindingDescription binding{
			.binding = 0,
			.stride = sizeof(Engine::Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};

		std::vector<VkVertexInputAttributeDescription> attributes = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Engine::Vertex, position)
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Engine::Vertex, normal)
			}
		};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &binding,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
			.pVertexAttributeDescriptions = attributes.data()
		};

		engine.pipelineManager.createGraphicsPipeline(GraphicsPipeline, GraphicsPipelineLayout, layouts, VK_FORMAT_B8G8R8A8_SRGB, shaderStages, vertexInputInfo, 0);
	}

	void generateParticles() {
		particles.generate( engine, simulationParams );

		HANDLE PsharedHandle = engine.resourceManager.exportBuffer(particles.PBHandle);

		CudaInterface::CudaBuffer PResult = cudaInterface.cudaContext.importBuffer(
			PsharedHandle,
			particles.PBHandle.allocationInfo.size,
			particles.PBHandle.size
		);

		cudaInterface.particlesKernel.cudaBuffer = PResult.devicePointer;
		cudaInterface.particlesKernel.externalMemory = PResult.externalMemory;

		HANDLE BPsharedHandle = engine.resourceManager.exportBuffer(particles.BPHandle);

		CudaInterface::CudaBuffer BPResult = cudaInterface.cudaContext.importBuffer(
			BPsharedHandle,
			particles.BPHandle.allocationInfo.size,
			particles.BPHandle.size
		);

		cudaInterface.particlesKernel.BPcudaBuffer = BPResult.devicePointer;
		cudaInterface.particlesKernel.BPexternalMemory = BPResult.externalMemory;
	}

	void setup() {
		engine.configure("Wisp", VK_API_VERSION_1_4);

		engine.resourceManager.setupDescriptorPool({
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
		});

		generateParticles();

		setupUB();

		generatePipeline();

		vkToCuda = engine.vulkanContext.createExportableSemaphore();
		cudaToVk = engine.vulkanContext.createExportableSemaphore();

		vkToCudaHandle = engine.vulkanContext.exportSemaphore(vkToCuda);
		cudaToVkHandle = engine.vulkanContext.exportSemaphore(cudaToVk);

		cudaInterface.particlesKernel.vkToCuda = cudaInterface.importSemaphore( vkToCudaHandle );
		cudaInterface.particlesKernel.cudaToVk = cudaInterface.importSemaphore( cudaToVkHandle );

		cudaInterface.initialize(engine.vulkanContext.physicalDeviceIF.uuid, simulationParams);

		cudaInterface.particlesKernel.signalCudaFinished();
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

			deltaTime = (float)(currentFrameTime - previousFrameTime) / (float)frequency;
			previousFrameTime = currentFrameTime;

			elapsedTime = (float)(currentFrameTime - startFrameTime) / (float)frequency;

			vkWaitForFences( engine.vulkanContext.logicalDeviceIF.handle, 1, &engine.getCurrentFrame().frameFence, true, UINT64_MAX);

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

			uniformData.projection = glm::perspective(glm::radians(45.0f), (float) engine.windowManager.windowSize.x / (float)engine.windowManager.windowSize.y, 0.1f, 200.0f);
			uniformData.projection[1][1] *= -1;
			uniformData.view = glm::lookAt(camera.position, camera.lookAt, camera.cameraUp);

			engine.resourceManager.copyDataIntoBuffer(UBHandle.allocationInfo.pMappedData, &uniformData, UBHandle.size);

			VkDeviceSize offset = 0;

			engine.barrier.transitionImageFromUndefinedToColorAttachment(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			Engine::RenderingAttachment renderingAttachment = engine.setRenderingAttachment(engine.swapchainManager.imageViews[engine.swapchainManager.imageIndex] );
			vkCmdBeginRendering(commandBuffer, &renderingAttachment.renderingInfo );
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &particles.particleGeometry.VHandle.buffer, &offset);
			vkCmdBindIndexBuffer(commandBuffer, particles.particleGeometry.IHandle.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 0, 1, &particles.PBDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 1, 1, &UBDescriptorSet, 0, nullptr);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);
			vkCmdDrawIndexed(commandBuffer, particles.particleGeometry.indices.size(), particles.PCount(), 0, 0, 0);
			vkCmdEndRendering(commandBuffer);
			engine.barrier.transitionImageFromColorAttachmentToPresent(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			
			vkEndCommandBuffer(commandBuffer);

			VkSemaphore waitSemaphores[] = {
				engine.getCurrentFrame().imageAvailable,
				cudaToVk
			};

			VkPipelineStageFlags waitStages[] = {
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
			};

			VkSemaphore signalSemaphores[] = {
				engine.swapchainManager.renderSemaphores[
					engine.swapchainManager.imageIndex
				],
				vkToCuda
			};

			VkSubmitInfo submitInfo{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.waitSemaphoreCount = 2,
				.pWaitSemaphores = waitSemaphores,
				.pWaitDstStageMask = waitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &commandBuffer,
				.signalSemaphoreCount = 2,
				.pSignalSemaphores = signalSemaphores
			};

			vkResetFences(engine.vulkanContext.logicalDeviceIF.handle, 1, &engine.getCurrentFrame().frameFence);

			vkQueueSubmit(
				engine.vulkanContext.queueIF.handle,
				1,
				&submitInfo,
				engine.getCurrentFrame().frameFence
			);

			cudaInterface.particlesKernel.cudaWaitingForVulkan();


			cudaInterface.particlesKernel.generateIDs();
			cudaInterface.particlesKernel.sortIDs();
			cudaInterface.particlesKernel.buildCellRanges();
			//cudaInterface.particlesKernel.findNeighbors();
			cudaInterface.particlesKernel.calculateDensity();
			cudaInterface.particlesKernel.calculatePressure();
			cudaInterface.particlesKernel.calculatePressureForces();
			cudaInterface.particlesKernel.calculateViscosityForces();

			cudaInterface.particlesKernel.updateParticles( 0.005 );

			cudaInterface.particlesKernel.signalCudaFinished();


			engine.vulkanContext.currentFrame = (engine.vulkanContext.currentFrame + 1) % engine.vulkanContext.MAX_FRAMES_IN_FLIGHT;

			VkPresentInfoKHR presentInfo{
				.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &engine.swapchainManager.renderSemaphores[engine.swapchainManager.imageIndex],
				.swapchainCount = 1,
				.pSwapchains = &engine.swapchainManager.swapchain,
				.pImageIndices = &engine.swapchainManager.imageIndex
			};

			VkResult presentResult = vkQueuePresentKHR(
				engine.vulkanContext.queueIF.handle,
				&presentInfo
			);
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