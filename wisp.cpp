#include "CudaInterface.cuh"
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

	//Fluid Depth Texture
	Engine::TextureHandle FDTextureHandle;
	VkDescriptorSet FDTextureDescriptorSet;
	VkDescriptorSetLayout FDTextureDescriptorSetLayout;

	//Graphics Pipeline
	VkPipeline GraphicsPipeline;
	VkPipelineLayout GraphicsPipelineLayout;

	//Depth Pipeline
	VkPipeline DepthPipeline;
	VkPipelineLayout DepthPipelineLayout;

	VkPipeline FinalPipeline;
	VkPipelineLayout FinalPipelineLayout;

	//Particles
	Particles particles;

	CPUSimulationParams simulationParams;

	struct UniformData {
		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
	} uniformData;

	struct Camera {
		glm::vec3 position{ 10.0f, 10.0f, 10.0f };
		glm::vec3 cameraUp{ 0, 1, 0 };
		glm::vec3 lookAt{ 0.0f, -0.0f, 0.0f };
	} camera;

	//struct Camera {
	//	glm::vec3 position{ 6.0f, 6.0f, 6.0f };
	//	glm::vec3 cameraUp{ 0, 1, 0 };
	//	glm::vec3 lookAt{ 0.0f, -0.0f, 0.0f };
	//} camera;

	//struct Camera {
	//	glm::vec3 position{ 3.0f, -2.0f, 3.0f };
	//	glm::vec3 cameraUp{ 0, 1, 0 };
	//	glm::vec3 lookAt{ 0.0f, -2.0f, 0.0f };
	//} camera;

	float deltaTime = 0.0f;
	void setupUB(){
		UBHandle.size = sizeof( UniformData );
		engine.resourceManager.setupBuffer(UBHandle, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, false, false );

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

	void setupFluidDepthTexture() {
		FDTextureHandle = engine.generateTextureHandle();
		FDTextureHandle.imageFormat = VK_FORMAT_D32_SFLOAT;
		FDTextureHandle.imageFilter = VK_FILTER_NEAREST;
		FDTextureHandle.width = engine.windowManager.windowSize.x;
		FDTextureHandle.height = engine.windowManager.windowSize.y;
		FDTextureHandle.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		engine.generateTexture(FDTextureHandle, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
		};

		FDTextureDescriptorSetLayout = engine.resourceManager.createDescriptorSetLayout({
			binding
		});

		FDTextureDescriptorSet = engine.resourceManager.allocateDescriptorSets({
			FDTextureDescriptorSetLayout
		});

		std::vector< VkWriteDescriptorSet > imageSet = { {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = FDTextureDescriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &FDTextureHandle.descriptor
		} };

		engine.resourceManager.updateDescriptorSet(imageSet);
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

		engine.pipelineManager.createGraphicsPipeline(GraphicsPipeline, GraphicsPipelineLayout, layouts, VK_FORMAT_B8G8R8A8_SRGB, shaderStages, vertexInputInfo, 0, false, true );
	}

	//NOTE: NEED TO WORK ON THE SHADER MODULE NAMES + FILE NAMES
	void generateDepthPipeline() {
		//DEPTH PIPELINE
		VkPipelineShaderStageCreateInfo vertexShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = engine.compileShader("vertexModuleNew", "assets/shaders/depthVertexShader.slang"),
			.pName = "main"
		};

		std::vector< VkPipelineShaderStageCreateInfo > shaderStages = { vertexShaderStage };

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

		engine.pipelineManager.createGraphicsPipeline(DepthPipeline, DepthPipelineLayout, layouts, VK_FORMAT_UNDEFINED, shaderStages, vertexInputInfo, 0, true, true);
	}

	void generateFinalPipeline() {
		//FINAL PIPELINE
		VkPipelineShaderStageCreateInfo vertexShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = engine.compileShader("vertexModuleNEWNEW", "assets/shaders/finalVertexShader.slang"),
			.pName = "main"
		};

		VkPipelineShaderStageCreateInfo fragmentShaderStage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = engine.compileShader("fragmentModuleNEWNEW", "assets/shaders/finalFragmentShader.slang"),
			.pName = "main"
		};

		std::vector< VkPipelineShaderStageCreateInfo > shaderStages = { vertexShaderStage, fragmentShaderStage };

		std::vector< VkDescriptorSetLayout > layouts = { FDTextureDescriptorSetLayout };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 0,
			.pVertexBindingDescriptions = nullptr,
			.vertexAttributeDescriptionCount = 0,
			.pVertexAttributeDescriptions = nullptr
		};

		engine.pipelineManager.createGraphicsPipeline(FinalPipeline, FinalPipelineLayout, layouts, VK_FORMAT_B8G8R8A8_SRGB, shaderStages, vertexInputInfo, 0, false, false);
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

		HANDLE PsharedHandleB = engine.resourceManager.exportBuffer(particles.PBHandleB);

		CudaInterface::CudaBuffer PResultB = cudaInterface.cudaContext.importBuffer(
			PsharedHandleB,
			particles.PBHandleB.allocationInfo.size,
			particles.PBHandleB.size
		);

		cudaInterface.particlesKernel.cudaBufferB = PResultB.devicePointer;
		cudaInterface.particlesKernel.externalMemoryB = PResultB.externalMemory;


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
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
		});

		generateParticles();

		setupUB();
		setupFluidDepthTexture();

		generatePipeline();
		generateDepthPipeline();
		generateFinalPipeline();

		vkToCuda = engine.vulkanContext.createExportableSemaphore();
		cudaToVk = engine.vulkanContext.createExportableSemaphore();

		vkToCudaHandle = engine.vulkanContext.exportSemaphore(vkToCuda);
		cudaToVkHandle = engine.vulkanContext.exportSemaphore(cudaToVk);

		cudaInterface.vkToCuda = cudaInterface.importSemaphore( vkToCudaHandle );
		cudaInterface.cudaToVk = cudaInterface.importSemaphore( cudaToVkHandle );

		cudaInterface.initialize(engine.vulkanContext.physicalDeviceIF.uuid, simulationParams);

		cudaInterface.signalCudaFinished();
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
			//uniformData.projection[1][1] *= -1;
			uniformData.view = glm::lookAt(camera.position, camera.lookAt, camera.cameraUp);

			engine.resourceManager.copyDataIntoBuffer(UBHandle.allocationInfo.pMappedData, &uniformData, UBHandle.size);

			engine.barrier.transitionDepthReadOnlyToAttachment(commandBuffer, FDTextureHandle.image);

			//NEED TO WORK ON THIS - SOMETHING SIMILAR TO ABSTRACTCION RENDERINGATTACHMENT
			VkRenderingAttachmentInfo depthAttachmentInfo{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = FDTextureHandle.imageView,
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {.depthStencil = {1.0f,  0}}
			};

			VkRenderingInfo renderingInfo{
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.renderArea{.extent{.width = static_cast<uint32_t>(engine.windowManager.windowSize.x), .height = static_cast<uint32_t>(engine.windowManager.windowSize.y) }},
				.layerCount = 1,
				.pDepthAttachment = &depthAttachmentInfo
			};

			VkDescriptorSet descriptorSet =
				cudaInterface.particlesKernel.currentBufferIndex == 0
				? particles.PBDescriptorSet
				: particles.PBDescriptorSetB;

			VkDeviceSize offset1 = 0;
			vkCmdBeginRendering(commandBuffer, &renderingInfo);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, DepthPipeline);
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &particles.particleGeometry.VHandle.buffer, &offset1);
			vkCmdBindIndexBuffer(commandBuffer, particles.particleGeometry.IHandle.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, DepthPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, DepthPipelineLayout, 1, 1, &UBDescriptorSet, 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, particles.particleGeometry.indices.size(), particles.PCount(), 0, 0, 0);
			vkCmdEndRendering(commandBuffer);
			engine.barrier.transitionDepthAttachmentToReadOnly(commandBuffer, FDTextureHandle.image);

			engine.barrier.transitionImageFromUndefinedToColorAttachment(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			Engine::RenderingAttachment renderingAttachment = engine.setRenderingAttachment(engine.swapchainManager.imageViews[engine.swapchainManager.imageIndex]);
			vkCmdBeginRendering(commandBuffer, &renderingAttachment.renderingInfo);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, FinalPipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, FinalPipelineLayout, 0, 1, &FDTextureDescriptorSet, 0, nullptr);
			vkCmdDraw(commandBuffer, 3, 1, 0, 0);
			vkCmdEndRendering(commandBuffer);
			engine.barrier.transitionImageFromColorAttachmentToPresent(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);


			//VkDeviceSize offset = 0;
			//engine.barrier.transitionImageFromUndefinedToColorAttachment(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);
			//Engine::RenderingAttachment renderingAttachment = engine.setRenderingAttachment(engine.swapchainManager.imageViews[engine.swapchainManager.imageIndex] );
			//vkCmdBeginRendering(commandBuffer, &renderingAttachment.renderingInfo );
			//vkCmdBindVertexBuffers(commandBuffer, 0, 1, &particles.particleGeometry.VHandle.buffer, &offset);
			//vkCmdBindIndexBuffer(commandBuffer, particles.particleGeometry.IHandle.buffer, 0, VK_INDEX_TYPE_UINT32);
			//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 0, 1, &particles.PBDescriptorSet, 0, nullptr);
			//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 1, 1, &UBDescriptorSet, 0, nullptr);
			//vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);
			//vkCmdDrawIndexed(commandBuffer, particles.particleGeometry.indices.size(), particles.PCount(), 0, 0, 0);
			//vkCmdEndRendering(commandBuffer);
			//engine.barrier.transitionImageFromColorAttachmentToPresent(commandBuffer, engine.swapchainManager.images[engine.swapchainManager.imageIndex]);

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

			constexpr unsigned int MAX_SUBSTEPS = 1;

			cudaInterface.timeStep.addFrameTimeToAccumulator(deltaTime);
			for (unsigned int step = 0; step < MAX_SUBSTEPS; ++step){
				cudaInterface.simulate();
			}

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