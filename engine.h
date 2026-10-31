#pragma once

#include "VulkanContext.h"
#include "WindowManager.h"
#include "SwapchainManager.h"
#include "PipelineManager.h"
#include "resourceManager.h"
#include "shaders.h"
#include "barriers.h"
#include "SolidShapes.h"

class Engine {
public:
	VulkanContext vulkanContext;
	WindowManager windowManager;
	SwapchainManager swapchainManager;
	PipelineManager pipelineManager;
	ResourceManager resourceManager;
	Shaders shadersIF;
	Barriers barrier;
	SolidShapes solidShapes;

	struct RNDGenerator;

	using TextureHandle = ResourceManager::TextureHandle;
	using BufferHandle = ResourceManager::BufferHandle;
	using Geometry = SolidShapes::Geometry;
	using Vertex = SolidShapes::Vertex;

public:
	void configure(std::string name, uint32_t API_VERSION );	
	void updateSwapchain();
	VulkanContext::Frame& getCurrentFrame();
	ResourceManager::TextureHandle& generateTextureHandle();
	VkShaderModule compileShader(std::string moduleName, std::string path);
	void generateTexture(ResourceManager::TextureHandle& handle);

	float generateRandomNumber( float min, float max );

	struct RenderingAttachment {
		VkRenderingAttachmentInfo attachmentInfo;
		VkRenderingInfo renderingInfo;
	};

	RenderingAttachment setRenderingAttachment(VkImageView& imageView);
};