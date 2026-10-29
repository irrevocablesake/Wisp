#pragma once

#include "VulkanContext.h"
#include "WindowManager.h"
#include "SwapchainManager.h"
#include "PipelineManager.h"
#include "resourceManager.h"
#include "shaders.h"
#include "barriers.h"

class Engine {
public:
	VulkanContext vulkanContext;
	WindowManager windowManager;
	SwapchainManager swapchainManager;
	PipelineManager pipelineManager;
	ResourceManager resourceManager;
	Shaders shadersIF;
	Barriers barrier;

	using TextureHandle = ResourceManager::TextureHandle;
	using BufferHandle = ResourceManager::BufferHandle;

public:
	void configure(std::string name, uint32_t API_VERSION );	
	void updateSwapchain();
	VulkanContext::Frame& getCurrentFrame();
	ResourceManager::TextureHandle& generateTextureHandle();
	VkShaderModule compileShader(std::string moduleName, std::string path);
	void generateTexture(ResourceManager::TextureHandle& handle);
};