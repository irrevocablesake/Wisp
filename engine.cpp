#include "engine.h"

void Engine::configure(std::string name, uint32_t API_VERSION) {
	vulkanContext.setup(name, API_VERSION);
	windowManager.setup(vulkanContext);
	swapchainManager.setup(vulkanContext, windowManager);
	pipelineManager.setup(vulkanContext);
	resourceManager.setup(vulkanContext);
	shadersIF.setupSLANG();
}

void Engine::updateSwapchain() {
	if (!swapchainManager.updateSwapchain) {
		swapchainManager.shouldUpdate();
	}
	else {
		swapchainManager.recreateSwapchain();
	}
}

VulkanContext::Frame& Engine::getCurrentFrame() {
	return vulkanContext.getCurrentFrame();
}

ResourceManager::TextureHandle& Engine::generateTextureHandle() {
	ResourceManager::TextureHandle instance;
	return instance;
}

VkShaderModule Engine::compileShader(std::string moduleName, std::string path) {
	return shadersIF.loadAndCompileShaders(vulkanContext.logicalDeviceIF.handle, "rayTracerShaderModule", "assets/shaders/rayTracerShader.slang");
}

void Engine::generateTexture(ResourceManager::TextureHandle& handle) {
	resourceManager.generateTexture(handle);
}