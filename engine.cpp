#include "engine.h"

#include<random>

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

struct Engine::RNDGenerator {
	private:
		std::random_device rd;
		std::mt19937 gen{ rd() };

	public:
		float getRange(float min, float max) {
			std::uniform_real_distribution<float> range{ min, max };
			return range(gen);
		}
	
} rndGenerator;

float Engine::generateRandomNumber(float min, float max) {
	return rndGenerator.getRange(min, max);
}

VulkanContext::Frame& Engine::getCurrentFrame() {
	return vulkanContext.getCurrentFrame();
}

ResourceManager::TextureHandle& Engine::generateTextureHandle() {
	ResourceManager::TextureHandle instance;
	return instance;
}

VkShaderModule Engine::compileShader(std::string moduleName, std::string path) {
	return shadersIF.loadAndCompileShaders(vulkanContext.logicalDeviceIF.handle, moduleName.c_str(), path.c_str() );
}

void Engine::generateTexture(ResourceManager::TextureHandle& handle) {
	resourceManager.generateTexture(handle);
}