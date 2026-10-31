#include "engine.h"

#include<random>

void Engine::configure(std::string name, uint32_t API_VERSION) {
	vulkanContext.setup(name, API_VERSION);
	windowManager.setup(vulkanContext);
	swapchainManager.setup(vulkanContext, windowManager);
	pipelineManager.setup(vulkanContext);
	resourceManager.setup(vulkanContext);
	shadersIF.setupSLANG();
	solidShapes.initialize(resourceManager);
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

Engine::RenderingAttachment Engine::setRenderingAttachment(VkImageView& imageView) {
	RenderingAttachment attachment{};
	attachment.attachmentInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = imageView,
		.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue{
			windowManager.clearColor
		}
	};

	attachment.renderingInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea{
			.extent{
				.width = static_cast<uint32_t>(windowManager.windowSize.x),
				.height = static_cast<uint32_t>(windowManager.windowSize.y)
			},
		},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachment.attachmentInfo
	};

	return attachment;
}