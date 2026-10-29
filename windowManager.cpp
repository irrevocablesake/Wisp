
#include "WindowManager.h"

void WindowManager::setup(VulkanContext& vulkanContext_) {
	vulkanContext = &vulkanContext_;

	window = SDL_CreateWindow(
		windowName.c_str(),
		windowWidth,
		windowHeight,
		windowFlags
	);

	SDL_GetWindowSize(window, &windowSize.x, &windowSize.y);
	SDL_Vulkan_CreateSurface(window, vulkanContext->instanceIF.handle, nullptr, &surface);

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vulkanContext->physicalDeviceIF.handle, surface, &surfaceCapabilites);
}