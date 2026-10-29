#pragma once

#include<volk/volk.h>

#include "WindowManager.h"
#include "VulkanContext.h"
#include<vector>

class SwapchainManager {
public:
	VkSwapchainCreateInfoKHR swapchainCI{};

	const VkFormat imageFormat{ VK_FORMAT_B8G8R8A8_SRGB };
	const VkColorSpaceKHR imageColorSpace{ VK_COLORSPACE_SRGB_NONLINEAR_KHR };
	VkSwapchainKHR swapchain;
	uint32_t imageCount{ 0 };
	std::vector<VkImage> images;
	std::vector< VkImageView> imageViews;

	std::vector< VkSemaphore > renderSemaphores;

	uint32_t imageIndex{ 0 };

	bool updateSwapchain{ false };

	WindowManager* windowManager;
	VulkanContext* vulkanContext;

	void setup(VulkanContext& vulkanContext, WindowManager& windowManager_ );
	void recreateSwapchain();
	void shouldUpdate();
};