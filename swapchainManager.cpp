#include "SwapchainManager.h"

void SwapchainManager::setup(VulkanContext& vulkanContext_, WindowManager& windowManager_) {
	windowManager = &windowManager_;
	vulkanContext = &vulkanContext_;

	swapchainCI = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = windowManager->surface,
		.minImageCount = windowManager->surfaceCapabilites.minImageCount,
		.imageFormat = imageFormat,
		.imageColorSpace = imageColorSpace,
		.imageExtent = {
			windowManager->surfaceCapabilites.currentExtent.width,
			windowManager->surfaceCapabilites.currentExtent.height
		},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR
	};

	vkCreateSwapchainKHR(vulkanContext->logicalDeviceIF.handle, &swapchainCI, nullptr, &swapchain);

	vkGetSwapchainImagesKHR(vulkanContext->logicalDeviceIF.handle, swapchain, &imageCount, nullptr);
	images.resize(imageCount);
	vkGetSwapchainImagesKHR(vulkanContext->logicalDeviceIF.handle, swapchain, &imageCount, images.data());
	imageViews.resize(imageCount);

	for (uint32_t i = 0; i < imageCount; i++) {
		VkImageViewCreateInfo imageViewCI{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = imageFormat,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1
			}
		};

		vkCreateImageView(vulkanContext->logicalDeviceIF.handle, &imageViewCI, nullptr, &imageViews[i]);
	}

	renderSemaphores.resize( imageCount );

	VkSemaphoreCreateInfo semaphoreCI{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	for (VkSemaphore& semaphore : renderSemaphores) {
		vkCreateSemaphore( vulkanContext->logicalDeviceIF.handle, &semaphoreCI, nullptr, &semaphore);
	}
}

void SwapchainManager::recreateSwapchain() {
	updateSwapchain = false;
	vkDeviceWaitIdle(vulkanContext->logicalDeviceIF.handle);
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vulkanContext->physicalDeviceIF.handle, windowManager->surface, &windowManager->surfaceCapabilites);
	swapchainCI.oldSwapchain = swapchain;
	swapchainCI.imageExtent = { .width = static_cast<uint32_t>(windowManager->windowSize.x), .height = static_cast<uint32_t>(windowManager->windowSize.y) };
	vkCreateSwapchainKHR(vulkanContext->logicalDeviceIF.handle, &swapchainCI, nullptr, &swapchain);
	for (uint32_t i = 0; i < imageCount; i++) {
		vkDestroyImageView(vulkanContext->logicalDeviceIF.handle, imageViews[i], nullptr);
	}
	vkGetSwapchainImagesKHR(vulkanContext->logicalDeviceIF.handle, swapchain, &imageCount, nullptr);
	images.resize(imageCount);
	vkGetSwapchainImagesKHR(vulkanContext->logicalDeviceIF.handle, swapchain, &imageCount, images.data());
	imageViews.resize(imageCount);
	for (uint32_t i = 0; i < imageCount; i++) {
		VkImageViewCreateInfo viewCI{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = imageFormat,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}
		};
		vkCreateImageView(vulkanContext->logicalDeviceIF.handle, &viewCI, nullptr, &imageViews[i]);
	}
	vkDestroySwapchainKHR(vulkanContext->logicalDeviceIF.handle, swapchainCI.oldSwapchain, nullptr);
}

void SwapchainManager::shouldUpdate() {
	updateSwapchain = !updateSwapchain;
}