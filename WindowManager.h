#pragma once

#include<volk/volk.h>

#include<SDL3/SDL.h>
#include<SDL3/SDL_vulkan.h>

#include<glm/glm.hpp>
#include "VulkanContext.h"

#include<string>

class WindowManager {

public:
	VulkanContext* vulkanContext;

	SDL_Window* window;
	VkSurfaceKHR surface;
	VkSurfaceCapabilitiesKHR surfaceCapabilites;

		std::string windowName = "Firefly";
		int windowWidth = 400;
		int windowHeight = 400;

		SDL_WindowFlags windowFlags{
				SDL_WINDOW_VULKAN | SDL_WINDOW_FULLSCREEN
		};

		glm::ivec2 windowSize{};

		VkClearColorValue clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };
	
public:
	void setup(VulkanContext& vulkanContext);
};