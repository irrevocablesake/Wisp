#pragma once

#include<volk/volk.h>
#include<vma/vk_mem_alloc.h>

#include<SDL3/SDL.h>
#include<SDL3/SDL_vulkan.h>

#include<vector>
#include<array>
#include<string>

class VulkanContext {
	public:
		std::string applicationName{};
		uint32_t API_VERSION{};

		struct Instance {
			VkInstance handle{};
		} instanceIF;
	
		struct PhysicalDevice {
			VkPhysicalDevice handle;
			uint32_t count;
			uint32_t index;
			std::vector<VkPhysicalDevice> devices;
			std::vector<VkPhysicalDeviceProperties2> properties;
		} physicalDeviceIF;

		struct LogicalDevice {
			VkDevice handle;
		} logicalDeviceIF;

		struct Queue {
			VkQueue handle;
			uint32_t family;
			std::vector< std::vector<VkQueueFamilyProperties2> > properties{};

			float priority = 1.0;
		} queueIF;

		struct Allocator {
			VmaAllocator handle;
		} allocatorIF;

		struct Extensions {
			struct Instance {
				uint32_t count;
				char const* const* extensions;
			} instanceIF;

			struct Device {
				const std::vector< const char* > extensions{
					VK_KHR_SWAPCHAIN_EXTENSION_NAME
				};
			} deviceIF;
		} extensionsIF;

		VkCommandPool commandPool;

		struct Frame {
			VkFence frameFence;

			VkSemaphore imageAvailable;

			VkCommandBuffer commandBuffer;
		};

		const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
		uint32_t currentFrame = 0;
		std::vector<Frame> frames;
		std::vector<VkSemaphore> renderFinished;

	public:
		void setupLibraries();
		void createInstance();
		void pickPhysicalDevice();
		void pickQueue();
		void setupLogicalDevice();
		void setupVMA();
		void setupCommandPool();
		void setupFrames();
		void setup( std::string name, uint32_t API_VERSION_PARAM );

		Frame& getCurrentFrame() {
			return frames[currentFrame];
		}
};