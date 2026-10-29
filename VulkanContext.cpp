#include "VulkanContext.h"

#define VOLK_IMPLEMENTATION
#include<volk/volk.h>

#define VMA_IMPLEMENTATION
#include<vma/vk_mem_alloc.h>

void VulkanContext::setup( std::string name, uint32_t API_VERSION_PARAM ) {

	applicationName = name;
	API_VERSION = API_VERSION_PARAM;

	setupLibraries();
	createInstance();
	pickPhysicalDevice();
	pickQueue();
	setupLogicalDevice();
	setupVMA();
	setupCommandPool();
	setupFrames();
}

void VulkanContext::setupLibraries() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Vulkan_LoadLibrary(NULL);
	volkInitialize();
}

void VulkanContext::createInstance() {
	VkApplicationInfo applicationInfo{
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = applicationName.c_str(),
			.apiVersion = API_VERSION
	};

	extensionsIF.instanceIF.extensions = SDL_Vulkan_GetInstanceExtensions(&extensionsIF.instanceIF.count);

	VkInstanceCreateInfo instanceCI{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &applicationInfo,
		.enabledExtensionCount = extensionsIF.instanceIF.count,
		.ppEnabledExtensionNames = extensionsIF.instanceIF.extensions
	};

	vkCreateInstance(&instanceCI, nullptr, &instanceIF.handle);
	volkLoadInstance( instanceIF.handle );
}

void VulkanContext::pickPhysicalDevice() {
	vkEnumeratePhysicalDevices( instanceIF.handle, &physicalDeviceIF.count, nullptr);

	physicalDeviceIF.devices.resize( physicalDeviceIF.count );
	
	vkEnumeratePhysicalDevices( instanceIF.handle, &physicalDeviceIF.count, physicalDeviceIF.devices.data());

	for (uint32_t i = 0; i < physicalDeviceIF.count; i++) {
		VkPhysicalDeviceProperties2 prop{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
		};

		vkGetPhysicalDeviceProperties2(physicalDeviceIF.devices[i], &prop);
		physicalDeviceIF.properties.push_back(prop);
	}
		bool found = false;

	for (uint32_t i = 0; i < physicalDeviceIF.count; i++) {

			bool cond1 = physicalDeviceIF.properties[i].properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
			if (cond1) {
				physicalDeviceIF.index = i;
				physicalDeviceIF.handle = physicalDeviceIF.devices[i];

				found = true;
			}

		
		if (found) {
			break;
		}
	}
}

void VulkanContext::pickQueue() {
	queueIF.properties.resize( physicalDeviceIF.count);

	for (uint32_t i = 0; i < physicalDeviceIF.count; i++) {
		uint32_t queueFamiliesCount{};
		vkGetPhysicalDeviceQueueFamilyProperties2( physicalDeviceIF.devices[i], &queueFamiliesCount, nullptr);
		queueIF.properties[i].resize(queueFamiliesCount);

		for (uint32_t j = 0; j < queueFamiliesCount; j++) {
			queueIF.properties[i][j].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
		}

		vkGetPhysicalDeviceQueueFamilyProperties2( physicalDeviceIF.devices[i], &queueFamiliesCount, queueIF.properties[i].data());

	}

	bool found = false;

	for (uint32_t i = 0; i < physicalDeviceIF.count; i++) {
		bool found = false;

		for (uint32_t j = 0; j < queueIF.properties[i].size(); j++) {
			bool cond2 = queueIF.properties[i][j].queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT;
			if (cond2) {
				queueIF.family = j;
				
				found = true;
				break;
			}

		}
		if (found) {
			break;
		}
	}
}

void VulkanContext::setupLogicalDevice() {
	VkDeviceQueueCreateInfo queueCI{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = queueIF.family,
		.queueCount = 1,
		.pQueuePriorities = &queueIF.priority
	};

	VkPhysicalDeviceVulkan11Features enabledVk11Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.shaderDrawParameters = VK_TRUE
	};

	VkPhysicalDeviceVulkan12Features enabledVk12Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES	,
		.pNext = &enabledVk11Features,
		.descriptorIndexing = true,
		.shaderSampledImageArrayNonUniformIndexing = true,
		.descriptorBindingVariableDescriptorCount = true,
		.runtimeDescriptorArray = true,
		.bufferDeviceAddress = true
	};

	VkPhysicalDeviceVulkan13Features enabledVk13Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &enabledVk12Features,
		.synchronization2 = true,
		.dynamicRendering = true
	};

	VkDeviceCreateInfo deviceCI{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &enabledVk13Features,
		.queueCreateInfoCount = static_cast<uint32_t>(1),
		.pQueueCreateInfos = &queueCI,
		.enabledExtensionCount = static_cast<uint32_t>(extensionsIF.deviceIF.extensions.size()),
		.ppEnabledExtensionNames = extensionsIF.deviceIF.extensions.data(),
	};

	vkCreateDevice( physicalDeviceIF.handle, &deviceCI, nullptr, &logicalDeviceIF.handle);
	volkLoadDevice( logicalDeviceIF.handle );

	vkGetDeviceQueue(logicalDeviceIF.handle, queueIF.family, 0, &queueIF.handle );
}

void VulkanContext::setupVMA() {
	VmaVulkanFunctions vulkanFns{
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr
	};

	VmaAllocatorCreateInfo allocatorCI{
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = physicalDeviceIF.handle,
		.device = logicalDeviceIF.handle,
		.pVulkanFunctions = &vulkanFns,
		.instance = instanceIF.handle
	};

	vmaCreateAllocator(&allocatorCI, &allocatorIF.handle);
}

void VulkanContext::setupCommandPool() {
	VkCommandPoolCreateInfo commandPoolCreateInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = queueIF.family
	};

	vkCreateCommandPool(logicalDeviceIF.handle, &commandPoolCreateInfo, nullptr, &commandPool);
}

void VulkanContext::setupFrames() {
	frames.resize(MAX_FRAMES_IN_FLIGHT);
	VkSemaphoreCreateInfo semaphoreCI{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	VkFenceCreateInfo fenceCI{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	for (Frame& frame : frames) {

		vkCreateFence(logicalDeviceIF.handle, &fenceCI, nullptr, &frame.frameFence);
		vkCreateSemaphore(logicalDeviceIF.handle, &semaphoreCI, nullptr, &frame.imageAvailable);

		VkCommandBufferAllocateInfo commandBufferAllocateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT
		};

		vkAllocateCommandBuffers(logicalDeviceIF.handle, &commandBufferAllocateCreateInfo, &frame.commandBuffer);
	}
}