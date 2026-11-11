#include "resourceManager.h"

#include <stdexcept>
#include<cstring>

void ResourceManager::copyDataIntoBuffer(void* destination, const void* source, size_t size) {
	memcpy(destination, source, size);
}

//NOTE: POSSIBLY? A BETTER WAY TO DO THE BOOL PARAMETRS?
void ResourceManager::setupBuffer(BufferHandle& handle, VkBufferUsageFlags flags, bool isHostMappable, bool isDedicated, bool isInterOp ) {
	VkExternalMemoryBufferCreateInfo externalBufferInfo{
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
	};

	VkBufferCreateInfo bufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = isInterOp ? &externalBufferInfo : nullptr,
		.size = handle.size,
		.usage = flags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
	};

	VmaAllocationCreateInfo allocationCreateInfo{
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	if (isInterOp) {
		allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

		VkExportMemoryAllocateInfo exportInfo{
			.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
			.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
		};

		vmaCreateDedicatedBuffer(
			vulkanContext->allocatorIF.handle,
			&bufferCreateInfo,
			&allocationCreateInfo,
			&exportInfo,
			&handle.buffer,
			&handle.allocation,
			&handle.allocationInfo
		);
	}
	else if ( isDedicated ) {
		allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

		vmaCreateBuffer(
			vulkanContext->allocatorIF.handle,
			&bufferCreateInfo,
			&allocationCreateInfo,
			&handle.buffer,
			&handle.allocation,
			&handle.allocationInfo
		);
	}
	else if( isHostMappable ){
		allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocationCreateInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
			VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;

		vmaCreateBuffer(
			vulkanContext->allocatorIF.handle,
			&bufferCreateInfo,
			&allocationCreateInfo,
			&handle.buffer,
			&handle.allocation,
			&handle.allocationInfo
		);
	}

	VkBufferDeviceAddressInfo bufferBDAInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = handle.buffer
	};

	handle.deviceAddress = vkGetBufferDeviceAddress(
		vulkanContext->logicalDeviceIF.handle,
		&bufferBDAInfo
	);
}


HANDLE ResourceManager::exportBuffer(BufferHandle& handle) {

	HANDLE sharedHandle = nullptr;

	VkResult result = vmaGetMemoryWin32Handle2(
		vulkanContext->allocatorIF.handle,
		handle.allocation,
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
		nullptr,
		&sharedHandle
	);

	if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to export Vulkan memory!");
	}

	return sharedHandle;
}

//WE REALLY NEED TO WRITE A SCENE GRAPH / TRANSITION BARRIERS AUTOMATIC
void ResourceManager::generateTexture(TextureHandle& handle, VkImageUsageFlags usage, VkImageAspectFlags aspectMask) {

	//Example usage: VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT
	VkImageCreateInfo textureImageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = handle.imageFormat,
		.extent{
			.width = static_cast<uint32_t>( handle.width ),
			.height = static_cast<uint32_t>( handle.height ),
			.depth = 1
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VmaAllocationCreateInfo textureImageAllocationCreateInfo{
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	//add validation later
	vmaCreateImage(vulkanContext->allocatorIF.handle, &textureImageCreateInfo, &textureImageAllocationCreateInfo, &handle.image, &handle.allocation, nullptr);

	VkCommandBuffer commandBuffer = barrier.beginOneTimeCommand(vulkanContext->logicalDeviceIF.handle, vulkanContext->commandPool);
	//depends upon the image type
	if (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
		barrier.transitionImageFromUndefinedToReadOnly(commandBuffer, handle.image);
	}
	else {
		barrier.transitionImageFromUndefinedToGeneral(commandBuffer, handle.image);
	}
	barrier.endOneTimeCommand(vulkanContext->logicalDeviceIF.handle, commandBuffer, vulkanContext->queueIF.handle);

	//aspect mask can't just be color bit : VK_IMAGE_ASPECT_COLOR_BIT
	VkImageViewCreateInfo textureImageViewCreateInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = handle.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = handle.imageFormat,
		.subresourceRange = {
			.aspectMask = aspectMask,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	vkCreateImageView(vulkanContext->logicalDeviceIF.handle, &textureImageViewCreateInfo, nullptr, &handle.imageView);

	VkSamplerCreateInfo samplerCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = handle.imageFilter,
		.minFilter = handle.imageFilter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.minLod = 0.0f,
		.maxLod = 0.0f
	};

	vkCreateSampler(vulkanContext->logicalDeviceIF.handle, &samplerCreateInfo, nullptr, &handle.sampler);

	handle.descriptor = {
		.sampler = handle.sampler,
		.imageView = handle.imageView,
		.imageLayout = handle.imageLayout
	};
}

void ResourceManager::setup(VulkanContext& vulkanContext_ ) {
	vulkanContext = &vulkanContext_;
}

void ResourceManager::setupDescriptorPool(std::map< VkDescriptorType, uint32_t > info) {
	std::vector< VkDescriptorPoolSize > poolSizes;
	uint32_t totalSet = 0;

	for (auto& pair : info) {
		poolSizes.push_back({
			.type = pair.first,
			.descriptorCount = pair.second
			});

		totalSet = totalSet + pair.second;
	}

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = totalSet,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	vkCreateDescriptorPool(vulkanContext->logicalDeviceIF.handle, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
}

VkDescriptorSetLayout ResourceManager::createDescriptorSetLayout(std::vector<VkDescriptorSetLayoutBinding> layoutBindings) {
	VkDescriptorSetLayout layout;

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data()
	};

	vkCreateDescriptorSetLayout(vulkanContext->logicalDeviceIF.handle, &descriptorSetLayoutCI, nullptr, &layout);

	return layout;
}

std::vector< VkDescriptorSet > ResourceManager::allocateDescriptorSets(std::vector< VkDescriptorSetLayout > layouts) {
	std::vector<VkDescriptorSet> descriptorSet(layouts.size());

	VkDescriptorSetAllocateInfo descriptorAllocateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	vkAllocateDescriptorSets(vulkanContext->logicalDeviceIF.handle, &descriptorAllocateInfo, descriptorSet.data());

	return descriptorSet;
}

//NOTE: It is better we use a memory barrier, to ensure that the copied data is available before we start working on it. TODO
void ResourceManager::uploadData( BufferHandle& bufferHandle, const void* data, size_t size) {
	VkBufferCreateInfo stagingInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
	};

	VmaAllocationCreateInfo stagingAllocInfo{
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	VkBuffer stagingBuffer;
	VmaAllocation stagingAllocation;
	VmaAllocationInfo stagingAllocInfoResult;

	vmaCreateBuffer(
		vulkanContext->allocatorIF.handle,
		&stagingInfo,
		&stagingAllocInfo,
		&stagingBuffer,
		&stagingAllocation,
		&stagingAllocInfoResult
	);

	memcpy(stagingAllocInfoResult.pMappedData, data, size);

	VkCommandBuffer commandBuffer = barrier.beginOneTimeCommand( vulkanContext->logicalDeviceIF.handle, vulkanContext->commandPool );
	VkBufferCopy copyRegion{ 
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, bufferHandle.buffer, 1, &copyRegion);
	barrier.endOneTimeCommand( vulkanContext->logicalDeviceIF.handle, commandBuffer, vulkanContext->queueIF.handle );

	vmaDestroyBuffer(vulkanContext->allocatorIF.handle, stagingBuffer, stagingAllocation);
}

VkDescriptorSet ResourceManager::allocateDescriptorSets( VkDescriptorSetLayout layout ){
	VkDescriptorSet descriptorSet;

	VkDescriptorSetAllocateInfo descriptorAllocateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &layout
	};

	vkAllocateDescriptorSets(vulkanContext->logicalDeviceIF.handle, &descriptorAllocateInfo, &descriptorSet );

	return descriptorSet;
}

void ResourceManager::updateDescriptorSet(std::vector< VkWriteDescriptorSet > descriptorSets) {
	vkUpdateDescriptorSets(vulkanContext->logicalDeviceIF.handle, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
}