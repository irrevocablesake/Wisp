#include "resourceManager.h"

#include<cstring>

void ResourceManager::copyDataIntoBuffer(void* destination, const void* source, size_t size) {
	memcpy(destination, source, size);
}

void ResourceManager::setupBuffer(BufferHandle& handle, VkBufferUsageFlags flags ) {
	VkBufferCreateInfo uniformBufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = handle.size,
		.usage = flags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
	};

	VmaAllocationCreateInfo uniformBufferAllocationCreateInfo{
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	//validation later
	vmaCreateBuffer(vulkanContext->allocatorIF.handle, &uniformBufferCreateInfo, &uniformBufferAllocationCreateInfo, &handle.buffer, &handle.allocation, &handle.allocationInfo);

	VkBufferDeviceAddressInfo uniformBDAInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = handle.buffer
	};

	handle.deviceAddress = vkGetBufferDeviceAddress(vulkanContext->logicalDeviceIF.handle, &uniformBDAInfo);
}

void ResourceManager::generateTexture( TextureHandle& handle) {
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
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VmaAllocationCreateInfo textureImageAllocationCreateInfo{
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	//add validation later
	vmaCreateImage(vulkanContext->allocatorIF.handle, &textureImageCreateInfo, &textureImageAllocationCreateInfo, &handle.image, &handle.allocation, nullptr);

	VkCommandBuffer commandBuffer = barrier.beginOneTimeCommand( vulkanContext->logicalDeviceIF.handle,vulkanContext->commandPool);
	barrier.transitionImageFromUndefinedToGeneral(commandBuffer, handle.image);
	barrier.endOneTimeCommand(vulkanContext->logicalDeviceIF.handle, commandBuffer, vulkanContext->queueIF.handle);

	VkImageViewCreateInfo textureImageViewCreateInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = handle.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = handle.imageFormat,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	vkCreateImageView(vulkanContext->logicalDeviceIF.handle, &textureImageViewCreateInfo, nullptr, &handle.imageView);

	handle.descriptor = {
		.sampler = VK_NULL_HANDLE,
		.imageView = handle.imageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL
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