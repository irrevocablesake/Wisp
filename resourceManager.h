#pragma once

#include<volk/volk.h>
#include<vma/vk_mem_alloc.h>

#define KHRONOS_STATIC 
#include<ktx.h>
#include<ktxvulkan.h>

#include "barriers.h"

#include "VulkanContext.h"

#include<map>
#include<vector>

class ResourceManager {
	Barriers barrier;
	VulkanContext* vulkanContext;

	public:
		struct BufferHandle {

			size_t size;

			VkBuffer buffer{ VK_NULL_HANDLE };

			VmaAllocation allocation{ VK_NULL_HANDLE };
			VmaAllocationInfo allocationInfo{};

			VkDeviceAddress deviceAddress{};
			void* mapped{ nullptr };
		};

		struct TextureHandle {
			VkImage image;
			VkImageView imageView;
			VmaAllocation allocation;
			VkSampler sampler;

			VkDescriptorImageInfo descriptor;

			VkFormat imageFormat;
			VkFilter imageFilter;

			uint32_t width;
			uint32_t height;
		};

		VkDescriptorPool descriptorPool;

	public:
		void setupBuffer(BufferHandle& handle, VkBufferUsageFlags flags);
		void copyDataIntoBuffer( void* destination, const void* source, size_t size );

		void generateTexture(TextureHandle& handle );

		void setup( VulkanContext& vulkanContext_ );

		void setupDescriptorPool(std::map< VkDescriptorType, uint32_t > info);
		VkDescriptorSetLayout createDescriptorSetLayout(std::vector<VkDescriptorSetLayoutBinding> layout);
		std::vector< VkDescriptorSet >  allocateDescriptorSets(std::vector< VkDescriptorSetLayout > layouts);
		void updateDescriptorSet(std::vector< VkWriteDescriptorSet > descriptorSets);

		VkDescriptorSet allocateDescriptorSets(VkDescriptorSetLayout layout );
};