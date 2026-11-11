#pragma once

#include "VulkanContext.h"

#define KHRONOS_STATIC
#include <ktx.h>
#include <ktxvulkan.h>

#include "barriers.h"

#include <map>
#include <vector>
#include <Windows.h>

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
			VkImageLayout imageLayout; // What layout the shader should expect the texture to be in

			uint32_t width;
			uint32_t height;
		};

		VkDescriptorPool descriptorPool;

	public:
		void setupBuffer(BufferHandle& handle, VkBufferUsageFlags flags, bool interop = false );
		void copyDataIntoBuffer( void* destination, const void* source, size_t size );

		void generateTexture( TextureHandle& handle, VkImageUsageFlags usage, VkImageAspectFlags aspectMask );

		void setup( VulkanContext& vulkanContext_ );

		void setupDescriptorPool(std::map< VkDescriptorType, uint32_t > info);
		VkDescriptorSetLayout createDescriptorSetLayout(std::vector<VkDescriptorSetLayoutBinding> layout);
		std::vector< VkDescriptorSet >  allocateDescriptorSets(std::vector< VkDescriptorSetLayout > layouts);
		void updateDescriptorSet(std::vector< VkWriteDescriptorSet > descriptorSets);

		VkDescriptorSet allocateDescriptorSets(VkDescriptorSetLayout layout );

		HANDLE exportBuffer(BufferHandle& handle);
};