#include "barriers.h"

void Barriers::transitionImageFromUndefinedToGeneral(VkCommandBuffer& commandBuffer, VkImage& image) {
	VkImageMemoryBarrier2 barrierLayoutTransition{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.image = image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1
			}
	};

	VkDependencyInfo barrierDependencyInfo{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrierLayoutTransition
	};

	vkCmdPipelineBarrier2(commandBuffer, &barrierDependencyInfo);
}

void Barriers::transitionImageFromGeneralToTransferSrc(VkCommandBuffer& commandBuffer, VkImage& image) {
	VkImageMemoryBarrier2 computeToTransfer{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
	.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
	.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
	.newLayout = VK_IMAGE_LAYOUT_GENERAL,
	.image = image,
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	VkDependencyInfo computeToTransferDependency{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &computeToTransfer
	};

	vkCmdPipelineBarrier2(
		commandBuffer,
		&computeToTransferDependency
	);
}

void Barriers::transitionImageFromTransferSrcToGeneral(VkCommandBuffer& commandBuffer, VkImage& image) {
	VkImageMemoryBarrier2 computeToTransfer{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
	.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
	.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
	.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
	.newLayout = VK_IMAGE_LAYOUT_GENERAL,
	.image = image,
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	VkDependencyInfo computeToTransferDependency{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &computeToTransfer
	};

	vkCmdPipelineBarrier2(
		commandBuffer,
		&computeToTransferDependency
	);
}

void Barriers::transitionImageTransferDstToPresent(VkCommandBuffer& commandBuffer, VkImage& image)
{
	VkImageMemoryBarrier2 barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.dstAccessMask = VK_ACCESS_2_NONE,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.image = image,
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	VkDependencyInfo dependency{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};

	vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void Barriers::transitionImageFromUndefinedToTransferDst(VkCommandBuffer& commandBuffer, VkImage& image) {
	VkImageMemoryBarrier2 barrier{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
	.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
	.srcAccessMask = VK_ACCESS_2_NONE,
	.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
	.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	.image = image,
	.subresourceRange{
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1
	}
	};

	VkDependencyInfo dependency{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};

	vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

VkCommandBuffer Barriers::beginOneTimeCommand( VkDevice &device, VkCommandPool &commandPool ) {
	VkFenceCreateInfo fenceOneTimeCreateInfo{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
	};
	
	vkCreateFence( device, &fenceOneTimeCreateInfo, nullptr, &fenceOneTime);

	VkCommandBuffer commandBufferOneTime;
	VkCommandBufferAllocateInfo commandBufferOneTimeAllocationInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.commandBufferCount = 1
	};

	vkAllocateCommandBuffers( device, &commandBufferOneTimeAllocationInfo, &commandBufferOneTime);

	VkCommandBufferBeginInfo commandBufferOneTimeBeginInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	vkBeginCommandBuffer(commandBufferOneTime, &commandBufferOneTimeBeginInfo);

	return commandBufferOneTime;
}

void Barriers::endOneTimeCommand( VkDevice device, VkCommandBuffer &commandBuffer, VkQueue queue ) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo oneTimeSubmitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};

	vkQueueSubmit( queue, 1, &oneTimeSubmitInfo, fenceOneTime);
	vkWaitForFences( device, 1, &fenceOneTime, VK_TRUE, UINT64_MAX);
}