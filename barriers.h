#pragma once

#include<volk/volk.h>

class Barriers {
	public:
		void transitionImageFromUndefinedToGeneral(VkCommandBuffer& commandBuffer, VkImage& image);
		void transitionImageFromGeneralToTransferSrc(VkCommandBuffer& commandBuffer, VkImage& image);
		void transitionImageFromTransferSrcToGeneral(VkCommandBuffer& commandBuffer, VkImage& image);
		void transitionImageTransferDstToPresent(VkCommandBuffer& commandBuffer, VkImage& image);
		void transitionImageFromUndefinedToTransferDst(VkCommandBuffer& commandBuffer, VkImage& image);

		VkCommandBuffer beginOneTimeCommand( VkDevice& device, VkCommandPool& commandPool );
		void endOneTimeCommand(VkDevice device, VkCommandBuffer& commandBuffer, VkQueue queue );

		VkFence fenceOneTime;
};