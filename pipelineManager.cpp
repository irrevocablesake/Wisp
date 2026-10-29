#include "PipelineManager.h"

void PipelineManager::setup(VulkanContext& vulkanContext_) {
	vulkanContext = &vulkanContext_;

}

void PipelineManager::createComputePipeline(VkPipeline& pipeline, VkPipelineShaderStageCreateInfo& shaderStages, VkPipelineLayout& pipelineLayout) {
	VkComputePipelineCreateInfo pipelineCI{
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = shaderStages,
		.layout = pipelineLayout
	};

	vkCreateComputePipelines(vulkanContext->logicalDeviceIF.handle, nullptr, 1, &pipelineCI, nullptr, &pipeline);
}