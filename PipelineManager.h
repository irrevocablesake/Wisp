#pragma once

#include<volk/volk.h>
#include "VulkanContext.h"

class PipelineManager {
public:
	VulkanContext* vulkanContext;
	
	void createComputePipeline(VkPipeline& pipeline, VkPipelineShaderStageCreateInfo& shaderStages, VkPipelineLayout& pipelineLayout);
	void createGraphicsPipeline(VkPipeline& pipeline, VkPipelineLayout& pipelineLayout, const std::vector<VkDescriptorSetLayout>& layout, VkFormat format, std::vector< VkPipelineShaderStageCreateInfo >& shaderStages, uint32_t pushConstantSize = 0);

	void setup(VulkanContext& vulkanContext);
};