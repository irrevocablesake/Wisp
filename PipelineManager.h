#pragma once

#include<volk/volk.h>
#include "VulkanContext.h"

class PipelineManager {
public:
	VulkanContext* vulkanContext;
	
	void createComputePipeline(VkPipeline& pipeline, VkPipelineShaderStageCreateInfo& shaderStages, VkPipelineLayout& pipelineLayout);
	void createGraphicsPipeline(VkPipeline& pipeline, VkPipelineLayout& pipelineLayout, const std::vector<VkDescriptorSetLayout>& layout, VkFormat format, std::vector< VkPipelineShaderStageCreateInfo >& shaderStages, VkPipelineVertexInputStateCreateInfo& vertexInputState, uint32_t pushConstantSize, VkPrimitiveTopology topology, bool isDepth, bool depthTestEnable );

	void setup(VulkanContext& vulkanContext);
};