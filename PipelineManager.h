#pragma once

#include<volk/volk.h>
#include "VulkanContext.h"

class PipelineManager {
public:
	VulkanContext* vulkanContext;
	
	void createComputePipeline(VkPipeline& pipeline, VkPipelineShaderStageCreateInfo& shaderStages, VkPipelineLayout& pipelineLayout);
	void setup(VulkanContext& vulkanContext);
};