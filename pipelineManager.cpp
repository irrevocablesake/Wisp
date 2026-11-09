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

void PipelineManager::createGraphicsPipeline(VkPipeline& pipeline, VkPipelineLayout& pipelineLayout, const std::vector<VkDescriptorSetLayout>& layout, VkFormat format, std::vector< VkPipelineShaderStageCreateInfo >& shaderStages, VkPipelineVertexInputStateCreateInfo& vertexInputState, uint32_t pushConstantSize ) {
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
	if (pushConstantSize == 0) {
		pipelineLayoutCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = static_cast<uint32_t>(layout.size()),
			.pSetLayouts = layout.data()
		};
	}
	else {
		VkPushConstantRange pushConstantRange{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = pushConstantSize
		};

		pipelineLayoutCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = static_cast<uint32_t>(layout.size()),
			.pSetLayouts = layout.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange
		};
	}

	vkCreatePipelineLayout(vulkanContext -> logicalDeviceIF.handle, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
	};

	std::vector< VkDynamicState > dynamicStates{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamicStates.data()
	};

	VkPipelineViewportStateCreateInfo viewportState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1
	};

	VkPipelineDepthStencilStateCreateInfo depthStencilState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS
	};

	VkPipelineRenderingCreateInfo renderingCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &format,
	};

	VkPipelineColorBlendAttachmentState blendAttachment{
		.colorWriteMask = 0xF
	};


	VkPipelineColorBlendStateCreateInfo colorBlendState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blendAttachment
	};

	VkPipelineRasterizationStateCreateInfo rasterizationState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo multisampleState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
	};

	VkGraphicsPipelineCreateInfo pipelineCreateInfo{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCreateInfo,
		.stageCount = static_cast< uint32_t >( shaderStages.size() ),
		.pStages = shaderStages.data(),
		.pVertexInputState = &vertexInputState,
		.pInputAssemblyState = &inputAssemblyState,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizationState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthStencilState,
		.pColorBlendState = &colorBlendState,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout
	};

	vkCreateGraphicsPipelines( vulkanContext->logicalDeviceIF.handle, nullptr, 1, &pipelineCreateInfo, nullptr, &pipeline);
}