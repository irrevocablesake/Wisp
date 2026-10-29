#include "shaders.h"

#include<array>

void Shaders::setupSLANG() {
	slang::createGlobalSession(slangGlobalSession.writeRef());

	auto slangTargets{
		std::to_array< slang::TargetDesc >({{
			.format{SLANG_SPIRV},
			.profile{slangGlobalSession->findProfile("spirv_1_4")}
		}})
	};

	auto slangOptions{
		std::to_array < slang::CompilerOptionEntry>({{
			slang::CompilerOptionName::EmitSpirvDirectly,
			{
				slang::CompilerOptionValueKind::Int, 1
			}
		}})
	};

	slang::SessionDesc slangSessionDesc{
		.targets{
			slangTargets.data()
		},
		.targetCount{
			SlangInt(slangTargets.size())
		},
		.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
		.compilerOptionEntries{
			slangOptions.data()
		},
		.compilerOptionEntryCount{
			uint32_t(slangOptions.size())
		}
	};

	slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef());
}

VkShaderModule Shaders::loadAndCompileShaders(VkDevice& device, const char* shaderName, const char* filePath) {
	Slang::ComPtr< slang::IModule > slangModule{
		slangSession->loadModuleFromSource(shaderName, filePath, nullptr, nullptr)
	};

	Slang::ComPtr< ISlangBlob > spirv;
	slangModule->getTargetCode(0, spirv.writeRef());

	VkShaderModuleCreateInfo shaderModuleCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = spirv->getBufferSize(),
		.pCode = (uint32_t*)spirv->getBufferPointer()
	};

	VkShaderModule shaderModule{};
	//should add a validation check here later
	vkCreateShaderModule( device, &shaderModuleCreateInfo, nullptr, &shaderModule);

	return shaderModule;
}