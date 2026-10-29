#pragma once

#include<volk/volk.h>

#include "slang/slang.h"
#include "slang/slang-com-ptr.h"

class Shaders {
	public:
		Slang::ComPtr< slang::IGlobalSession > slangGlobalSession;
		Slang::ComPtr< slang::ISession > slangSession;

		void setupSLANG();
		VkShaderModule loadAndCompileShaders(VkDevice& device, const char* shaderName, const char* filePath);
};