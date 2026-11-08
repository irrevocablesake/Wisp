#pragma once

#include<volk/volk.h>
#include<iostream>

struct ValidationHelpers {
	void validateResult(VkResult result, std::string message = "ERROR") {
		if (result != VK_SUCCESS) {
			std::cerr << "ERROR: " << message << std::endl;
			exit(result);
		}
	}

	void validateResult(bool result, std::string message = "ERROR") {
		if (!result) {
			std::cerr << "ERROR: " << message << std::endl;
			exit(result);
		}
	}

	void validateSwapchain(VkResult result, bool& value) {
		if (result < VK_SUCCESS) {
			if (result == VK_ERROR_OUT_OF_DATE_KHR) {
				value = true;
				return;
			}

			std::cerr << "ERROR: Swapchain Validation Failed" << std::endl;
			exit(result);
		}
	}
} validationIF;