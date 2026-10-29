#pragma once

#include<volk/volk.h>
#include<iostream>

using namespace std;

struct ValidationHelpers {
	void validateResult(VkResult result, string message = "ERROR") {
		if (result != VK_SUCCESS) {
			cerr << "ERROR: " << message << endl;
			exit(result);
		}
	}

	void validateResult(bool result, string message = "ERROR") {
		if (!result) {
			cerr << "ERROR: " << message << endl;
			exit(result);
		}
	}

	void validateSwapchain(VkResult result, bool& value) {
		if (result < VK_SUCCESS) {
			if (result == VK_ERROR_OUT_OF_DATE_KHR) {
				value = true;
				return;
			}

			cerr << "ERROR: Swapchain Validation Failed" << endl;
			exit(result);
		}
	}
} validationIF;