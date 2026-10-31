#pragma once

#include<cuda_runtime.h>
#include<Windows.h>
#include<array>

class CudaContext {
	public: 
		struct CudaBuffer {
			cudaExternalMemory_t externalMemory{};
			void* devicePointer = nullptr;
		};

		void initialize(const std::array< uint8_t, 16 >& deviceUUID);

		CudaBuffer importBuffer(HANDLE sharedHandle, uint64_t allocationSize, uint64_t bufferSize );
		cudaExternalSemaphore_t importSemaphores( HANDLE vkToCudaHandle );

	public:
		int deviceIndex = -1;

		cudaStream_t stream{};
};