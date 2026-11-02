#include "CudaContext.h"

#include <cuda_runtime.h>

using namespace std;

void CudaContext::initialize(const std::array<uint8_t, 16>& vulkanUUID) {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp properties{};

		cudaGetDeviceProperties(&properties, i);

		bool matches = std::memcmp( properties.uuid.bytes, vulkanUUID.data(), 16 ) == 0;

		if (matches) {
			cudaSetDevice(i);


			deviceIndex = i;


			cudaStreamCreate(&stream);


			return;
		}
	}
}

CudaContext::CudaBuffer CudaContext::importBuffer( HANDLE sharedHandle, uint64_t  allocationSize, uint64_t  bufferSize ) {
	CudaBuffer cudaBuffer;
	cudaExternalMemoryHandleDesc memoryDesc{};

	memoryDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	memoryDesc.handle.win32.handle = sharedHandle;
	memoryDesc.size = allocationSize;
	memoryDesc.flags = cudaExternalMemoryDedicated;

	cudaImportExternalMemory(
		&cudaBuffer.externalMemory,
		&memoryDesc
	);


	cudaExternalMemoryBufferDesc bufferDesc{};
	bufferDesc.offset = 0;
	bufferDesc.size = bufferSize;
	bufferDesc.flags = 0;

	cudaExternalMemoryGetMappedBuffer(
		&cudaBuffer.devicePointer,
		cudaBuffer.externalMemory,
		&bufferDesc
	);

	CloseHandle(sharedHandle);

	return cudaBuffer;
}

cudaExternalSemaphore_t CudaContext::importSemaphores( HANDLE vkToCudaHandle ){
	cudaExternalSemaphore_t semaphore{};

	cudaExternalSemaphoreHandleDesc desc{};
	desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	desc.handle.win32.handle = vkToCudaHandle;
	desc.flags = 0;

	cudaImportExternalSemaphore( &semaphore, &desc );

	return semaphore;
}