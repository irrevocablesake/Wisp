#pragma once

#include "GPUSimulationParams.h"

#include<cuda_runtime.h>
#include <math_constants.h>

class TimeStep {
public:
	const GPUSimulationParams *simulationParams;
	cudaStream_t stream;

	struct PartialStep {
		float maxVelocity = 0.0f;
		float maxAcceleration = 0.0f;
	} ;

	struct Step {
		float deltaTime = 0.0f;
		float maxVelocity = 0.0f;
		float maxAcceleration = 0.0f;
		float accumulator = 0.0f;
	} ;

	struct StepBuffer {
		PartialStep* partialStepBuffer;
		Step* stepBuffer;

		int MAX_BLOCKS = 0;
	} stepBuffer;

	void initialize( const GPUSimulationParams &_simulationParams, cudaStream_t &stream );
	void addFrameTimeToAccumulator( float frameDeltaTime );
	void calculateAdaptiveTimeStep();
};