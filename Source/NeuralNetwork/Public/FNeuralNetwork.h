#pragma once

#include "CoreMinimal.h"

struct FNeuralLayer
{
	int32 NumNodesIn;
	int32 NumNodesOut;

	TArray<float> Weights;
	TArray<float> Biases;
	TArray<float> Activations;

	TArray<float> Inputs;
	TArray<float> NodeValues;

	void Initialize(int32 InNumNodesIn, int32 InNumNodesOut);
	TArray<float> CalculateOutputs(const TArray<float>& InInputs);

	void CalculateOutputLayerNodeValues(const TArray<float>& ExpectedOutputs);
	void CalculateHiddenLayerNodeValues(const FNeuralLayer& NextLayer);
	void ApplyGradients(float LearningRate);
};

class FNeuralNetwork
{
public:
	TArray<FNeuralLayer> Layers;

	void Initialize(const TArray<int32>& LayerSizes);

	TArray<float> CalculateOutputs(TArray<float> Inputs);

	void Learn(const TArray<float>& TrainingInput, const TArray<float>& ExpectedOutput, float LearningRate);
};