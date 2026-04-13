#include "FNeuralNetwork.h"

void FNeuralNetwork::Initialize(const TArray<int32>& LayerSizes)
{
	Layers.Empty();

	if (LayerSizes.Num() < 2) {
		return;
	}

	for (int32 i = 0; i < LayerSizes.Num() - 1; ++i) {
		FNeuralLayer NewLayer;

		NewLayer.Initialize(LayerSizes[i], LayerSizes[i + 1]);
		Layers.Add(NewLayer);
	}
}

TArray<float> FNeuralNetwork::CalculateOutputs(TArray<float> Inputs)
{
	for (FNeuralLayer& Layer : Layers) {
		Inputs = Layer.CalculateOutputs(Inputs);
	}

	return Inputs;
}

void FNeuralNetwork::Learn(const TArray<float>& TrainingInput, const TArray<float>& ExpectedOutput, float LearningRate)
{
	CalculateOutputs(TrainingInput);

	FNeuralLayer& OutputLayer = Layers.Last();
	OutputLayer.CalculateOutputLayerNodeValues(ExpectedOutput);

	for (int32 i = Layers.Num() - 2; i >= 0; --i) {
		Layers[i].CalculateHiddenLayerNodeValues(Layers[i + 1]);
	}

	for (FNeuralLayer& Layer : Layers) {
		Layer.ApplyGradients(LearningRate);
	}
}

void FNeuralLayer::Initialize(int32 InNumNodesIn, int32 InNumNodesOut)
{
	NumNodesIn = InNumNodesIn;
	NumNodesOut = InNumNodesOut;

	int32 TotalWeights = NumNodesIn * NumNodesOut;

	Weights.Init(0.0f, TotalWeights);
	Biases.Init(0.0f, NumNodesOut);

	for (int32 i = 0; i < TotalWeights; ++i) {
		Weights[i] = FMath::FRandRange(-1.0f, 1.0f);
	}
}

TArray<float> FNeuralLayer::CalculateOutputs(const TArray<float>& InInputs)
{
	Inputs = InInputs;

	TArray<float> Outputs;
	Outputs.Init(0.0f, NumNodesOut);

	for (int32 NodeOut = 0; NodeOut < NumNodesOut; ++NodeOut) {
		float WeightedInput = Biases[NodeOut];

		for (int32 NodeIn = 0; NodeIn < NumNodesIn; ++NodeIn) {
			WeightedInput += Inputs[NodeIn] * Weights[NodeIn * NumNodesOut + NodeOut];
		}

		Outputs[NodeOut] = 1.0f / (1.0f + FMath::Exp(-WeightedInput));
	}

	Activations = Outputs;

	return Outputs;
}

void FNeuralLayer::CalculateOutputLayerNodeValues(const TArray<float>& ExpectedOutputs)
{
	NodeValues.Init(0.0f, NumNodesOut);

	for (int32 NodeOut = 0; NodeOut < NumNodesOut; ++NodeOut) {
		float CoastDerivative = Activations[NodeOut] - ExpectedOutputs[NodeOut];

		float ActivationDerivative = Activations[NodeOut] * (1.0f - Activations[NodeOut]);

		NodeValues[NodeOut] = CoastDerivative * ActivationDerivative;
	}
}

void FNeuralLayer::CalculateHiddenLayerNodeValues(const FNeuralLayer& NextLayer)
{
	NodeValues.Init(0.0f, NumNodesOut);

	for (int32 NodeOut = 0; NodeOut < NumNodesOut; ++NodeOut) {
		float DerivativeSum = 0.0f;

		for (int32 NodeNext = 0; NodeNext < NextLayer.NumNodesOut; ++NodeNext) {
			float WeightToNextNode = NextLayer.Weights[NodeOut * NextLayer.NumNodesOut + NodeNext];
			DerivativeSum += WeightToNextNode * NextLayer.NodeValues[NodeNext];
		}

		float ActivationDerivative = Activations[NodeOut] * (1.0f - Activations[NodeOut]);
		NodeValues[NodeOut] = DerivativeSum * ActivationDerivative;
	}
}

void FNeuralLayer::ApplyGradients(float LearningRate)
{
	for (int32 NodeOut = 0; NodeOut < NumNodesOut; ++NodeOut) {
		for (int32 NodeIn = 0; NodeIn < NumNodesIn; ++NodeIn) {
			float Gradient = Inputs[NodeIn] * NodeValues[NodeOut];
			Weights[NodeIn * NumNodesOut + NodeOut] -= Gradient * LearningRate;
		}

		Biases[NodeOut] -= NodeValues[NodeOut] * LearningRate;
	}
}
