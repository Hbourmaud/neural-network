#include "UNrealNetTestingHelpers.h"
#include "FNeuralNetwork.h"

void UNrealNetTestingHelpers::RunForwardPassTest()
{
	FNeuralNetwork MyNetwork;

	MyNetwork.Initialize({ 2, 3, 1 });

	TArray<float> TestInput = { 1.0f, 0.0f };

	TArray<float> Prediction = MyNetwork.CalculateOutputs(TestInput);

	if (Prediction.Num() > 0){
		UE_LOG(LogTemp, Log, TEXT("Forward pass test completed. Output: %f"), Prediction[0]);
	} else {
		UE_LOG(LogTemp, Warning, TEXT("Forward pass test completed. No output generated."));
	}
}

void UNrealNetTestingHelpers::TrainAndTestXOR()
{
	FNeuralNetwork MyNetwork;

	MyNetwork.Initialize({ 2, 4, 1 });

	TArray<TArray<float>> TrainingInputs = {
	{ 0.0f, 0.0f },
	{ 0.0f, 1.0f },
	{ 1.0f, 0.0f },
	{ 1.0f, 1.0f }
	};

	TArray<TArray<float>> ExpectedOutputs = {
		{ 0.0f },
		{ 1.0f },
		{ 1.0f },
		{ 0.0f }
	};

	float LearningRate = 0.5f;

	int32 IterationsNumber = 10000;

	UE_LOG(LogTemp, Warning, TEXT("Starting XOR Training (%d iterations)"), IterationsNumber);

	for (int32 i = 0; i < IterationsNumber; ++i) {
		int32 Index = FMath::RandRange(0, 3);
		MyNetwork.Learn(TrainingInputs[Index], ExpectedOutputs[Index], LearningRate);
	}

	UE_LOG(LogTemp, Warning, TEXT("Results : "));

	for (int32 i = 0; i < 4; ++i) {
		TArray<float> Prediction = MyNetwork.CalculateOutputs(TrainingInputs[i]);

		UE_LOG(LogTemp, Log, TEXT("Input: [%.1f, %.1f] | Expected: %.1f | Prediction: %f"),
			TrainingInputs[i][0], TrainingInputs[i][1], ExpectedOutputs[i][0], Prediction[0]);
	}
}