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