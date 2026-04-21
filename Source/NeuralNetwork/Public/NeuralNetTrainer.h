#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include <FNeuralNetwork.h>
#include <FMNISTLoader.h>
#include "NeuralNetTrainer.generated.h"

UCLASS(BlueprintType, Blueprintable)
class NEURALNETWORK_API UNeuralNetTrainer : public UObject
{
	GENERATED_BODY()
	
public:
	UFUNCTION(BlueprintCallable, Category = "Neural Network Trainer")
	bool InitTrainer(int32 HiddenNodes = 64, float InLearningRate = 0.1f, int32 InEpochs = 3);

	UFUNCTION(BlueprintCallable, Category = "Neural Network Trainer")
	bool TrainBatch(int32 BatchSize = 500);

	UFUNCTION(BlueprintPure, Category = "Neural Network Trainer")
	float GetTrainingProgress() const;

	UFUNCTION(BlueprintCallable, Category = "Neural Network Trainer")
	float TestAccuracy(int32 NumTests = 100);

private:
	FNeuralNetwork Network;
	FMNISTDataset Dataset;

	float LearningRate;

	int32 TotalEpochs;
	int32 CurrentEpoch;
	int32 CurrentTrainIndex;

	int32 TrainingSamples;
	int32 NumTestsReserved;
};
