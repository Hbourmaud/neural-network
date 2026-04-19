#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UNeuralNetTestingHelpers.generated.h"

UCLASS()
class NEURALNETWORK_API UNeuralNetTestingHelpers : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
	
public:
	UFUNCTION(BlueprintCallable, Category = "Neural Network Test")
	static void RunForwardPassTest();

	UFUNCTION(BlueprintCallable, Category = "Neural Network Test")
	static void TrainAndTestXOR();

	UFUNCTION(BlueprintCallable, Category = "Neural Network Test")
	static void TrainAndTestMNIST();
};
