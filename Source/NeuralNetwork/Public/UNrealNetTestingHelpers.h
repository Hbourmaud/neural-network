#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UNrealNetTestingHelpers.generated.h"

UCLASS()
class NEURALNETWORK_API UNrealNetTestingHelpers : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
	
public:
	UFUNCTION(BlueprintCallable, Category = "Neural Network Test")
	static void RunForwardPassTest();

	UFUNCTION(BlueprintCallable, Category = "Neural Network Test")
	static void TrainAndTestXOR();
};
