#pragma once

#include "CoreMinimal.h"

struct FMNISTDataset
{
	TArray<TArray<float>> Images;

	TArray<TArray<float>> Labels;
};

class FMNISTLoader
{
public:
	static bool LoadMNIST(const FString& ImageFilePath, const FString& LabelFilePath, FMNISTDataset& OutDataset);

private:
	// read 32bit int from MNIST binary file
	static int32 ReadBigEndianInt(const TArray<uint8>& Data, int32& Offset);
};