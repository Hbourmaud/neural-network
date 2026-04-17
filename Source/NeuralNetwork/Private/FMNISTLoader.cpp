#include "FMNISTLoader.h"

int32 FMNISTLoader::ReadBigEndianInt(const TArray<uint8>& Data, int32& Offset)
{
	// bit shift to construct the 32bit int
	int32 Value = (Data[Offset] << 24) | (Data[Offset + 1] << 16) | (Data[Offset + 2] << 8) | Data[Offset + 3];
	Offset += 4;

	return Value;
}

bool FMNISTLoader::LoadMNIST(const FString& ImageFilePath, const FString& LabelFilePath, FMNISTDataset& OutDataset)
{
	TArray<uint8> ImageData;
	TArray<uint8> LabelData;

	if (!FFileHelper::LoadFileToArray(ImageData, *ImageFilePath) || !FFileHelper::LoadFileToArray(LabelData, *LabelFilePath)) {
		UE_LOG(LogTemp, Error, TEXT("Failed to find MNIST files at paths : \n%s\n%s"), *ImageFilePath, *LabelFilePath);

		return false;
	}

	int32 LabelOffset = 0;
	int32 LabelMagicNumber = ReadBigEndianInt(LabelData, LabelOffset);
	int32 NumLabels = ReadBigEndianInt(LabelData, LabelOffset);

	int32 ImageOffset = 0;
	int32 ImageMagicNumber = ReadBigEndianInt(ImageData, ImageOffset);
	int32 NumImages = ReadBigEndianInt(ImageData, ImageOffset);
	int32 NumRows = ReadBigEndianInt(ImageData, ImageOffset);
	int32 NumCols = ReadBigEndianInt(ImageData, ImageOffset);

	if (NumImages != NumLabels)
	{
		UE_LOG(LogTemp, Error, TEXT("Image count and Label count do not match!"));

		return false;
	}

	int32 NumPixels = NumRows * NumCols;

	OutDataset.Images.SetNum(NumImages);
	OutDataset.Labels.SetNum(NumImages);

	for (int32 i = 0; i < NumImages; ++i) {
		OutDataset.Images[i].SetNum(NumPixels);

		for (int32 k = 0; k < NumPixels; ++k) {
			OutDataset.Images[i][k] = ImageData[ImageOffset++] / 255.0f;
		}

		uint8 LabelValue = LabelData[LabelOffset++];
		OutDataset.Labels[i].Init(0.0f, 10);

		if (LabelValue < 10) {
			OutDataset.Labels[i][LabelValue] = 1.0f;
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("Successfully loaded %d images!"), NumImages);

	return true;
}