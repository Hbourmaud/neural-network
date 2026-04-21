#include "UNeuralNetTestingHelpers.h"
#include "FNeuralNetwork.h"

#include "FMNISTLoader.h"
#include "Misc/Paths.h"

void UNeuralNetTestingHelpers::RunForwardPassTest()
{
	FNeuralNetwork MyNetwork;

	MyNetwork.Initialize({ 2, 3, 1 });

	TArray<float> TestInput = { 1.0f, 0.0f };

	TArray<float> Prediction = MyNetwork.CalculateOutputs(TestInput);

	if (Prediction.Num() > 0){
		UE_LOG(LogTemp, Log, TEXT("Forward pass test completed. Output: %f"), Prediction[0]);
	} else {
		UE_LOG(LogTemp, Warning, TEXT("Forward pass test completed."));
	}
}

void UNeuralNetTestingHelpers::TrainAndTestXOR()
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

void UNeuralNetTestingHelpers::TrainAndTestMNIST()
{
	FMNISTDataset Dataset;
	FString ImagePath = FPaths::ProjectContentDir() + "MNIST/train-images.idx3-ubyte";
	FString LabelPath = FPaths::ProjectContentDir() + "MNIST/train-labels.idx1-ubyte";

	if (!FMNISTLoader::LoadMNIST(ImagePath, LabelPath, Dataset)) {
		return;
	}

	FNeuralNetwork MyNetwork;

	MyNetwork.Initialize({ 784, 64, 10 });

	float LearningRate = 0.1f;
	int32 NumTests = 20;

	int32 TrainingSamples = FMath::Min(60000, Dataset.Images.Num() - NumTests);

	UE_LOG(LogTemp, Warning, TEXT("Starting MNIST Training on %d images"), TrainingSamples);

	for (int32 i = 0; i < TrainingSamples; ++i)	{
		MyNetwork.Learn(Dataset.Images[i], Dataset.Labels[i], LearningRate);
	}

	UE_LOG(LogTemp, Warning, TEXT("MNIST Training complete. Testing on images :"));

	int32 TestStart = TrainingSamples;
	int32 CorrectAnswers = 0;

	for (int32 i = TestStart; i < TestStart + NumTests; ++i) {
		TArray<float> Prediction = MyNetwork.CalculateOutputs(Dataset.Images[i]);
		
		int32 PredictedDigit = 0;
		float HighestConfidence = Prediction[0];

		for (int32 j = 1; j < 10; ++j) {
			if (Prediction[j] > HighestConfidence) {
				HighestConfidence = Prediction[j];
				PredictedDigit = j;
			}
		}

		int32 ActualDigit = 0;

		for (int32 j = 0; j < 10; ++j) {
			if (Dataset.Labels[i][j] > 0.5f) {
				ActualDigit = j;

				break;
			}
		}

		if (PredictedDigit == ActualDigit) {
			CorrectAnswers++;

			UE_LOG(LogTemp, Log, TEXT("SUCCESS Expected: %d, Guessed: %d (Confidence: %.2f)"), ActualDigit, PredictedDigit, HighestConfidence);
		} else {
			UE_LOG(LogTemp, Error, TEXT("FAIL Expected: %d, Guessed: %d"), ActualDigit, PredictedDigit);
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("Test Accuracy: %d / %d"), CorrectAnswers, NumTests);
}
