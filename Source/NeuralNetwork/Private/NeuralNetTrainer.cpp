#include "NeuralNetTrainer.h"

bool UNeuralNetTrainer::InitTrainer(int32 HiddenNodes, float InLearningRate, int32 InEpochs)
{
	FString ImagePath = FPaths::ProjectContentDir() + "MNIST/train-images.idx3-ubyte";
	FString LabelPath = FPaths::ProjectContentDir() + "MNIST/train-labels.idx1-ubyte";

	if (Dataset.Images.Num() == 0) {
		if (!FMNISTLoader::LoadMNIST(ImagePath, LabelPath, Dataset)) {
			return false;
		}
	}

	Network.Initialize({ 784, HiddenNodes, 10 });

	LearningRate = InLearningRate;

	TotalEpochs = FMath::Max(1, InEpochs);
	
	CurrentEpoch = 0;
	CurrentTrainIndex = 0;

	NumTestsReserved = 1000;
	TrainingSamples = FMath::Max(0, Dataset.Images.Num() - NumTestsReserved);

	UE_LOG(LogTemp, Warning, TEXT("Trainer Initialized! Total Training Images: %d"), TrainingSamples);
	return true;
}

bool UNeuralNetTrainer::TrainBatch(int32 BatchSize)
{
	if (CurrentEpoch >= TotalEpochs) {
		return true;
	}

	for (int32 i = 0; i < BatchSize; ++i) {
		Network.Learn(Dataset.Images[CurrentTrainIndex], Dataset.Labels[CurrentTrainIndex], LearningRate);
		CurrentTrainIndex++;

		if (CurrentTrainIndex >= TrainingSamples) {
			CurrentTrainIndex = 0;
			CurrentEpoch++;

			UE_LOG(LogTemp, Warning, TEXT("Epoch %d completed!"), CurrentEpoch);

			if (CurrentEpoch >= TotalEpochs) {
				return true;
			}
		}
	}

	return false;
}

float UNeuralNetTrainer::GetTrainingProgress() const
{
	if (TotalEpochs <= 0 || TrainingSamples <= 0) {
		return 0.0f;
	}

	float TotalImagesToProcess = TotalEpochs * TrainingSamples;
	float ImagesProcessed = (CurrentEpoch * TrainingSamples) + CurrentTrainIndex;

	return ImagesProcessed / TotalImagesToProcess;
}

float UNeuralNetTrainer::TestAccuracy(int32 NumTests)
{
	int32 TestStartId = FMath::Min(TrainingSamples, Dataset.Images.Num());

	int32 ImagesLeftForTesting = Dataset.Images.Num() - TestStartId;

	int32 SafeNumTests = FMath::Min3(NumTests, NumTestsReserved, ImagesLeftForTesting);

	if (SafeNumTests <= 0) {
		return 0.0f;
	}

	int32 CorrectAnswers = 0;

	for (int32 i = TestStartId; i < TestStartId + SafeNumTests; ++i) {
		TArray<float> Prediction = Network.CalculateOutputs(Dataset.Images[i]);

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
		}
	}

	float Accuracy = (float)CorrectAnswers / (float)SafeNumTests;
	UE_LOG(LogTemp, Warning, TEXT("Test Accuracy over %d images: %.1f%%"), SafeNumTests, Accuracy * 100.0f);

	return Accuracy;
}