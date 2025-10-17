# DistilBERT classifier fine-tuned on the Go Emotions Dataset

This repository goal is to train a DistilBERT classifier on the Go Emotions Dataset, using DVC as an end-to-end pipeline workflow executor. It keeps track of dataset, training code, and trained model versions.

DVC first runs the "src/load_data", which reads the data from the Go Emotions Dataset and prepares it to the format expected by the DistilBERT training repository. Then another script trains the classifier and runs the evaluation, saving the final classifier and the evaluation results.

## Usage

### Download a trained model

The model is stored in Hugging Face storage, and the git repository stores only its reference. You can use git to download it.

```
git clone https://huggingface.co/RafaelBMR/distilbert-go-emotions trained_classifier
```

### Train a model

All hyper parameters, configurations and paths are controlled with the "params.yaml" file. Once those are set, the pipeline is executed with a single command.

```
dvc repro
```