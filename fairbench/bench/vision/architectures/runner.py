from tqdm import tqdm


def run_dataset(classifiers, test_loader, classifier, predict, device):
    import torch

    assert predict in {
        "predict",
        "probabilities",
    }, "Invalid predict value. Use 'predict' or 'probabilities'."

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(classifier, str):
        classifier = classifier.lower()
        assert (
            classifier in classifiers
        ), f"Available classifiers are: {list(classifiers)}"
        classifier = classifiers[classifier](device)
    assert isinstance(classifier, torch.nn.Module), "Classifier is not a torch model."

    y, yhat, sens = [], [], []
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        for images, labels, biases, _ in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)  # Forward pass

            if predict == "predict":
                preds = outputs.data.max(1, keepdim=True)[1].squeeze(1)
                yhat.extend(preds.cpu().tolist())
            elif predict == "probabilities":
                probs = torch.softmax(outputs, dim=1)
                yhat.extend(probs.cpu().tolist())

            y.extend(labels.cpu().tolist())
            sens.extend(biases.cpu().tolist())
    return y, yhat, sens
