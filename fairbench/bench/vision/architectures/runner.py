def run_dataset(
    classifiers, test_loader, classifier, predict, device, unpacking=(0, 1, 2)
):
    import torch
    from tqdm import tqdm

    predictors = {
        "predict": lambda outputs: outputs.data.max(1, keepdim=True)[1].squeeze(1),
        "probabilities": lambda outputs: torch.softmax(outputs, dim=1)[:, 1],
    }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(predict, str):
        predict = predictors[predict]
    if isinstance(classifier, str):
        classifier = classifiers[classifier.lower()](device)
    assert isinstance(classifier, torch.nn.Module), "Classifier is not a torch model."

    y, yhat, sens = [], [], []
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[unpacking[0]], data[unpacking[1]]
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)  # Forward pass
            preds = predict(outputs)
            yhat.extend(preds.cpu().tolist())

            y.extend(labels.cpu().tolist())
            if unpacking[2] is not None:
                biases = data[unpacking[2]]
                sens.extend(biases.cpu().tolist())
    return y, yhat, sens
