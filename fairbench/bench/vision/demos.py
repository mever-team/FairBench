if __name__ == "__main__":
    # y, yhat = imagenet9(
    #     classifier="mavias",  # torch.nn.Module or string ("flac", "badd", "mavias")
    #     data_root="/fssd2/user-data/gsarridis/backgrounds_challenge_data",
    #     predict="predict",  # "predict" for class predictions, "probabilities" for probabilities
    #     device="cuda",  # "cpu" or "cuda" (or GPU device, e.g., "cuda:0")
    # )
    y, yhat, sens = celeba(
        classifier="flac",
        data_root="./data/celeba",
        predict="predict",
        device="cuda",
    )
    y, yhat, sens = utkface(
        classifier="badd",
        data_root="./data/utk_face",
        predict="predict",
        device="cuda",
    )
    y, yhat, sens = waterbirds(
        classifier="mavias",
        data_root="./data/waterbirds",
        predict="predict",
        device="cuda",
    )
