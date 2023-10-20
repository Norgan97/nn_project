from torchvision import transforms as T

preprocessing_func = T.Compose(
        [
            T.Resize((224, 244)),
            T.ToTensor()
        ]
    )


def preprocess(img):
    return preprocessing_func(img)
