from resnet_mod import resnet50_mod


def model():
    return resnet50_mod(pretrained=False, progress=False)
