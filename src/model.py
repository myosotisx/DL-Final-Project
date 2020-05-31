from .resnet_mod import resnet50_mod, resnet18_mod


def model():
    # return resnet50_mod(pretrained=False, progress=False)
    return resnet18_mod(pretrained=False, progress=False)
