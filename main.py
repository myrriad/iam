# load pdfs 

# taken from https://godatadriven.com/blog/how-to-build-your-first-image-classifier-using-pytorch/

from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_set = ImageFolder('images/train', transform=train_transform)
test_set = ImageFolder('images/test', transform=test_transform)