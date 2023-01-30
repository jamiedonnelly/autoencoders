import torchvision.models as models

model_dict = {
    'alex':models.alexnet,
    'res18':models.resnet18,
    'res34':models.resnet34,
    'res50':models.resnet50,
    'res101':models.resnet101,
    'res152':models.resnet152,
    'vgg11':models.vgg11,
    'vgg13':models.vgg13,
    'vgg19':models.vgg19,
    'vgg19bn':models.vgg19_bn
}