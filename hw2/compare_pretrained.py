"""
Homework target model
AlexNet
GoogLeNet
vgg16
resnet18

Must need ILSVRC2012 Validation set for evaluation
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import scipy
from matplotlib import pyplot as plt


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def compare_model(dev):
    # https://csm-kr.tistory.com/m/6
    valid_set = torchvision.datasets.ImageNet(root='./validset', transform=transform, split='val')
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=4)

    result = []

    for i in range(4):
        if i == 0:
            model = torchvision.models.alexnet(pretrained=True).to(dev)
        if i == 1:
            model = torchvision.models.vgg16(pretrained=True).to(dev)
        if i == 2:
            model = torchvision.models.googlenet(pretrained=True).to(dev)
        if i == 3:
            model = torchvision.models.resnet18(pretrained=True).to(dev)

        model.eval()

        acc_top1 = 0
        acc_top5 = 0
        total = 0

        '''Not need back prop'''
        with torch.no_grad():
            for j, (img, label) in enumerate(valid_loader):
                img = img.to(dev)
                label = label.to(dev)
                # Evaluate by batch size
                output = model(img)

                """rank 1"""
                _, pred = torch.max(output, 1)
                total += label.size(0)
                acc_top1 += (pred == label).sum().item()

                """rank 5"""
                _, rank5 = output.topk(5, 1, True, True)
                rank5 = rank5.t()
                correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))
                correct5 = correct5.contiguous()

                for k in range(6):
                    correct_k = correct5[:k].view(-1).float().sum(0, keepdim=True)
                acc_top5 += correct_k.item()

                print("step : {} / {}".format(j + 1, len(valid_set) / int(label.size(0))))
                print("Top-1 Accuracy :  {0:0.2f}%".format(acc_top1 / total * 100))
                print("Top-5 Accuracy :  {0:0.2f}%".format(acc_top5 / total * 100))

        print("Final result")
        print("Top-1 Accuracy :  {0:0.2f}%".format(acc_top1 / total * 100))
        print("Top-5 Accuracy:  {0:0.2f}%".format(acc_top5 / total * 100))
        result.append(acc_top1 / total * 100)
        result.append(acc_top5 / total * 100)
    return result


def draw_result(eval_output):
    model_label = [
        'alexNet Top-1',
        'alexNet Top-5',
        'VGG16 Top-1',
        'VGG16 Top-5',
        'googLeNet Top-1',
        'googLeNet Top-5',
        'resnet18 Top-1',
        'resnet18 Top-5'
    ]

    fig, ax = plt.subplots()
    ax.barh(model_label, eval_output, height=0.6, color='orange', alpha=0.8)
    plt.xlabel('Accuracy (%)')
    plt.title('Compare 4 models')
    for i, v in enumerate(eval_output):
        ax.text(v + 3, i + .25, str(v), color='black')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluation_result = compare_model(device)
    draw_result(evaluation_result)
