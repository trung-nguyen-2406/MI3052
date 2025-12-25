# data_setup.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128):
    print("Dang chuan bi du lieu CIFAR-10...")
    
    # 1. Chuan hoa du lieu (Bước bắt buộc cho ResNet)
    # Các thông số mean/std chuẩn của CIFAR10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2. Tai du lieu ve may
    # trainset: Dữ liệu dùng để học
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # testset: Dữ liệu dùng để kiểm tra
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    print("Da tai va xu ly xong du lieu!")
    return trainloader, testloader, classes

# Test thu xem code chay duoc khong
if __name__ == "__main__":
    train_loader, _, _ = get_cifar10_loaders()
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Shape cua mot batch anh: {images.shape}") 
    # Ket qua nen la: torch.Size([128, 3, 32, 32])