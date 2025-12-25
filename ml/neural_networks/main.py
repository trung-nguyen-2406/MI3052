import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torchvision.models as models

# --- IMPORT FILE CUA BAN ---
# Đảm bảo file data_setup.py và sgda.py nằm cùng thư mục với main.py
from data_setup import get_cifar10_loaders 
from sgda import SGDA

# --- CAU HINH ---
NUM_EPOCHS = 50     
BATCH_SIZE = 128
LR_INIT = 0.1      # Toc do hoc khoi tao
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): DEVICE = 'mps' # Cho Macbook M1/M2

print(f"Dang su dung thiet bi: {DEVICE}")

def train_model(optimizer_name, trainloader, testloader):
    print(f"\n>>> Bat dau training voi thuat toan: {optimizer_name}")
    
    # 1. Khoi tao Model ResNet-18
    # num_classes=10 vi CIFAR10 co 10 lop
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(DEVICE)

    # 2. Chon Optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGDA':
        # Cac tham so kappa, sigma co the chinh sua. 
        # O day de kappa=0.9, sigma=0.5 nhu thuong thay trong bai bao
        optimizer = SGDA(model.parameters(), lr=LR_INIT, kappa=0.9, sigma=0.5)
    else:
        # SGD thong thuong (Baseline)
        optimizer = optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=5e-4)

    # Luu ket qua de ve bieu do
    train_losses = []
    test_accuracies = []
    
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Dinh nghia closure cho SGDA (va SGD cung dung chung cau truc cho tien)
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            # Buoc update quan trong
            # Voi SGDA, no se tinh toan va tu dieu chinh LR ben trong
            loss = optimizer.step(closure)
            
            running_loss += loss.item()

        # Tinh loss trung binh cua epoch
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # Danh gia tren tap Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        test_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Hoan thanh {optimizer_name} trong {total_time:.1f}s")
    
    return train_losses, test_accuracies

def main():
    # Lay du lieu
    trainloader, testloader, _ = get_cifar10_loaders(BATCH_SIZE)

    # Chay thuc nghiem 1: SGDA (De xuat)
    sgda_losses, sgda_accs = train_model('SGDA', trainloader, testloader)

    # Chay thuc nghiem 2: SGD (So sanh)
    sgd_losses, sgd_accs = train_model('SGD', trainloader, testloader)

    # --- VE BIEU DO ---
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Bieu do 1: Training Loss (Giong Hinh 7 trong bai bao)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, sgda_losses, label='SGDA (Proposed)', color='blue', marker='o')
    plt.plot(epochs_range, sgd_losses, label='SGD', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss comparison')
    plt.legend()
    plt.grid(True)

    # Bieu do 2: Test Accuracy (Giong Hinh 6 trong bai bao)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, sgda_accs, label='SGDA (Proposed)', color='blue', marker='o')
    plt.plot(epochs_range, sgd_accs, label='SGD', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_result.png') # Luu anh
    print("\nDa luu bieu do vao file 'comparison_result.png'. Mo file ra xem ket qua nhe!")
    plt.show()

if __name__ == "__main__":
    main()