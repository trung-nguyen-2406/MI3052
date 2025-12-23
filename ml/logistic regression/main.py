import torch, time, os, urllib.request
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print("Using device:", device)


def load_dataset(name):
    if name == "mushrooms":
        url, path = \
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms", "/tmp/mushrooms"
    elif name == "w8a":
        url, path = \
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a", "/tmp/w8a"
    else:
        raise ValueError

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    X, y = load_svmlight_file(path)
    X = X.toarray()

    if name == "w8a":
        X = StandardScaler().fit_transform(X)

    y = torch.tensor(y, device=device)
    y = torch.where(y > 0, 1.0, -1.0)

    return torch.tensor(X, device=device), y

def loss_fn(w, X, y):
    z = X @ w
    logistic = torch.mean(torch.nn.functional.softplus(-y * z))
    reg = 0.5 * torch.dot(w, w) / X.shape[0]
    return logistic + reg


def GD(X, y, lr=1e-2, iters=3000):
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    losses = []

    for _ in range(iters):
        J = loss_fn(w, X, y)
        J.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        losses.append(J.item())

    return w.detach(), losses

def GDA(X, y, lr=0.1, k=0.75, iters=3000):
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    losses = []
    prev = loss_fn(w, X, y).item()

    for _ in range(iters):
        J = loss_fn(w, X, y)
        J.backward()
        with torch.no_grad():
            w_new = w - lr * w.grad

        new = loss_fn(w_new, X, y).item()
        if new > prev:
            lr *= k
        else:
            w.data = w_new.data
            prev = new

        w.grad.zero_()
        losses.append(prev)

    return w.detach(), losses

def Nesterov(X, y, lr=0.001, m=0.95, iters=3000):
    w = torch.zeros(X.shape[1], device=device)
    v = torch.zeros_like(w)
    losses = []

    for _ in range(iters):
        yk = (w + m * v).detach().requires_grad_(True)
        J = loss_fn(yk, X, y)
        J.backward()
        with torch.no_grad():
            v = m * v - lr * yk.grad
            w = w + v
        losses.append(J.item())

    return w, losses



def run(name):
    print(f"\n===== {name} =====")
    X, y = load_dataset(name)

    w_gd, l_gd = GD(X, y)
    w_gda, l_gda = GDA(X, y)
    w_nes, l_nes = Nesterov(X, y)

    acc = lambda w: accuracy_score(y.cpu(), torch.sign(X @ w).cpu())

    print(f"GD       acc={acc(w_gd):.4f}")
    print(f"GDA      acc={acc(w_gda):.4f}")
    print(f"Nesterov acc={acc(w_nes):.4f}")

    plt.plot(l_gd, label="GD")
    plt.plot(l_gda, label="GDA")
    plt.plot(l_nes, label="Nesterov")
    plt.title(name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run("mushrooms")
    run("w8a")
