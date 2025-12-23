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



def GDA(X, y, lr=0.5, k=0.75, iters=1000):
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



def run(name):
    print(f"\n===== {name} =====")
    X, y = load_dataset(name)

    w_gda1, l_gda1 = GDA(X, y, 0.75)
    w_gda2, l_gda2 = GDA(X, y, 0.85)
    w_gda3, l_gda3 = GDA(X, y, 0.95)

    acc = lambda w: accuracy_score(y.cpu(), torch.sign(X @ w).cpu())

    print(f"GDA 0.75    acc={acc(w_gda1):.4f}")
    print(f"GDA 0.85    acc={acc(w_gda2):.4f}")
    print(f"GDA 0.95    acc={acc(w_gda3):.4f}")

    plt.plot(l_gda1, label="GDA 0.75")
    plt.plot(l_gda2, label="GDA 0.85")
    plt.plot(l_gda3, label="GDA 0.95")
    plt.title(name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run("w8a")
