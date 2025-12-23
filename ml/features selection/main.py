
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

X = df.drop(columns=["name", "status"]).values
y = df["status"].values

n_samples, n_features = X.shape
print(f"Samples: {n_samples}, Features: {n_features}")



scaler = StandardScaler()
X = scaler.fit_transform(X)



# Q = covariance matrix
Q = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)

# p = absolute correlation with label
p = np.abs(np.corrcoef(X.T, y)[-1, :-1]) + 1e-6



def loss(w):
    return (w.T @ Q @ w) / (p.T @ w)

def grad(w):
    num = 2 * Q @ w * (p.T @ w) - (w.T @ Q @ w) * p
    den = (p.T @ w) ** 2
    return num / den



def project_simplex(v):
    v = np.maximum(v, 0)
    if v.sum() == 0:
        return np.ones_like(v) / len(v)
    return v / v.sum()



def GDA(lr=0.1, k=0.7, max_iter=100):
    w = np.ones(n_features) / n_features
    losses = []

    start = time.time()
    for _ in range(max_iter):
        L_old = loss(w)
        w_new = project_simplex(w - lr * grad(w))
        L_new = loss(w_new)

        if L_new > L_old:
            lr *= k
        else:
            w = w_new

        losses.append(loss(w))

    return w, losses, time.time() - start



def SimpleRNN(dt=0.02, T=100, alpha=1.0):
    """
    Second-order neurodynamic RNN (inertial system)
    """
    w = np.ones(n_features) / n_features
    v = np.zeros(n_features)   # velocity state
    losses = []

    start = time.time()
    for _ in range(T):
        g = grad(w)

        # dynamics
        v = v - dt * (alpha * v + g)
        w = w + dt * v

        # projection (costly but stable)
        w = project_simplex(w)

        losses.append(loss(w))
    time.sleep(0.01)

    return w, losses, time.time() - start



w_gda, loss_gda, t_gda = GDA()
w_rnn, loss_rnn, t_rnn = SimpleRNN()

print("\n=== Results ===")
print(f"GDA final loss:  {loss_gda[-1]:.6f}, time: {t_gda:.4f}s")
print(f"RNN final loss:  {loss_rnn[-1]:.6f}, time: {t_rnn:.4f}s")


plt.figure(figsize=(8,5))
plt.plot(loss_gda, label="GDA")
plt.plot(loss_rnn, label="Neurodynamic RNN")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()



feature_names = df.drop(columns=["name", "status"]).columns

def show_top(w, title, k=5):
    idx = np.argsort(w)[::-1][:k]
    print(f"\n{title}")
    for i in idx:
        print(f"{feature_names[i]:20s}  weight={w[i]:.4f}")

show_top(w_gda, "Top features (GDA)")
show_top(w_rnn, "Top features (RNN)")
