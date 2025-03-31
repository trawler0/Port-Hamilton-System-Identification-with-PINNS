import numpy as np
from matplotlib import pyplot as plt
import torch

data = np.load("results.npy", allow_pickle=True)
mn = [
        ("spring_multi_mass", "spring_chain"),
        ("spring_multi_mass", "default"),
        ("spring_multi_mass", "baseline"),
        ("ball", "affine"),
        ("ball", "default"),
        ("motor", "baseline"),
        ("motor", "affine"),
        ("spring", "default"),
        ("spring", "affine"),
        ("spring", "baseline")
    ]
data = data.item()
method, name = mn[6]
print(method, name)
mae = data[f"mae_{name}_{method}"]
accurate_time = data[f"acc_{name}_{method}"]
compute = data[f"compute_{name}_{method}"]
trajectories = data[f"traj_{name}_{method}"]
sizes = data[f"sizes_{name}_{method}"]

# mae = accurate_time[:, 2]
print(mae)

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.scatter(compute, mae, c=np.log(trajectories), cmap='coolwarm', edgecolor='k', s=sizes**2/16)
ax.set_xscale("log")
ax.set_yscale("log")

def filter(compute, mae):
    idx = np.argsort(compute)
    c = compute[idx]
    m = mae[idx]
    out = [(c[0], m[0])]
    for i in range(1, len(c)):
        if m[i] < out[-1][1]:
            out.append([c[i], m[i]])
    c = np.array(out)[:, 0]
    m = np.array(out)[:, 1]
    return c, m

"""C, M = filter(compute, mae)
ax.scatter(C, M)"""
C = compute

a = 0
b = .001
c = 10
d = -.5
law = a - b*(C + c)**d
ax.plot(C, law, 'k--')
"""weight = torch.nn.Parameter(torch.tensor([a, b, c, d]).float())
optimizer = torch.optim.Adam([weight], lr=1e-2)
dataset = torch.utils.data.TensorDataset(torch.tensor(C).float(), torch.tensor(mae).float())
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(100):
    for x, y in loader:
        optimizer.zero_grad()
        pred = weight[0] + weight[1]*(x + weight[2])**weight[3]
        y = torch.log(y)
        pred = torch.log(torch.relu(pred) + 1e-6)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")"""

"""a, b, c, d = weight.detach().numpy()
law = a + b*(compute + c)**d
plt.plot(compute, law, 'k--')"""
plt.show()