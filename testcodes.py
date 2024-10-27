from model import PHNNModel
import torch
from tqdm import tqdm
from utils import Dataset
device = "cpu"

model = PHNNModel(4, 64, J="sigmoid", R="sigmoid", grad_H="gradient").to(device)
X = torch.rand(5000, 256, 4).to(device)
U = torch.rand(5000, 256, 1).to(device)
y = torch.rand(5000, 256, 4).to(device)
ds = Dataset(X.view(-1, 4), U.view(-1, 1), y.view(-1, 4))
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-1)
for j in tqdm(range(len(X))):
        x, u = X[j], U[j]
        optimizer.zero_grad()
        model(x, u).sum().backward()
        optimizer.step()
for x, u, y in tqdm(loader):
        optimizer.zero_grad()
        model(x, u).sum().backward()
        optimizer.step()