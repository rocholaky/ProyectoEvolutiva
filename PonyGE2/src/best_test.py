import torch
model = torch.load("PonyGE2\\src\\best.pkl")
model.to_string()