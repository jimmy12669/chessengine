import torch
import numpy as np
import os

m = torch.load("models/nnue_best.pt", map_location="cpu", weights_only=False)
sd = m["state"]

np.savez("models/weights.npz",
    ft_W  = sd["ft.weight"].numpy(),
    ft_b  = sd["ft.bias"].numpy(),
    l1_W  = sd["l1.weight"].numpy(),
    l1_b  = sd["l1.bias"].numpy(),
    l2_W  = sd["l2.weight"].numpy(),
    l2_b  = sd["l2.bias"].numpy(),
    out_W = sd["out.weight"].numpy(),
    out_b = sd["out.bias"].numpy(),
)
print("done")