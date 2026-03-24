import torch
import numpy

# True = ignore
B=1
T = 10
mask_len = 3

mask = torch.zeros(B, T, dtype=torch.bool)
mask[:, -mask_len:] = True
print(mask)
second_mask = torch.zeros(B, T, dtype=torch.bool)

start_mask = T - mask_len
second_mask[:, start_mask:] = True
print(second_mask)