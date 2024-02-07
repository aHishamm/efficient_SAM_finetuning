import os
import torch
os.chdir('/Users/ahishamm/Documents/projects/efficientSAM/EfficientSAM/')
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import zipfile
efficient_sam_vitt_model = build_efficient_sam_vitt()
efficient_sam_vitt_model.eval()

with zipfile.ZipFile("/Users/ahishamm/Documents/projects/efficientSAM/EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
efficient_sam_vits_model = build_efficient_sam_vits()
efficient_sam_vits_model.eval()
torch.jit.save(torch.jit.script(build_efficient_sam_vits()), "torchscripted_model/efficient_sam_vits_torchscript.pt")
print(type(efficient_sam_vits_model))
print(efficient_sam_vits_model)