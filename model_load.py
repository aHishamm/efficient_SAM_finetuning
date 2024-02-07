import torch
from torch.optim import Adam
from monai.losses import DiceFocalLoss
import transformers, datasets 
from tqdm import tqdm
from statistics import mean 
from torch.nn.functional import threshold, normalize
#Loading the torchscript weights into a model 
loaded_model = torch.jit.load("/Users/ahishamm/Documents/projects/efficientSAM/EfficientSAM/torchscripted_model/efficient_sam_vits_torchscript.pt")
loaded_model.eval() 
print(loaded_model.named_parameters())
counter = 0
counter_pruned = 0 
for name, param in loaded_model.named_parameters(): 
    counter += 1
    if name.startswith('image_encoder') or name.startswith('prompt_encoder'): 
        #Freeze the weights of the mask decoder 
        param.requires_grad_(False)
        counter_pruned += 1
    #print(f'Name: {name}')
    
print(f'Number of layers: {counter}')
print(f'Number of layers after pruning: {counter_pruned}')

#Defining the hyperparameters that will be used for finetuning efficientSAM 
segmentation_loss = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
print(f'Details of the segmentation loss \n {segmentation_loss}')
optimizer = Adam(loaded_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)





#Update train_dataloader and import dataset for training the model 
num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.to(device)

loaded_model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = loaded_model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = segmentation_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')