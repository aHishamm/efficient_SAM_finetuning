import torch
from torch.optim import Adam
from monai.losses import DiceFocalLoss
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers, datasets 
from transformers import SamProcessor
from tqdm import tqdm
from statistics import mean 
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize

#SAMDataset class for preprocessing input dataset for training 
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


#Loading the dataset for training 
ds = load_dataset('ahishamm/isic_masks',split='train') 



#Import SAMProcessor processor information 
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
print(processor)

train_dataset = SAMDataset(dataset=ds,processor=processor)
train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)

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