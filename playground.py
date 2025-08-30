from src.reloc3r_variants.data.datasets.megadepth import MegaDepth
from src.reloc3r_variants.models.Reloc3rWithDiffusionHead import Reloc3rWithDiffusionHead

from torch.utils.data import DataLoader

import torch

##### Initialize Model #####
device = torch.device("cuda:0")
model = Reloc3rWithDiffusionHead().to(device)


##### Initialize Data Loader #####
data_set = 500 @ MegaDepth(split='train', resolution=(512,384))
data_loader = DataLoader(data_set, batch_size=16, shuffle=True, drop_last=True)
data_loader.data_sampler = data_loader.dataset.make_sampler(batch_size=16, shuffle=True, world_size=1, rank=0, drop_last=True)

data_loader.dataset.set_epoch(0)
data_loader.data_sampler.set_epoch(0)
with torch.no_grad():
    for batch in data_loader:
        for k in batch[0]:
            if isinstance(batch[0][k], torch.Tensor):
                batch[0][k] = batch[0][k].to(device)
                batch[1][k] = batch[1][k].to(device)
        out = model(batch[0], batch[1])
        break
