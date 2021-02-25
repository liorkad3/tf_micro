import torch
from torch.utils.tensorboard import SummaryWriter

log_dir = 'logs/'

writer = SummaryWriter(log_dir)

for i in range(1000):
    print(f'logging: {i}')
    writer.add_scalar('test', i, i)