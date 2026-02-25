import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import BaseModel
from tqdm import tqdm
from utils._utils import make_data_loader



def inference(args, data_loader, model):
    """ model inference """

    model.eval()
    preds = []
    
    with torch.no_grad():
      for inputs, labels in tqdm(test_loader):  
        inputs, labels = inputs.to(device), labels.to(device)
           
        y_hat = model(inputs)
            
        y_hat.argmax()

        _, predicted = torch.max(y_hat, 1)
        preds.extend(map(lambda t: t.item(), predicted))

    return preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2025 DL Term Project')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Model's state_dict") 
    parser.add_argument('--batch-size', default=16, help='test loader batch size')
    parser.add_argument('--data', default='data/', help='image dataset directory') 

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    model = BaseModel()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # load test (or validation) dataset in image folder
    test_loader = make_data_loader(args, mode='test')

    # write model inference
    preds = inference(args, test_loader, model)
        
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))