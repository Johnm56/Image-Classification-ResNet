import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])

    model.eval()
    
    pbar = tqdm(data_loader)
    for i, (x, y) in enumerate(pbar):
        
        """
        TODO: Implement your testing code. 
              The testing (validation) process is not essential, but it helps to evaluate and improve the model’s generalization performance.
        """

    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2025 DL Term Project')
    parser.add_argument('--model-path', default='checkpoints/model.pth', help="Model's state_dict")  # # Path to the folder where the trained model is saved
    parser.add_argument('--data', default='data/', type=str, help='data folder')  # Path to the dataset to be used for testing
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # hyperparameters
    args.batch_size = 4
    
    # Make Data loader and Model
    test_loader = make_data_loader(args)

    # instantiate model
    model = BaseModel()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Test The Model
    pred, true = test(args, test_loader, model)
        
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))

    