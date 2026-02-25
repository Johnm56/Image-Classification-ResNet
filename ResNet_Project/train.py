import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader,val_loader, model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        mixup_prob = 0.5  # Probability of applying MixUp
        alpha = 0.8       # MixUp hyperparameter
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        for inputs, labels in tqdm(data_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(args.device), labels.to(args.device)  # Ensure both are on same device
            r = np.random.rand()
            if r < mixup_prob:
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0), device=args.device)

                # Don't index inputs before moving to device
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = labels, labels[index]

                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += acc(outputs, labels)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Train Accuracy: {epoch_acc*100:.2f}%")

        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                val_corrects += acc(outputs, labels)
                val_total += labels.size(0)
        val_acc = val_corrects / val_total
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        scheduler.step()
        
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2025 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict") # Path to the folder where the trained model will be saved
    parser.add_argument('--data', default='data/', type=str, help='data folder') # Path to the dataset to be used for training
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 12
    args.learning_rate = 1e-4
    args.batch_size = 32

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, val_loader = make_data_loader(args, mode='train')
    model = BaseModel()
    model.to(device)

    # Training The Model
    train(args, train_loader,val_loader,model)