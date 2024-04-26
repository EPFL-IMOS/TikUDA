from pathlib import Path
import argparse
import time

import torch
import random

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import *
from model import *
from training_testing import *

import warnings
warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser(description="UDAR for Spatial-Temporal GNN in IoT Air Quality Multisensor Fusion", 
                                     add_help=False)

    # Data
    parser.add_argument("--data-dir", type=str, default="data/",
                        help='Path to the data')
    
    parser.add_argument("--source_dataset", type=str, default="R-212.csv",
                        help='Source dataset file')
    parser.add_argument("--target_dataset", type=str, default="R-69.csv",
                        help='Target dataset file')
    
    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="exp/",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')

    # Optim
    parser.add_argument("--seed", type=int, default=42,
                        help='Seed for experiments')
    parser.add_argument("--epochs", type=int, default=150,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Batch size')
    parser.add_argument("--base-lr", type=float, default=3e-4,
                        help='Base Learning rate')

    # Running
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
                        
    return parser

def mains(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    source_file = args.data_dir + args.source_dataset
    target_file = args.data_dir + args.target_dataset

    x_source, y_source, x_target, y_target, y_scaler = training_testing_dataframes(source_file,target_file)

    x_source, y_source, x_target, y_target = torch.from_numpy(x_source), torch.from_numpy(y_source),torch.from_numpy(x_target), torch.from_numpy(y_target)

    #Create Dataset
    data_x_y_source = torch.utils.data.TensorDataset(x_source, y_source)
    data_x_y_target = torch.utils.data.TensorDataset(x_target, y_target)

    #Create Dataloader
    train_dl_source = torch.utils.data.DataLoader(data_x_y_source, batch_size = args.batch_size, shuffle = False, drop_last = True, num_workers = args.num_workers)
    test_dl_source = torch.utils.data.DataLoader(data_x_y_source, batch_size= 1, shuffle = False)

    train_dl_target = torch.utils.data.DataLoader(data_x_y_target, batch_size = args.batch_size, shuffle = False, drop_last = True, num_workers = args.num_workers)
    test_dl_target = torch.utils.data.DataLoader(data_x_y_target, batch_size= 1, shuffle = False)
    
    #define Model
    model = FGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
        
    model.train()

    for e in range(10):#args.epochs):
        t0 = time.time()
        train_loss = training_loop(model, optimizer, train_dl_source, train_dl_target, e, args.epochs, device)
        t1 = time.time()

        print('Time of one iteration : ',t1-t0)
        
        # Print progress at every 5 epochs
        if (e + 1) % 5 == 0:
            print(f"Epoch: {e+1}")
            print('Training Loss:', train_loss)
        
    prediction, ground_truth = testing_loop(model,test_dl_target,device)

    print(f"NRMSE: {mean_squared_error(prediction,ground_truth, squared = False)}")
    print(f"NMAE: {mean_absolute_error(prediction,ground_truth)}")
        
        
    gt_denormalized = y_scaler.inverse_transform(ground_truth.reshape(-1, 1))
    pred_denormalized = y_scaler.inverse_transform(prediction.reshape(-1, 1))
    pred_denormalized = pred_denormalized.squeeze()


    print(f"RMSE: {mean_squared_error(pred_denormalized,gt_denormalized, squared = False)}")
    print(f"MAE: {mean_absolute_error(pred_denormalized,gt_denormalized)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_arguments()])
    args = parser.parse_args()
    
    seed_everything(args.seed)
    mains(args)