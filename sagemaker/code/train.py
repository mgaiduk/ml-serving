import argparse
from datetime import datetime, timedelta
import os
import pandas as pd
import snowflake.connector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

from snowflake_utils import connect

from typing import Dict

from model import CollaborativeFilteringModel, my_hash, save_model

def collect_dataset(ctx: snowflake.connector.SnowflakeConnection, input: str) -> pd.DataFrame:
    # train on a 7-day lookback window
    filter_time = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    # Collect dataset
    sql = f"""
    select * from {input}
    where min_timestamp >= '{filter_time}'
    """
    df = pd.read_sql(sql, ctx)
    return df

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = torch.tensor(dataframe[['USERID', 'MEDIAID', 'MEDIATAKENBYID']].values, dtype=torch.int32)
        self.labels = torch.tensor(dataframe[['TIMESPENT', 'NREACTIONS']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        has_reactions = (labels[1] > 0).float()
        return {
            'features': {'USERID': features[0], 'MEDIAID': features[1], 'MEDIATAKENBYID': features[2]},
            'labels': {'TIMESPENT': labels[0], 'HAS_REACTIONS': has_reactions}
        }


def dict_to_device(dic: Dict, device: str) -> Dict:
    return {k: v.to(device) for k, v in dic.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                    help='Input dataset snowflake table')
    parser.add_argument("--profile-name", type=str, required=False, help="sso profile for aws cli")
    parser.add_argument("--user-dict-size", type=int, default=50000, help="Total number of unique user embeddings in CF model")
    parser.add_argument("--post-dict-size", type=int, default=1800, help="Total number of unique post embeddings in CF model")
    parser.add_argument("--embedding-dim", type=int, default=4, help="embedding size")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'], help="Model dir to save model from within sagemaker pipeline")
    args = parser.parse_args()

    secret_id = os.environ["SECRET_ID"]
    account = os.environ["SF_ACCOUNT"]
    warehouse = os.environ["SF_WAREHOUSE"]
    database = os.environ["SF_DATABASE"].upper()
    schema = os.environ["SF_SCHEMA"].upper()
    region = os.environ["AWS_REGION"]

    protocol = "https"
    ctx = connect(secret_id, account, warehouse, database, schema, protocol, region, profile_name=args.profile_name)
    df = collect_dataset(ctx, input=args.input)
    df[['USERID', 'MEDIAID', 'MEDIATAKENBYID']] = df[['USERID', 'MEDIAID', 'MEDIATAKENBYID']].applymap(my_hash)
    

    yesterday = datetime.now() - timedelta(days=1)
    val_df = df[df["MIN_TIMESTAMP"] > yesterday.strftime('%Y-%m-%d')]
    train_df = df[df["MIN_TIMESTAMP"] <= yesterday.strftime('%Y-%m-%d')]
    print(f"train_df shape: {train_df.shape}, val df shape: {val_df.shape}")
    print(f"Cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_dt = CustomDataset(train_df)
    val_dt = CustomDataset(val_df)
    batch_size = 4096  # Adjust based on your needs
    shuffle = True  # Shuffle the data
    num_workers = 2
    train_dataloader = DataLoader(train_dt, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dt, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # train lööp
    print(f'Unique number of users: {len(df["USERID"].unique())}, posts: {len(df["MEDIAID"].unique())}')
    num_users = args.user_dict_size
    num_posts = args.post_dict_size
    embedding_dim = args.embedding_dim
    model = CollaborativeFilteringModel(num_users=num_users, num_posts=num_posts, embedding_dim=embedding_dim).to(device)

    criterion = nn.BCELoss()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = args.num_epochs


    for epoch in range(num_epochs):
        for batch in train_dataloader:
            features = batch['features']
            labels = batch['labels']
            features = dict_to_device(features, device)
            labels = dict_to_device(labels, device)
            optimizer.zero_grad()
            outputs = model(features["USERID"], features["MEDIAID"])
            targets = labels["HAS_REACTIONS"]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, train loss: {loss.item()}')

        # Validation phase
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for batch in val_dataloader:
                features = batch['features']
                labels = batch['labels']
                features = dict_to_device(features, device)
                labels = dict_to_device(labels, device)
                outputs = model(features["USERID"], features["MEDIAID"])
                targets = labels["HAS_REACTIONS"]
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())  # Collect all targets
                all_outputs.extend(outputs.cpu().numpy())  # Collect all outputs
        auc_roc_score = roc_auc_score(all_targets, all_outputs)
        
        # Print average validation loss per epoch
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_dataloader)}, AUC-ROC: {auc_roc_score}')

    save_model(model, os.path.join(args.model_dir, 'model.pth'))
    

if __name__ == "__main__":
    main()