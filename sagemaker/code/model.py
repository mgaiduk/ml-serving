import os
import json
import torch
import torch.nn as nn
import xxhash

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_posts, embedding_dim):
        super(CollaborativeFilteringModel, self).__init__()
        self.num_users = num_users
        self.num_posts = num_posts
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.post_embedding = nn.Embedding(num_posts, embedding_dim)
        
    def forward(self, user_ids, post_ids):
        user_ids %= self.num_users
        post_ids %= self.num_posts
        user_embedded = self.user_embedding(user_ids)
        post_embedded = self.post_embedding(post_ids)
        interaction = (user_embedded * post_embedded).sum(dim=1)
        prediction = torch.sigmoid(interaction)
        return prediction

def my_hash(s: str) -> int:
    return xxhash.xxh32(s).intdigest()

def save_model(model, path="model.pth"):
    # Create a dictionary to hold the model state and configuration
    model_info = {
        "state_dict": model.state_dict(),
        "num_users": model.num_users,
        "num_posts": model.num_posts,
        "embedding_dim": model.user_embedding.embedding_dim  # Assuming both embeddings have the same dim
    }
    # Save the dictionary to a file
    torch.save(model_info, path)

def load_model(path="model.pth"):
    # Load the dictionary containing the model information
    model_info = torch.load(path, map_location=torch.device('cpu'))
    # Recreate the model using the saved configuration
    model = CollaborativeFilteringModel(
        num_users=model_info["num_users"],
        num_posts=model_info["num_posts"],
        embedding_dim=model_info["embedding_dim"]
    )
    # Load the model state
    model.load_state_dict(model_info["state_dict"])
    return model


# Sagemaker inference fns

