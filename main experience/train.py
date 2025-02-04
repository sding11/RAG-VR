import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_scheduler
import random

# Set random seed
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
default_learning_rate = 1e-5  # Learning rate, controls the step size of weight updates during training
default_batch_size = 8  # Number of samples per batch during training
default_epochs = 6  # Number of times the entire dataset is traversed during training
default_max_length = 128  # Maximum length of input text sequences
default_scheduler_warmup_steps = 100  
hard_negative_weight = 2.0  # Weight for hard negative samples, increasing their impact in the loss function

# Dataset class
class QADataset(Dataset):
    def __init__(self, positive_samples, negative_samples, hard_negative_samples, tokenizer):
        self.samples = []

        # Add positive samples (label 1)
        for question, answer in positive_samples:
            self.samples.append((question, answer, 1, 0))  # is_hard_negative flag set to 0

        # Add random negative samples (label 0)
        for question, answer in negative_samples:
            self.samples.append((question, answer, 0, 0))  # is_hard_negative flag set to 0

        # Add hard negative samples (label 0)
        for question, answer in hard_negative_samples:
            self.samples.append((question, answer, 0, 1))  # is_hard_negative flag set to 1

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer, label, is_hard_negative = self.samples[idx]
        query_encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=default_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        doc_encoding = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=default_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "doc_input_ids": doc_encoding["input_ids"].squeeze(0),
            "doc_attention_mask": doc_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
            "is_hard_negative": torch.tensor(is_hard_negative, dtype=torch.float)
        }

# Dual-tower model
class DualTowerModel(nn.Module):
    def __init__(self):
        super(DualTowerModel, self).__init__()
        self.query_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.doc_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(self.query_encoder.config.hidden_size, 128)  # Reduce dimension to 128

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        # Query tower
        query_hidden = self.query_encoder(query_input_ids, attention_mask=query_attention_mask)[0][:, 0, :]
        query_embeds = F.normalize(self.fc(query_hidden), p=2, dim=1)

        # Document tower
        doc_hidden = self.doc_encoder(doc_input_ids, attention_mask=doc_attention_mask)[0][:, 0, :]
        doc_embeds = F.normalize(self.fc(doc_hidden), p=2, dim=1)

        return query_embeds, doc_embeds

# Weighted cosine similarity loss
class WeightedCosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.2, hard_negative_weight=2.0):
        super(WeightedCosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight

    def forward(self, query_embeds, doc_embeds, labels, is_hard_negative):
        cos_sim = F.cosine_similarity(query_embeds, doc_embeds, dim=1)
        
        # Compute base loss for positive and negative samples
        positive_loss = labels * (1 - cos_sim)
        negative_loss = (1 - labels) * F.relu(cos_sim - self.margin)
        
        # Apply weight to hard negative samples
        weighted_negative_loss = negative_loss * (1 + is_hard_negative * (self.hard_negative_weight - 1))
        
        return torch.mean(positive_loss + weighted_negative_loss)

# Training function
def train_model(model, train_loader, device, epochs=3):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=default_learning_rate, weight_decay=0.01)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=default_scheduler_warmup_steps, num_training_steps=len(train_loader) * epochs
    )
    loss_fn = WeightedCosineSimilarityLoss(margin=0.2, hard_negative_weight=hard_negative_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            doc_input_ids = batch["doc_input_ids"].to(device)
            doc_attention_mask = batch["doc_attention_mask"].to(device)
            labels = batch["label"].to(device)
            is_hard_negative = batch["is_hard_negative"].to(device)

            optimizer.zero_grad()

            # Forward pass
            query_embeds, doc_embeds = model(query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask)
            loss = loss_fn(query_embeds, doc_embeds, labels, is_hard_negative)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Model saving function
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Main function
def train_and_save_all_models(input_dir):
    for scene_folder in os.listdir(input_dir):
        scene_path = os.path.join(input_dir, scene_folder)
        if os.path.isdir(scene_path):
            print(f"Processing scene: {scene_folder}")
            
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            positive_samples, negative_samples, hard_negative_samples = load_samples_from_folder(scene_path)
            negative_samples = random.sample(negative_samples, len(positive_samples))

            num_samples = min(len(hard_negative_samples), len(positive_samples))
            hard_negative_samples = random.sample(hard_negative_samples, num_samples)

            dataset = QADataset(positive_samples, negative_samples, hard_negative_samples, tokenizer)
            train_loader = DataLoader(dataset, batch_size=default_batch_size, shuffle=True)
            
            model = DualTowerModel()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_model(model, train_loader, device, epochs=default_epochs)
            
            save_path = os.path.join(scene_path, f"{scene_folder}_dual_tower_model.pth")
            save_model(model, save_path)

# Run main function
input_dir = "./train"
train_and_save_all_models(input_dir)
