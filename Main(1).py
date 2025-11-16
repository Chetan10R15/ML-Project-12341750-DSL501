"""
Feedback-Sensitive Persona-Aware ESC Chatbot
Student: Rathod Chetankumar A (12341750)

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    AutoTokenizer,
    AutoModel
)
import pandas as pd
import numpy as np
import json
import os
import ast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("UPDATED Feedback-Sensitive Persona-Aware Chatbot")
print("Using emotion_2k_processed.csv Dataset")
print("Student: Rathod Chetankumar A (12341750)")
print("="*70)

# ==================== Configuration ====================
class Config:
    # Model settings
    pal_model = "facebook/blenderbot_small-90M"
    feedback_model = "distilbert-base-uncased"
    
    # Training settings
    batch_size = 2
    learning_rate = 2.5e-5
    num_epochs = 3  # Reduced for faster training
    max_length = 256
    max_samples = 100  # Number of samples to use from dataset
    
    # Feedback types
    feedback_types = ['none', 'correction', 'preference', 'dissatisfaction']
    
    # Paths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data"
    model_dir = "./models"
    results_dir = "./results"

config = Config()

# Create directories
for d in [config.data_dir, config.model_dir, config.results_dir]:
    os.makedirs(d, exist_ok=True)

print(f"\n✓ Device: {config.device}")

# Import the data loading functions from the updated script
exec(open('updated_data_loader.py').read())

# ==================== Feedback Detector ====================
class FeedbackDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.feedback_model)
        self.encoder = AutoModel.from_pretrained(config.feedback_model)
        self.classifier = nn.Linear(768, len(config.feedback_types))
    
    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=128, return_tensors='pt').to(config.device)
        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)
    
    def predict(self, text):
        self.eval()
        with torch.no_grad():
            logits = self.forward(text)
            pred_idx = torch.argmax(logits, dim=1).item()
        return config.feedback_types[pred_idx]

def train_feedback_detector(data):
    print("\n[4/6] Training Feedback Detector...")
    model = FeedbackDetector().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    
    texts = [d['dialogue_history'].split('User:')[-1].strip() for d in data]
    labels = [config.feedback_types.index(d['feedback_type']) for d in data]
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        correct = 0
        
        for i in range(0, len(texts), config.batch_size):
            batch_texts = texts[i:i+config.batch_size]
            batch_labels = torch.tensor(labels[i:i+config.batch_size]).to(config.device)
            
            optimizer.zero_grad()
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
        
        acc = correct / len(labels)
        print(f"  Epoch {epoch+1}/3 - Loss: {total_loss/len(texts):.4f} - Acc: {acc:.3f}")
    
    torch.save(model.state_dict(), os.path.join(config.model_dir, 'feedback_detector.pt'))
    print("✓ Feedback detector trained and saved")
    return model

# ==================== PAL Models ====================
class PALBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(config.pal_model)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(config.pal_model)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss if labels is not None else outputs
    
    def generate_response(self, dialogue, persona_list):
        self.eval()
        persona_text = " ".join(persona_list) if persona_list else ""
        input_text = f"Provide empathetic emotional support. Persona: {persona_text}. Dialogue: {dialogue} Supporter:"
        
        inputs = self.tokenizer(input_text, max_length=config.max_length,
                               truncation=True, return_tensors='pt').to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=80,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class FeedbackSensitivePAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(config.pal_model)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(config.pal_model)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss if labels is not None else outputs
    
    def generate_response(self, dialogue, persona_list, feedback=None):
        self.eval()
        persona_text = " ".join(persona_list) if persona_list else ""
        
        if feedback and feedback != 'none':
            input_text = f"Provide empathetic support. User gave {feedback} feedback. Persona: {persona_text}. Dialogue: {dialogue} Supporter:"
        else:
            input_text = f"Provide empathetic emotional support. Persona: {persona_text}. Dialogue: {dialogue} Supporter:"
        
        inputs = self.tokenizer(input_text, max_length=config.max_length,
                               truncation=True, return_tensors='pt').to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=80,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== Dataset & Training ====================
class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        persona_text = " ".join(item['persona']) if item['persona'] else ""
        input_text = f"Persona: {persona_text}. Dialogue: {item['dialogue_history']}"
        target_text = item['response']
        
        inputs = self.tokenizer(input_text, max_length=config.max_length,
                               padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer(target_text, max_length=80,
                                padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def train_pal_baseline(data):
    print("\n[5/6] Training Baseline PAL...")
    model = PALBaseline().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    dataset = SimpleDataset(data, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"  Epoch {epoch+1} - Avg Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), os.path.join(config.model_dir, 'pal_baseline.pt'))
    print("✓ Baseline PAL trained")
    return model

def train_feedback_pal(data):
    print("\n[6/6] Training Feedback-Sensitive PAL...")
    model = FeedbackSensitivePAL().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    dataset = SimpleDataset(data, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"  Epoch {epoch+1} - Avg Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), os.path.join(config.model_dir, 'feedback_sensitive_pal.pt'))
    print("✓ Feedback-Sensitive PAL trained")
    return model

# ==================== Main Pipeline ====================
def main_pipeline(csv_path='emotion_2k_processed.csv'):
    print("\n" + "="*70)
    print("TRAINING PIPELINE WITH EMOTION DATASET")
    print("="*70)
    
    # Load and process emotion dataset
    data = create_training_data_from_emotion_dataset(
        csv_path=csv_path,
        output_path=os.path.join(config.data_dir, 'processed_training_data.csv'),
        max_samples=config.max_samples
    )
    
    if not data:
        print("✗ Failed to process data")
        return None
    
    # Train models
    feedback_detector = train_feedback_detector(data)
    baseline_pal = train_pal_baseline(data)
    feedback_pal = train_feedback_pal(data)
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModels saved in: {config.model_dir}")
    print(f"Data saved in: {config.data_dir}")
    
    return baseline_pal, feedback_pal, feedback_detector

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'emotion_2k_processed.csv'
    
    try:
        main_pipeline(csv_path)
        print("\n✅ SUCCESS! Models trained on emotion dataset.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nTo run: python updated_main.py [path_to_emotion_2k_processed.csv]")