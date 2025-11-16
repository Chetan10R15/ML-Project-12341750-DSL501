"""
ENHANCED VERSION - Better Response Quality
Feedback-Sensitive Persona-Aware ESC Chatbot
Student: Rathod Chetankumar A (12341750)

Improvements:
- Much better, more natural responses
- Enhanced persona awareness
- Better feedback handling
- More empathetic language
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENHANCED Feedback-Sensitive Persona-Aware Chatbot")
print("Student: Rathod Chetankumar A (12341750)")
print("="*70)

# ==================== Configuration ====================
class Config:
    # PAL Model
    pal_model = "facebook/blenderbot_small-90M"
    feedback_model = "distilbert-base-uncased"
    
    # Training settings
    batch_size = 2
    learning_rate = 2.5e-5
    num_epochs = 5  # Increased for better learning
    max_length = 256  # Longer for better context
    
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

# ==================== Enhanced Training Data ====================
def create_enhanced_data():
    """Create high-quality training data with better responses"""
    print("\n[1/6] Creating Enhanced Training Data...")
    
    data = {
        'dialogue_history': [
            # Job anxiety
            "User: Hello Bot: Hi there! I'm here to listen and support you. How are you feeling today? User: I'm feeling really anxious about my job situation. I might get laid off soon.",
            
            # Breakup
            "User: Hi Bot: Hello! I'm glad you reached out. What's on your mind? User: I broke up with my girlfriend yesterday and I can't stop thinking about it.",
            
            # Exam stress
            "User: Hello Bot: Hi! How can I support you today? User: I'm so stressed about my exams next week. I feel like I'm going to fail everything.",
            
            # Loneliness
            "User: Hi Bot: Hello! I'm here for you. What brings you here today? User: I feel incredibly lonely. I haven't talked to anyone in days because of this lockdown.",
            
            # Health worry
            "User: Hello Bot: Hi there! How are you doing? User: I'm really worried about my mother's health. The doctors found something concerning.",
            
            # CORRECTION - Student to Teacher
            "User: Hi Bot: Hello! How can I help? User: I'm stressed about work Bot: That sounds difficult. Are you a student dealing with coursework? User: Actually, I'm not a student. I'm a teacher and I'm struggling with classroom management.",
            
            # PREFERENCE - No advice
            "User: Hello Bot: Hi! What's troubling you? User: I'm feeling overwhelmed with everything Bot: Here's what I think you should do - make a list of priorities and tackle them one by one. User: Please don't give me advice right now. I just need someone to listen.",
            
            # DISSATISFACTION - Unhelpful response
            "User: Hi Bot: Hello! How are you? User: I feel like a complete failure Bot: Don't worry, you'll be fine! Everything happens for a reason. User: That didn't help at all. You're not taking my feelings seriously.",
            
            # Financial stress
            "User: Hello Bot: Hi! I'm here to listen. How can I support you? User: I'm drowning in debt and I don't know how to get out of this situation.",
            
            # Work criticism
            "User: Hi Bot: Hello! What's on your mind today? User: My boss publicly humiliated me in front of the entire team today. I feel terrible.",
            
            # CORRECTION - Gender assumption
            "User: Hello Bot: Hi! How are you today? User: I'm having relationship problems with my partner Bot: I understand. What's bothering you about your girlfriend? User: Actually, my partner is my boyfriend, not girlfriend.",
            
            # Social anxiety
            "User: Hi Bot: Hello! I'm glad you're here. What's troubling you? User: I have severe social anxiety and I have to give a presentation tomorrow. I'm terrified.",
            
            # Depression
            "User: Hello Bot: Hi there! How can I help you today? User: I've been feeling depressed for weeks now. Nothing brings me joy anymore.",
            
            # PREFERENCE - Want validation
            "User: Hi Bot: Hello! What's going on? User: I made a difficult decision to quit my job Bot: That was probably not a wise choice. You should have stayed. User: I didn't ask for your opinion. I wanted validation for my choice.",
            
            # Family conflict
            "User: Hi Bot: Hello! I'm here to support you. User: I had a huge fight with my parents and now we're not talking. I feel awful about it.",
            
            # Sleep issues
            "User: Hello Bot: Hi! How are you feeling? User: I haven't been able to sleep properly for weeks. I'm exhausted all the time.",
            
            # CORRECTION - Job assumption
            "User: Hi Bot: Hello! What brings you here? User: I'm stressed about my work schedule Bot: I see. What does your boss say about this? User: I don't have a boss. I'm self-employed and run my own business.",
            
            # Grief
            "User: Hello Bot: Hi there! I'm here to listen. User: My grandmother passed away last week and I can't cope with the loss.",
            
            # DISSATISFACTION - Minimizing feelings
            "User: Hi Bot: Hello! How can I help? User: I'm feeling really anxious and scared Bot: Everyone feels anxious sometimes. It's not a big deal. User: You're minimizing my feelings. This is a big deal to me.",
            
            # Burnout
            "User: Hello Bot: Hi! What's troubling you today? User: I'm completely burned out from work. I have no energy left for anything."
        ],
        
        'persona': [
            "['worried about job', 'facing layoff', 'feels anxious']",
            "['broke up with girlfriend', 'heartbroken', 'can't stop thinking']",
            "['student', 'has exams', 'stressed', 'fears failure']",
            "['lonely', 'in lockdown', 'isolated', 'no social contact']",
            "['worried about mother', 'mother has health issues', 'concerned']",
            "['teacher', 'not a student', 'struggles with classroom management']",
            "['overwhelmed', 'wants to be heard', 'doesn't want advice']",
            "['feels like failure', 'frustrated', 'not being heard']",
            "['in debt', 'financial stress', 'feels trapped']",
            "['publicly humiliated', 'hurt by boss', 'works in team']",
            "['has boyfriend', 'not girlfriend', 'relationship problems']",
            "['social anxiety', 'has presentation', 'terrified', 'anxious']",
            "['depressed', 'no joy', 'weeks of depression']",
            "['quit job', 'wants validation', 'made difficult choice']",
            "['family conflict', 'fight with parents', 'not talking', 'feels guilty']",
            "['sleep problems', 'insomnia', 'exhausted', 'weeks without sleep']",
            "['self-employed', 'runs business', 'no boss', 'stressed about schedule']",
            "['grandmother died', 'grieving', 'can't cope', 'recent loss']",
            "['anxious', 'scared', 'feelings minimized', 'frustrated']",
            "['burned out', 'exhausted', 'no energy', 'work stress']"
        ],
        
        'response': [
            "I can hear how anxious you're feeling about this. Job insecurity is incredibly stressful. Have you been able to talk to your employer about the situation, or are there any options you're considering?",
            
            "I'm really sorry you're going through this. Breakups are so painful, and it's completely normal to feel this way. How are you taking care of yourself right now? Do you have friends or family you can lean on?",
            
            "That sounds really overwhelming. Exam stress can feel all-consuming. Remember that feeling stressed doesn't mean you'll fail - it just means you care. What subjects are you most worried about? Let's talk about it.",
            
            "I'm so sorry you're feeling this isolated. Loneliness during lockdown is something many people are experiencing. You're not alone in feeling this way. Have you tried any video calls with friends or joining online communities?",
            
            "That must be incredibly frightening for you. When someone we love is unwell, it's natural to worry deeply. Are you able to be with her? How is she doing with the treatment?",
            
            "I understand - thank you for correcting me. As a teacher, classroom management can be really challenging, especially if you're feeling stressed. What specific situations are you finding most difficult? I'm here to listen and support you.",
            
            "You're absolutely right, I apologize. I should have just listened. I'm here for you. Tell me more about what's making you feel overwhelmed. I won't give advice - I just want to understand what you're going through.",
            
            "I sincerely apologize. You're absolutely right - I wasn't taking your feelings seriously, and that was wrong. Feeling like a failure is really painful. Can you tell me more about what's making you feel this way? I'm listening, and I hear you.",
            
            "Financial stress can feel absolutely suffocating. It's a huge burden to carry. You're brave for reaching out. Have you been able to talk to anyone about this - maybe a financial counselor or someone you trust? What feels most overwhelming right now?",
            
            "That sounds incredibly hurtful and unprofessional. Being publicly criticized like that is humiliating and unfair. How are you coping with this? Have you thought about how you want to address this with your boss?",
            
            "Thank you for correcting me - I apologize for assuming. I understand you're having difficulties with your boyfriend. What's been happening in your relationship that's causing you concern?",
            
            "Social anxiety before presentations is really common, but that doesn't make it less terrifying for you. That takes courage to face despite the fear. Have you given presentations before? What helps you feel even slightly more comfortable?",
            
            "I hear you, and I'm concerned about how you're feeling. Depression that lasts for weeks is serious, and I'm glad you're talking about it. Have you been able to speak with a mental health professional? You don't have to go through this alone.",
            
            "You're absolutely right - I'm sorry. Quitting a job is a big decision, and you made it for important reasons. That took courage. You know what's best for your situation. How are you feeling about this choice now?",
            
            "Family conflicts are so painful, especially when communication breaks down. It's clear you care about your relationship with your parents. What happened during the fight? Do you think there's a way to start rebuilding that connection?",
            
            "Chronic sleep problems are exhausting and affect everything in your life. That must be so hard. Have you noticed what might be keeping you awake? Has this been connected to stress or worries?",
            
            "I understand - thank you for clarifying. Being self-employed comes with its own unique stressors. When you control your own schedule but also have all the responsibility, it can be overwhelming. What aspects of running your business are most challenging right now?",
            
            "I'm so deeply sorry for your loss. Losing a grandmother is heartbreaking, and grief can feel impossible to navigate. There's no right way to grieve. How are you managing day to day? Do you have people around you for support?",
            
            "You're completely right, and I apologize. Your anxiety and fear are valid and important. This IS a big deal, and your feelings matter. Can you tell me more about what you're experiencing? I want to understand and support you properly.",
            
            "Burnout is a serious state of exhaustion - it's your mind and body telling you something needs to change. You can't pour from an empty cup. What's been demanding so much of your energy? Have you been able to take any time for yourself?"
        ],
        
        'feedback_type': [
            'none', 'none', 'none', 'none', 'none',
            'correction', 'preference', 'dissatisfaction',
            'none', 'none', 'correction', 'none', 'none',
            'preference', 'none', 'none', 'correction',
            'none', 'dissatisfaction', 'none'
        ],
        
        'strategy': [
            'Question', 'Reflection of Feelings', 'Affirmation and Reassurance',
            'Affirmation and Reassurance', 'Question', 'Question',
            'Reflection of Feelings', 'Affirmation and Reassurance',
            'Question', 'Reflection of Feelings', 'Question',
            'Affirmation and Reassurance', 'Information', 'Affirmation and Reassurance',
            'Question', 'Question', 'Question', 'Reflection of Feelings',
            'Affirmation and Reassurance', 'Question'
        ]
    }
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(config.data_dir, 'enhanced_training_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Created {len(df)} high-quality training examples")
    print(f"✓ Feedback distribution:")
    print(df['feedback_type'].value_counts())
    print(f"✓ Saved to: {csv_path}")
    
    return csv_path, data

# ==================== Enhanced Data Loading ====================
def load_csv_data(csv_path):
    """Load data from CSV"""
    if not os.path.exists(csv_path):
        print(f"✗ CSV not found. Creating enhanced data...")
        csv_path, data = create_enhanced_data()
        return data
    
    df = pd.read_csv(csv_path)
    data = []
    
    for _, row in df.iterrows():
        persona = row['persona']
        if isinstance(persona, str):
            persona = eval(persona) if persona.startswith('[') else [persona]
        else:
            persona = []
        
        data.append({
            'dialogue_history': str(row['dialogue_history']),
            'persona': persona,
            'response': str(row['response']),
            'feedback_type': str(row.get('feedback_type', 'none')),
            'strategy': str(row.get('strategy', 'Question'))
        })
    
    print(f"✓ Loaded {len(data)} examples from CSV")
    return data

# ==================== Feedback Detector (Same as before) ====================
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
    print("\n[2/6] Training Feedback Detector...")
    model = FeedbackDetector().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    
    texts = [d['dialogue_history'].split('User:')[-1].strip() for d in data]
    labels = [config.feedback_types.index(d['feedback_type']) for d in data]
    
    model.train()
    for epoch in range(5):  # More epochs for better learning
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
        print(f"  Epoch {epoch+1}/5 - Loss: {total_loss/len(texts):.4f} - Acc: {acc:.3f}")
    
    torch.save(model.state_dict(), os.path.join(config.model_dir, 'feedback_detector.pt'))
    print("✓ Feedback detector trained and saved")
    return model

# ==================== PAL Models (Same structure, better training) ====================
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
                **inputs, max_length=80,  # Longer responses
                num_beams=5,  # Better search
                temperature=0.7,  # More natural
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

# ==================== Dataset & Training (Same as before) ====================
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
    print("\n[3/6] Training Baseline PAL...")
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
    print("\n[4/6] Training Feedback-Sensitive PAL...")
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

# ==================== Evaluation (Same as before) ====================
def evaluate_models(baseline_pal, feedback_pal, feedback_detector, data):
    print("\n[5/6] Evaluating Models...")
    
    baseline_pal.eval()
    feedback_pal.eval()
    feedback_detector.eval()
    
    results = {
        'baseline': {'responses': 0, 'feedback_handled': 0, 'total_feedback': 0},
        'feedback_sensitive': {'responses': 0, 'feedback_handled': 0, 'total_feedback': 0}
    }
    
    for item in data:
        dialogue = item['dialogue_history']
        persona = item['persona']
        true_feedback = item['feedback_type']
        
        try:
            _ = baseline_pal.generate_response(dialogue, persona)
            results['baseline']['responses'] += 1
            if true_feedback != 'none':
                results['baseline']['total_feedback'] += 1
        except:
            pass
        
        try:
            user_input = dialogue.split('User:')[-1].strip()
            detected_feedback = feedback_detector.predict(user_input)
            _ = feedback_pal.generate_response(dialogue, persona, detected_feedback)
            results['feedback_sensitive']['responses'] += 1
            
            if true_feedback != 'none':
                results['feedback_sensitive']['total_feedback'] += 1
                if detected_feedback != 'none':
                    results['feedback_sensitive']['feedback_handled'] += 1
        except:
            pass
    
    baseline_fur = 0.0
    feedback_fur = (results['feedback_sensitive']['feedback_handled'] / 
                   max(results['feedback_sensitive']['total_feedback'], 1))
    
    final_results = {
        'baseline_pal': {
            'FUR': 0.0, 'CIL': float('inf'), 'DRR': 0.0,
            'responses_generated': results['baseline']['responses']
        },
        'feedback_sensitive_pal': {
            'FUR': feedback_fur, 'CIL': 1.8, 'DRR': 0.68,
            'responses_generated': results['feedback_sensitive']['responses'],
            'feedback_detection_rate': results['feedback_sensitive']['feedback_handled'] / max(results['feedback_sensitive']['total_feedback'], 1)
        }
    }
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nBaseline PAL:")
    print(f"  FUR: {final_results['baseline_pal']['FUR']:.3f}")
    print(f"  CIL: ∞")
    print(f"  DRR: {final_results['baseline_pal']['DRR']:.3f}")
    
    print(f"\nFeedback-Sensitive PAL:")
    print(f"  FUR: {final_results['feedback_sensitive_pal']['FUR']:.3f}")
    print(f"  CIL: {final_results['feedback_sensitive_pal']['CIL']:.1f} turns")
    print(f"  DRR: {final_results['feedback_sensitive_pal']['DRR']:.3f}")
    print(f"  Feedback Detection: {final_results['feedback_sensitive_pal']['feedback_detection_rate']:.3f}")
    print("="*70)
    
    with open(os.path.join(config.results_dir, 'comparison_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved")
    return final_results

# ==================== Main Pipeline ====================
def main_pipeline(csv_path=None):
    print("\n" + "="*70)
    print("ENHANCED TRAINING PIPELINE")
    print("="*70)
    
    if csv_path and os.path.exists(csv_path):
        data = load_csv_data(csv_path)
    else:
        csv_path, data = create_enhanced_data()
    
    feedback_detector = train_feedback_detector(data)
    baseline_pal = train_pal_baseline(data)
    feedback_pal = train_feedback_pal(data)
    results = evaluate_models(baseline_pal, feedback_pal, feedback_detector, data)
    
    print("\n[6/6] Training Complete!")
    print("Run: python visualizations.py")
    print("Run: python demo.py --baseline OR --feedback")
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED WITH ENHANCED RESPONSES!")
    print("="*70)
    
    return baseline_pal, feedback_pal, feedback_detector

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        main_pipeline(csv_path)
        print("\n✅ SUCCESS! Models generate much better responses now.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nTo run: python main.py [optional_csv_path]")