import torch
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import os
import sys

# Import from main
sys.path.append('.')

class Config:
    pal_model = "facebook/blenderbot_small-90M"
    feedback_model = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "./models"
    feedback_types = ['none', 'correction', 'preference', 'dissatisfaction']

config = Config()

# Load models
class FeedbackDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.feedback_model)
        self.encoder = AutoModel.from_pretrained(config.feedback_model)
        self.classifier = nn.Linear(768, 4)
    
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

class PALBaseline:
    def __init__(self):
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(config.pal_model)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(config.pal_model)
        self.model.to(config.device)
        self.model.eval()
        self.history = []
        self.persona = []
    
    def chat(self, user_input):
        self.history.append(f"User: {user_input}")
        dialogue = " ".join(self.history[-3:])
        persona_text = " ".join(self.persona) if self.persona else ""
        
        input_text = f"Persona: {persona_text}. Dialogue: {dialogue}"
        inputs = self.tokenizer(input_text, max_length=128, truncation=True,
                               return_tensors='pt').to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=60, num_beams=4)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(f"Bot: {response}")
        
        return response, "none"  # Baseline doesn't detect feedback
    
    def reset(self):
        self.history = []
        self.persona = []

class FeedbackSensitivePAL:
    def __init__(self, feedback_detector):
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(config.pal_model)
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(config.pal_model)
        self.model.to(config.device)
        self.model.eval()
        self.detector = feedback_detector
        self.history = []
        self.persona = []
    
    def chat(self, user_input):
        self.history.append(f"User: {user_input}")
        
        # Detect feedback
        feedback = self.detector.predict(user_input)
        
        # Handle feedback
        acknowledgment = ""
        if feedback == 'correction':
            acknowledgment = "I understand, let me correct that. "
            if "i'm" in user_input.lower() or "i am" in user_input.lower():
                words = user_input.lower().split()
                for i, w in enumerate(words):
                    if w in ['i\'m', 'i', 'am']:
                        info = " ".join(words[i+1:i+3])
                        self.persona.append(f"I am {info}")
                        break
        elif feedback == 'preference':
            acknowledgment = "I'll keep that in mind. "
        elif feedback == 'dissatisfaction':
            acknowledgment = "I apologize, let me try differently. "
        
        # Generate response
        dialogue = " ".join(self.history[-3:])
        persona_text = " ".join(self.persona) if self.persona else ""
        
        if feedback != 'none':
            input_text = f"Feedback: {feedback}. Persona: {persona_text}. Dialogue: {dialogue}"
        else:
            input_text = f"Persona: {persona_text}. Dialogue: {dialogue}"
        
        inputs = self.tokenizer(input_text, max_length=128, truncation=True,
                               return_tensors='pt').to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=60, num_beams=4)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = acknowledgment + response
        
        self.history.append(f"Bot: {response}")
        
        return response, feedback
    
    def reset(self):
        self.history = []
        self.persona = []

def demo_baseline():
    """Demo baseline PAL"""
    print("\n" + "="*70)
    print("BASELINE PAL DEMO (No Feedback Handling)")
    print("="*70)
    print("\nCommands: 'reset', 'quit'")
    print("="*70 + "\n")
    
    chatbot = PALBaseline()
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            chatbot.reset()
            print("Reset!\n")
            continue
        
        response, _ = chatbot.chat(user_input)
        print(f"Bot: {response}\n")

def demo_feedback():
    """Demo feedback-sensitive PAL"""
    print("\n" + "="*70)
    print("FEEDBACK-SENSITIVE PAL DEMO (With Feedback Handling)")
    print("="*70)
    print("\nCommands: 'persona', 'reset', 'quit'")
    print("="*70 + "\n")
    
    # Load feedback detector
    print("Loading models...")
    try:
        detector = FeedbackDetector()
        detector.load_state_dict(
            torch.load(os.path.join(config.model_dir, 'feedback_detector.pt'),
                      map_location=config.device)
        )
        detector.eval()
        print("✓ Models loaded\n")
    except:
        print("⚠️  Trained models not found. Using untrained detector.\n")
        detector = FeedbackDetector()
        detector.eval()
    
    chatbot = FeedbackSensitivePAL(detector)
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            chatbot.reset()
            print("Reset!\n")
            continue
        if user_input.lower() == 'persona':
            print(f"Persona: {chatbot.persona if chatbot.persona else '(empty)'}\n")
            continue
        
        response, feedback = chatbot.chat(user_input)
        print(f"Bot: {response}")
        if feedback != 'none':
            print(f"[Detected: {feedback}]")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='Demo baseline PAL')
    parser.add_argument('--feedback', action='store_true', help='Demo feedback-sensitive PAL')
    args = parser.parse_args()
    
    if args.baseline:
        demo_baseline()
    elif args.feedback:
        demo_feedback()
    else:
        print("\nUsage:")
        print("  python demo.py --baseline   # Test baseline PAL")
        print("  python demo.py --feedback   # Test feedback-sensitive PAL")