"""
Feedback-Sensitive Persona-Aware Emotional Support Chatbot
Complete ML Pipeline with Streamlit Interface
Student: Rathod Chetankumar A (12341750)
Course: DSL501 - Machine Learning Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# 1. FEEDBACK DETECTION MODULE (Multi-task NLU Classifier)

class FeedbackDetector:
    """
    Multi-class classifier for detecting user feedback types:
    - Correction: User corrects bot's misunderstanding
    - Dissatisfaction: User expresses response wasn't helpful
    - Preference: User states preferences for interaction style
    """
    
    def __init__(self):
        # In production: Load DistilRoBERTa/BERT model
        # self.model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base')
        # self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        # Pattern-based fallback for demo
        self.correction_patterns = [
            r"i'?m not|i am not|actually|that'?s wrong|incorrect|no,?\s*i",
            r"i never said|i didn'?t say|you'?re wrong",
            r"that'?s not (true|right|correct)"
        ]
        
        self.dissatisfaction_patterns = [
            r"didn'?t help|not helpful|stop|unhelpful|useless",
            r"don'?t (like|want) (this|that)|this (isn'?t|is not) working",
            r"you'?re not helping|that makes it worse"
        ]
        
        self.preference_patterns = [
            r"don'?t (give|provide|tell|offer) (me )?(advice|suggestions)",
            r"(just|only) (listen|be there)|i (prefer|rather|want) (to )?just",
            r"stop (asking|suggesting)|i want you to"
        ]
    
    def detect(self, text: str) -> Dict[str, bool]:
        """Detect feedback types in user message"""
        text_lower = text.lower()
        
        correction = any(re.search(pattern, text_lower) for pattern in self.correction_patterns)
        dissatisfaction = any(re.search(pattern, text_lower) for pattern in self.dissatisfaction_patterns)
        preference = any(re.search(pattern, text_lower) for pattern in self.preference_patterns)
        
        return {
            'correction': correction,
            'dissatisfaction': dissatisfaction,
            'preference': preference,
            'any_feedback': correction or dissatisfaction or preference
        }
    
    def train(self, train_data: pd.DataFrame):
        """
        Train feedback detection model
        Expected columns: ['text', 'correction_label', 'dissatisfaction_label', 'preference_label']
        """
        # In production: Fine-tune DistilRoBERTa on augmented feedback data
        # X = train_data['text']
        # y = train_data[['correction_label', 'dissatisfaction_label', 'preference_label']]
        # Fine-tune multi-label classification model
        pass

# 2. PERSONA EXTRACTION MODULE

class PersonaExtractor:
    """
    Extract persona information from conversation history
    Uses pattern matching and NLP to identify user characteristics
    """
    
    def __init__(self):
        # In production: Load BART or T5 model fine-tuned on Persona-Chat
        # self.model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
        # self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        
        self.persona_patterns = [
            (r"i'?m (a|an) ([^,.!?]+)", "User is {match}"),
            (r"i am (a|an) ([^,.!?]+)", "User is {match}"),
            (r"i work (as|at|in) ([^,.!?]+)", "User works {match}"),
            (r"i study ([^,.!?]+)", "User studies {match}"),
            (r"i live (in|at) ([^,.!?]+)", "User lives {match}"),
            (r"i (feel|am feeling) ([^,.!?]+)", "User feels {match}"),
            (r"i have (been diagnosed with|been dealing with) ([^,.!?]+)", "User has {match}"),
            (r"my (job|work) is ([^,.!?]+)", "User's job is {match}"),
            (r"i'?m married|i am married", "User is married"),
            (r"i have (\d+) (kids|children)", "User has {match}"),
        ]
    
    def extract(self, text: str) -> List[str]:
        """Extract persona statements from text"""
        personas = []
        text_lower = text.lower()
        
        for pattern, template in self.persona_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    # Extract the meaningful part (skip articles)
                    relevant_match = match.group(2) if match.group(1) in ['a', 'an', 'as', 'at', 'in'] else match.group(1)
                    persona = template.format(match=relevant_match.strip())
                    personas.append(persona)
                elif len(match.groups()) == 1:
                    persona = template.format(match=match.group(1).strip())
                    personas.append(persona)
                else:
                    personas.append(template)
        
        return personas
    
    def train(self, train_data: pd.DataFrame):
        """
        Train persona extraction model on Persona-Chat dataset
        Expected columns: ['conversation_history', 'persona_statements']
        """
        # In production: Fine-tune BART on Persona-Chat for persona generation
        pass

# 3. PERSONA MEMORY MODULE

class PersonaMemory:
    """
    Dynamic persona memory with ADD, REVISE, DELETE operations
    Maintains user persona and preferences
    """
    
    def __init__(self):
        self.persona_list: List[str] = []
        self.preferences: Dict[str, any] = {}
        self.timestamp_map: Dict[str, datetime] = {}
    
    def add_persona(self, persona: str):
        """Add new persona information"""
        if persona not in self.persona_list:
            self.persona_list.append(persona)
            self.timestamp_map[persona] = datetime.now()
    
    def revise_persona(self, old_persona: str, new_persona: str):
        """Revise existing persona information"""
        if old_persona in self.persona_list:
            idx = self.persona_list.index(old_persona)
            self.persona_list[idx] = new_persona
            self.timestamp_map[new_persona] = datetime.now()
            del self.timestamp_map[old_persona]
    
    def delete_last_persona(self):
        """Delete most recent persona (used when correction detected)"""
        if self.persona_list:
            removed = self.persona_list.pop()
            if removed in self.timestamp_map:
                del self.timestamp_map[removed]
    
    def add_preference(self, key: str, value: any):
        """Add user preference"""
        self.preferences[key] = value
    
    def get_persona_context(self) -> str:
        """Get formatted persona context for model input"""
        if not self.persona_list:
            return ""
        return "User information: " + "; ".join(self.persona_list)
    
    def get_all(self) -> Dict:
        """Get all stored information"""
        return {
            'persona': self.persona_list,
            'preferences': self.preferences
        }

# 4. STRATEGY ADAPTER MODULE

class StrategyAdapter:
    """
    Learnable strategy adapter that selects appropriate support strategies
    based on context, persona, and preferences
    """
    
    def __init__(self):
        self.strategy_weights = {
            'question': 1.0,
            'restatement': 0.75,
            'reflection': 0.5,
            'self_disclosure': 0.25,
            'affirmation': 0.75,
            'suggestion': 0.75,
            'information': 0.75,
            'others': 0.375
        }
    
    def select_strategy(self, 
                       user_text: str, 
                       persona_available: bool,
                       preferences: Dict,
                       conversation_length: int) -> str:
        """
        Select appropriate support strategy based on context
        
        Strategy Selection Logic (based on PAL paper):
        - Question: When lacking persona information or exploring situation
        - Reflection: When user expresses emotions
        - Affirmation: When user needs encouragement (esp. with no_advice preference)
        - Suggestion: When persona is available and user seeks advice
        - Restatement: When clarifying user's situation
        - Information: When providing facts or resources
        """
        
        # Check preferences first
        if preferences.get('no_advice', False):
            return 'affirmation'
        
        if preferences.get('only_listen', False):
            return 'reflection'
        
        # Analyze user message
        is_seeking_advice = bool(re.search(r'what should i|how can i|advice|help me|what do i do', user_text.lower()))
        is_expressing_emotion = bool(re.search(r'i feel|i am (sad|angry|frustrated|anxious|depressed|stressed)', user_text.lower()))
        is_asking_question = bool(re.search(r'\?', user_text))
        
        # Strategy selection
        if is_expressing_emotion:
            return 'reflection'
        
        if is_seeking_advice and persona_available:
            return 'suggestion'
        
        if is_seeking_advice and not persona_available:
            return 'question'
        
        if conversation_length < 3:
            return 'question'
        
        if persona_available and conversation_length >= 5:
            return 'affirmation'
        
        return 'reflection'
    
    def get_strategy_weight(self, strategy: str) -> float:
        """Get weight for strategy-based controllable generation (alpha value)"""
        return self.strategy_weights.get(strategy, 0.5)

# 5. RESPONSE GENERATOR MODULE

class ResponseGenerator:
    """
    Generate empathetic responses using strategy and persona information
    Uses strategy-based controllable generation
    """
    
    def __init__(self):
        # In production: Load fine-tuned Blenderbot-90M
        # self.model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-90M')
        # self.tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-90M')
        
        # Template-based responses for demo
        self.response_templates = {
            'question': [
                "Can you tell me more about what you're going through?",
                "What aspect of this concerns you the most?",
                "How long have you been feeling this way?",
                "What's been happening that led to these feelings?"
            ],
            'reflection': [
                "I hear that you're experiencing difficult emotions. {persona_context}That must be really challenging.",
                "It sounds like you're dealing with a lot right now. Your feelings are completely valid.",
                "I understand this is affecting you deeply. {persona_context}Would you like to explore these feelings more?",
                "That sounds really tough. {persona_context}I appreciate you sharing this with me."
            ],
            'affirmation': [
                "I want you to know that your feelings matter. {persona_context}You're doing the best you can.",
                "You're showing strength by reaching out. That takes real courage.",
                "I acknowledge what you're experiencing. {persona_context}You're not alone in this.",
                "It's okay to feel this way. {persona_context}These feelings are valid."
            ],
            'suggestion': [
                "Based on what you've shared, {persona_context}have you considered talking to someone you trust about this?",
                "Given your situation, {persona_context}it might help to take small steps. Would you be open to trying one thing this week?",
                "Considering what you've told me, {persona_context}focusing on self-care could be beneficial. What brings you comfort?",
                "{persona_context}Sometimes it helps to break things down into smaller, manageable pieces. What feels most urgent right now?"
            ],
            'restatement': [
                "So what I'm hearing is that {restate}. Is that right?",
                "It sounds like you're saying {restate}. Did I understand that correctly?",
                "Let me make sure I understand - {restate}. Is that accurate?"
            ],
            'information': [
                "Here's some information that might help: {info}",
                "Based on what you've shared, {persona_context}here are some resources that could be useful.",
                "Let me share what I know about this: {info}"
            ]
        }
    
    def generate(self, 
                strategy: str,
                user_text: str,
                persona_context: str,
                acknowledgment: str = "") -> str:
        """
        Generate response using selected strategy
        
        In production, this would use:
        P_final(r_t|r_<t, d, p) ∝ P(r_t|r_<t, d, p) * (P(r_t|r_<t, d, p) / P(r_t|r_<t, d))^α
        where α is strategy-dependent weight
        """
        
        templates = self.response_templates.get(strategy, self.response_templates['reflection'])
        template = np.random.choice(templates)
        
        # Format persona context
        persona_text = ""
        if persona_context and '{persona_context}' in template:
            persona_text = f"given that {persona_context.lower().replace('user information: ', '')}, "
        
        response = template.format(persona_context=persona_text)
        
        # Add acknowledgment if feedback was detected
        if acknowledgment:
            response = acknowledgment + response
        
        return response
    
    def train(self, train_data: pd.DataFrame):
        """
        Fine-tune response generation model on ESConv dataset
        Expected columns: ['context', 'persona', 'strategy', 'response']
        """
        # In production: Fine-tune Blenderbot-90M on ESConv with persona augmentation
        pass

# 6. EVALUATION METRICS MODULE

class EvaluationMetrics:
    """
    Track and calculate evaluation metrics:
    - FUR (Feedback Utilization Rate)
    - CIL (Correction Incorporation Latency)
    - DRR (Dissatisfaction Recovery Rate)
    """
    
    def __init__(self):
        self.feedback_detected = 0
        self.feedback_utilized = 0
        self.corrections = []
        self.dissatisfaction_events = []
        self.recovery_count = 0
        self.turn_count = 0
    
    def record_feedback(self, feedback_type: str, utilized: bool):
        """Record feedback detection and utilization"""
        self.feedback_detected += 1
        if utilized:
            self.feedback_utilized += 1
        
        if feedback_type == 'correction':
            self.corrections.append(self.turn_count)
        elif feedback_type == 'dissatisfaction':
            self.dissatisfaction_events.append(self.turn_count)
    
    def record_recovery(self):
        """Record successful recovery from dissatisfaction"""
        self.recovery_count += 1
    
    def increment_turn(self):
        """Increment conversation turn counter"""
        self.turn_count += 1
    
    def get_fur(self) -> float:
        """Calculate Feedback Utilization Rate"""
        if self.feedback_detected == 0:
            return 0.0
        return self.feedback_utilized / self.feedback_detected
    
    def get_cil(self) -> float:
        """Calculate average Correction Incorporation Latency"""
        if len(self.corrections) <= 1:
            return 0.0
        
        latencies = [self.corrections[i+1] - self.corrections[i] for i in range(len(self.corrections)-1)]
        return np.mean(latencies) if latencies else 0.0
    
    def get_drr(self) -> float:
        """Calculate Dissatisfaction Recovery Rate"""
        if len(self.dissatisfaction_events) == 0:
            return 0.0
        return self.recovery_count / len(self.dissatisfaction_events)
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics"""
        return {
            'FUR': self.get_fur(),
            'CIL': self.get_cil(),
            'DRR': self.get_drr(),
            'Total Feedback': self.feedback_detected,
            'Corrections': len(self.corrections),
            'Dissatisfaction Events': len(self.dissatisfaction_events)
        }

# 7. MAIN CHATBOT PIPELINE

class FeedbackSensitivePAL:
    """
    Main chatbot pipeline integrating all modules
    """
    
    def __init__(self):
        self.feedback_detector = FeedbackDetector()
        self.persona_extractor = PersonaExtractor()
        self.persona_memory = PersonaMemory()
        self.strategy_adapter = StrategyAdapter()
        self.response_generator = ResponseGenerator()
        self.metrics = EvaluationMetrics()
        self.conversation_history = []
    
    def process_message(self, user_message: str) -> Tuple[str, Dict]:
        """
        Process user message through the complete pipeline
        
        Pipeline Flow:
        1. Detect feedback (corrections, dissatisfaction, preferences)
        2. Update persona memory based on feedback
        3. Extract new persona information
        4. Select appropriate strategy
        5. Generate response with acknowledgment
        6. Update metrics
        """
        
        # Step 1: Detect feedback
        feedback = self.feedback_detector.detect(user_message)
        acknowledgment = ""
        
        # Step 2: Handle feedback
        if feedback['correction']:
            acknowledgment = "I apologize for the misunderstanding. Thank you for correcting me. "
            self.persona_memory.delete_last_persona()
            self.metrics.record_feedback('correction', utilized=True)
        
        if feedback['dissatisfaction']:
            acknowledgment = "I understand that wasn't helpful. Let me try a different approach. "
            self.metrics.record_feedback('dissatisfaction', utilized=True)
        
        if feedback['preference']:
            # Extract preference
            if re.search(r"don'?t (give|provide) advice", user_message.lower()):
                self.persona_memory.add_preference('no_advice', True)
                acknowledgment = "I understand - I'll focus on listening and supporting you without giving advice. "
            elif re.search(r"(just|only) listen", user_message.lower()):
                self.persona_memory.add_preference('only_listen', True)
                acknowledgment = "Of course - I'm here to listen. "
            
            self.metrics.record_feedback('preference', utilized=True)
        
        # Step 3: Extract persona information
        new_personas = self.persona_extractor.extract(user_message)
        for persona in new_personas:
            self.persona_memory.add_persona(persona)
        
        # Step 4: Select strategy
        persona_available = len(self.persona_memory.persona_list) > 0
        strategy = self.strategy_adapter.select_strategy(
            user_message,
            persona_available,
            self.persona_memory.preferences,
            len(self.conversation_history)
        )
        
        # Step 5: Generate response
        persona_context = self.persona_memory.get_persona_context()
        response = self.response_generator.generate(
            strategy,
            user_message,
            persona_context,
            acknowledgment
        )
        
        # Step 6: Update conversation history and metrics
        self.conversation_history.append({
            'turn': len(self.conversation_history) + 1,
            'user': user_message,
            'assistant': response,
            'strategy': strategy,
            'feedback': feedback,
            'persona': self.persona_memory.persona_list.copy()
        })
        
        self.metrics.increment_turn()
        
        # Return response and metadata
        metadata = {
            'strategy': strategy,
            'feedback_detected': feedback,
            'persona': self.persona_memory.persona_list,
            'preferences': self.persona_memory.preferences
        }
        
        return response, metadata
    
    def reset(self):
        """Reset chatbot state"""
        self.persona_memory = PersonaMemory()
        self.metrics = EvaluationMetrics()
        self.conversation_history = []

# 8. STREAMLIT INTERFACE

def main():
    st.set_page_config(
        page_title="Feedback-Sensitive PAL",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .user-message {
            background-color: #4F46E5;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .assistant-message {
            background-color: #F3F4F6;
            color: black;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }
        .persona-tag {
            background-color: #EEF2FF;
            color: #4F46E5;
            padding: 5px 10px;
            border-radius: 15px;
            margin: 3px;
            display: inline-block;
            font-size: 0.85em;
        }
        .strategy-badge {
            background-color: #DBEAFE;
            color: #1E40AF;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FeedbackSensitivePAL()
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Project Information")
        st.markdown("**Student:** Rathod Chetankumar A")
        st.markdown("**ID:** 12341750")
        st.markdown("**Course:** DSL501 - ML Project")
        
        st.divider()
        
        st.title("📈 Evaluation Metrics")
        metrics = st.session_state.chatbot.metrics.get_all_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FUR", f"{metrics['FUR']:.2%}", 
                     help="Feedback Utilization Rate: How often feedback is incorporated")
        with col2:
            st.metric("DRR", f"{metrics['DRR']:.2%}",
                     help="Dissatisfaction Recovery Rate: Success in addressing dissatisfaction")
        
        st.metric("CIL", f"{metrics['CIL']:.1f} turns",
                 help="Correction Incorporation Latency: Average turns between corrections")
        
        st.divider()
        
        st.title("📝 Feedback Statistics")
        st.write(f"**Total Feedback:** {metrics['Total Feedback']}")
        st.write(f"**Corrections:** {metrics['Corrections']}")
        st.write(f"**Dissatisfaction:** {metrics['Dissatisfaction Events']}")
        
        st.divider()
        
        st.title("👤 Extracted Persona")
        persona_list = st.session_state.chatbot.persona_memory.persona_list
        if persona_list:
            for persona in persona_list:
                st.markdown(f'<span class="persona-tag">{persona}</span>', unsafe_allow_html=True)
        else:
            st.info("No persona information yet")
        
        st.divider()
        
        st.title("⚙️ User Preferences")
        prefs = st.session_state.chatbot.persona_memory.preferences
        if prefs:
            for key, value in prefs.items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("No preferences set")
        
        st.divider()
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.chatbot.reset()
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.title("🤖 Feedback-Sensitive Persona-Aware Emotional Support Chatbot")
    st.markdown("*Extending PAL with Dynamic Feedback Adaptation*")
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        with st.expander("ℹ️ How to Use", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔍 Corrections**")
                st.markdown("Say things like:")
                st.code('"Actually, I\'m not a student"')
                st.code('"That\'s incorrect"')
            
            with col2:
                st.markdown("**😔 Dissatisfaction**")
                st.markdown("Express if not helpful:")
                st.code('"That didn\'t help"')
                st.code('"This isn\'t working"')
            
            with col3:
                st.markdown("**⚙️ Preferences**")
                st.markdown("Set your preferences:")
                st.code('"Don\'t give me advice"')
                st.code('"Just listen"')
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                strategy_badge = f'<span class="strategy-badge">{msg.get("strategy", "")}</span>'
                st.markdown(f'<div class="assistant-message">{msg["content"]} {strategy_badge}</div>', 
                           unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Input area
    st.divider()
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            key="user_input",
            placeholder="Share what's on your mind... (Try: 'I'm a student feeling stressed' or 'Actually, I'm not a student')",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    # Process message
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Generate response
        response, metadata = st.session_state.chatbot.process_message(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'strategy': metadata['strategy']
        })
        
        # Rerun to update display
        st.rerun()
    
    # Example prompts
    st.divider()
    st.markdown("**💡 Try these examples:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😰 I'm a student feeling overwhelmed", use_container_width=True):
            st.session_state.messages.append({'role': 'user', 'content': "I'm a student feeling overwhelmed with exams"})
            response, metadata = st.session_state.chatbot.process_message("I'm a student feeling overwhelmed with exams")
            st.session_state.messages.append({'role': 'assistant', 'content': response, 'strategy': metadata['strategy']})
            st.rerun()
    
    with col2:
        if st.button("❌ Actually, I'm not a student", use_container_width=True):
            st.session_state.messages.append({'role': 'user', 'content': "Actually, I'm not a student"})
            response, metadata = st.session_state.chatbot.process_message("Actually, I'm not a student")
            st.session_state.messages.append({'role': 'assistant', 'content': response, 'strategy': metadata['strategy']})
            st.rerun()
    
    with col3:
        if st.button("🚫 Don't give me advice", use_container_width=True):
            st.session_state.messages.append({'role': 'user', 'content': "Don't give me advice, just listen"})
            response, metadata = st.session_state.chatbot.process_message("Don't give me advice, just listen")
            st.session_state.messages.append({'role': 'assistant', 'content': response, 'strategy': metadata['strategy']})
            st.rerun()

if __name__ == "__main__":
    main()