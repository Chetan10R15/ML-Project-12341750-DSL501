"""
Updated Data Loading Functions for emotion_2k_processed.csv
"""

import pandas as pd
import numpy as np
import json
import os
import ast
from typing import List, Dict

# ==================== Load emotion_2k_processed.csv ====================
def load_emotion_dataset(csv_path='emotion_2k_processed.csv'):
    """
    Load and process the emotion_2k_processed.csv dataset
    
    Dataset columns:
    - Index: Row index
    - Situation: The user's situation/statement
    - emotion: The emotion label
    - empathetic_dialogues: The conversation between Customer and Agent
    - labels: Additional labels
    """
    print("\n[1/6] Loading emotion_2k_processed.csv Dataset...")
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} examples from CSV")
    
    return df

# ==================== Process Dataset for Training ====================
def process_emotion_dataset(df, max_samples=100):
    """
    Process the emotion dataset into the format needed for training
    
    Args:
        df: DataFrame with emotion dataset
        max_samples: Maximum number of samples to process (default 100 for training)
    
    Returns:
        List of dictionaries with processed data
    """
    print(f"\n[2/6] Processing {min(max_samples, len(df))} samples...")
    
    processed_data = []
    
    # Take only max_samples for training
    df_subset = df.head(max_samples)
    
    for idx, row in df_subset.iterrows():
        try:
            situation = str(row['Situation'])
            emotion = str(row['emotion'])
            dialogue = str(row['empathetic_dialogues'])
            
            # Parse the dialogue to extract conversation
            # Format: "Customer: ... Agent: ..."
            dialogue_parts = dialogue.split('Agent :')
            
            if len(dialogue_parts) >= 2:
                customer_part = dialogue_parts[0].replace('Customer :', '').strip()
                agent_part = dialogue_parts[1].strip()
                
                # Build dialogue history
                dialogue_history = f"User: {situation} Bot: {agent_part}"
                
                # Extract persona from situation and emotion
                persona = [emotion, 'needs support', 'emotional situation']
                
                # Determine feedback type (randomly assign for training diversity)
                feedback_types = ['none', 'correction', 'preference', 'dissatisfaction']
                feedback_weights = [0.7, 0.1, 0.1, 0.1]  # 70% none, 10% each other
                feedback_type = np.random.choice(feedback_types, p=feedback_weights)
                
                # Determine strategy based on emotion
                emotion_to_strategy = {
                    'excited': 'Affirmation and Reassurance',
                    'proud': 'Affirmation and Reassurance',
                    'caring': 'Reflection of Feelings',
                    'impressed': 'Affirmation and Reassurance',
                    'confident': 'Affirmation and Reassurance',
                    'grateful': 'Reflection of Feelings',
                    'ashamed': 'Reflection of Feelings',
                    'hopeful': 'Affirmation and Reassurance',
                    'surprised': 'Question',
                    'annoyed': 'Reflection of Feelings',
                    'disappointed': 'Reflection of Feelings',
                    'joyful': 'Affirmation and Reassurance',
                    'afraid': 'Affirmation and Reassurance',
                    'anxious': 'Affirmation and Reassurance',
                    'lonely': 'Reflection of Feelings',
                    'furious': 'Reflection of Feelings',
                    'terrified': 'Affirmation and Reassurance',
                    'devastated': 'Reflection of Feelings',
                    'jealous': 'Reflection of Feelings',
                    'sad': 'Reflection of Feelings',
                    'disgusted': 'Reflection of Feelings',
                    'guilty': 'Reflection of Feelings',
                    'embarrassed': 'Reflection of Feelings',
                    'nostalgic': 'Reflection of Feelings',
                    'trusting': 'Affirmation and Reassurance',
                    'faithful': 'Affirmation and Reassurance',
                    'sentimental': 'Reflection of Feelings',
                    'anticipating': 'Question',
                    'prepared': 'Affirmation and Reassurance',
                    'content': 'Affirmation and Reassurance'
                }
                
                strategy = emotion_to_strategy.get(emotion.lower(), 'Question')
                
                # Create enhanced empathetic response
                response = generate_empathetic_response(situation, emotion, agent_part)
                
                processed_data.append({
                    'dialogue_history': dialogue_history,
                    'persona': persona,
                    'response': response,
                    'feedback_type': feedback_type,
                    'strategy': strategy,
                    'original_emotion': emotion
                })
                
        except Exception as e:
            print(f"  Warning: Skipping row {idx} due to error: {e}")
            continue
    
    print(f"✓ Successfully processed {len(processed_data)} samples")
    print(f"✓ Emotion distribution:")
    emotion_counts = pd.Series([d['original_emotion'] for d in processed_data]).value_counts().head(10)
    print(emotion_counts)
    
    return processed_data

# ==================== Generate Enhanced Empathetic Response ====================
def generate_empathetic_response(situation, emotion, original_response):
    """
    Generate an enhanced empathetic response based on situation and emotion
    
    Args:
        situation: The user's situation
        emotion: The detected emotion
        original_response: Original agent response from dataset
    
    Returns:
        Enhanced empathetic response
    """
    
    # Empathy starters based on emotion
    empathy_starters = {
        'sad': "I'm really sorry you're going through this. ",
        'anxious': "I can hear how worried you are about this. ",
        'afraid': "That sounds really frightening. ",
        'angry': "I understand why you feel so upset about this. ",
        'frustrated': "That sounds incredibly frustrating. ",
        'lonely': "I'm so sorry you're feeling this isolated. ",
        'devastated': "I can't imagine how difficult this must be for you. ",
        'guilty': "It takes courage to acknowledge these feelings. ",
        'ashamed': "Thank you for sharing this with me. ",
        'disappointed': "I can understand why you feel let down. ",
        'jealous': "Those feelings are completely valid. ",
        'disgusted': "That sounds really unpleasant. ",
        'surprised': "Wow, that must have been unexpected! ",
        'excited': "That's wonderful! ",
        'joyful': "I'm so happy for you! ",
        'proud': "You should feel proud of yourself. ",
        'grateful': "That's so thoughtful of you to appreciate that. ",
        'hopeful': "It's great that you're staying positive. ",
        'content': "I'm glad things are going well for you. ",
        'confident': "That's a great mindset to have! ",
        'caring': "That shows how much you care. ",
        'impressed': "That's really impressive! ",
        'nostalgic': "Those are precious memories. ",
        'faithful': "That's a wonderful commitment. ",
        'trusting': "It's good that you have faith in that. "
    }
    
    starter = empathy_starters.get(emotion.lower(), "I hear you. ")
    
    # If original response is good, use it with enhancement
    if len(original_response) > 20:
        return starter + original_response
    else:
        # Generate a basic empathetic response
        return f"{starter}Tell me more about what you're experiencing. How are you coping with this?"

# ==================== Save Processed Data ====================
def save_processed_data(data, output_path='processed_training_data.csv'):
    """
    Save processed data to CSV
    
    Args:
        data: List of dictionaries with processed data
        output_path: Path to save CSV
    """
    print(f"\n[3/6] Saving processed data...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(data)} processed samples to {output_path}")
    print(f"✓ Feedback distribution:")
    fb_counts = df['feedback_type'].value_counts()
    print(fb_counts)

# ==================== Main Processing Function ====================
def create_training_data_from_emotion_dataset(
    csv_path='emotion_2k_processed.csv',
    output_path='processed_training_data.csv',
    max_samples=100
):
    """
    Main function to load, process, and save training data
    
    Args:
        csv_path: Path to emotion_2k_processed.csv
        output_path: Path to save processed data
        max_samples: Maximum samples to process
    
    Returns:
        List of processed data dictionaries
    """
    print("="*70)
    print("PROCESSING EMOTION DATASET FOR TRAINING")
    print("="*70)
    
    # Load dataset
    df = load_emotion_dataset(csv_path)
    
    if df is None:
        print("✗ Failed to load dataset")
        return None
    
    # Process dataset
    processed_data = process_emotion_dataset(df, max_samples)
    
    # Save processed data
    save_processed_data(processed_data, output_path)
    
    print("\n" + "="*70)
    print("✓ DATA PROCESSING COMPLETE!")
    print("="*70)
    
    return processed_data

# ==================== Usage Example ====================
if __name__ == "__main__":
    # Process the emotion dataset
    data = create_training_data_from_emotion_dataset(
        csv_path='emotion_2k_processed.csv',
        output_path='processed_training_data.csv',
        max_samples=100  # Start with 100 samples for faster training
    )
    
    if data:
        print(f"\n✓ Ready to use {len(data)} samples for training!")
        print("\nSample processed data:")
        print(json.dumps(data[0], indent=2))