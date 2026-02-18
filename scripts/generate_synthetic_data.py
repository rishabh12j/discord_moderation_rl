"""
Generate Multilingual Synthetic Discord Data for RL Moderation Environment
Using textdetox/xlmr-large-toxicity-classifier for ground-truth scoring.
"""
import os
import json
import random
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_directories():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def generate_synthetic_data(num_conversations=500, messages_per_conv=20):
    print("="*70)
    print("Generating Multilingual Synthetic Data (EN, ES, FR, DE, HI)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XLM-RoBERTa model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("textdetox/xlmr-large-toxicity-classifier").to(device)
    model.eval()

    # 1. Multilingual Vocabularies
    safe_texts = [
        # English
        "Hello everyone, how are you?", "Thanks for the help!", "I agree completely.", 
        "Does anyone have the link to the docs?", "Great job on the release!",
        # Spanish
        "¡Hola a todos! ¿Cómo están?", "¡Muchas gracias por la ayuda!", "Estoy totalmente de acuerdo.",
        "¿Alguien tiene el enlace a la documentación?", "¡Buen trabajo con el lanzamiento!",
        # French
        "Bonjour tout le monde, comment ça va?", "Merci beaucoup pour votre aide!", 
        "Je suis tout à fait d'accord.", "Quelqu'un a le lien vers la doc?",
        # German
        "Hallo zusammen, wie geht es euch?", "Vielen Dank für die Hilfe!", "Ich stimme voll und ganz zu.",
        "Hat jemand den Link zu den Docs?", "Tolle Arbeit beim Release!",
        # Hindi (Romanized)
        "Sabko namaste, kaise ho aap log?", "Madad ke liye dhanyawad!", "Main poori tarah se sehmat hu.",
        "Kisi ke paas docs ka link hai kya?", "Naye update par badhiya kaam kiya!"
    ]
    
    borderline_texts = [
        # English
        "This is kinda stupid to be honest.", "Nobody asked for your opinion.",
        # Spanish
        "Esto es un poco estúpido para ser honesto.", "A nadie le importa tu opinión.",
        # French
        "C'est un peu stupide pour être honnête.", "Personne n'a demandé ton avis.",
        # German
        "Das ist ehrlich gesagt ziemlich dumm.", "Niemand hat nach deiner Meinung gefragt.",
        # Hindi
        "Yeh sach mein thoda bekar hai.", "Tumhari raye kisne mangi?"
    ]
    
    toxic_texts = [
        # English
        "You are an absolute idiot.", "Shut up, you pathetic loser.", 
        "Go to hell everyone.", "You're completely r*tarded.", 
        # Spanish
        "Eres un completo idiota.", "Cállate, perdedor patético.",
        "Váyanse al diablo todos.", "Eres completamente estúpido.",
        # French
        "Tu es un idiot absolu.", "Ferme ta gueule, pauvre perdant.",
        "Allez tous en enfer.", "Tu es complètement débile.",
        # German
        "Du bist ein absoluter Idiot.", "Halt die Klappe, du erbärmlicher Verlierer.",
        "Fahrt alle zur Hölle.", "Du bist komplett zurückgeblieben.",
        # Hindi
        "Tu ek number ka gadha hai.", "Bakwas band kar, kutte.",
        "Bhaad mein jao sab.", "Tu bilkul pagal hai."
    ]

    # 2. Create Synthetic Users (Behavior logic remains the same)
    num_users = int(num_conversations * 1.5)
    users = {}
    
    for i in range(num_users):
        rand_val = random.random()
        if rand_val < 0.70: profile = "good_user"
        elif rand_val < 0.90: profile = "borderline"
        else: profile = "troll"
            
        users[f"user_{i}"] = {"profile": profile, "join_days_ago": random.randint(1, 365), "messages": [], "scores": []}

    # 3. Generate Threads
    print(f"Generating {num_conversations} threads ({num_conversations * messages_per_conv} messages)...")
    all_messages = []
    
    for conv_id in range(num_conversations):
        conv_type = random.choices(['safe', 'mixed', 'toxic'], weights=[0.7, 0.2, 0.1])[0]
        
        for msg_id in range(messages_per_conv):
            if conv_type == 'safe':
                valid_users = [u for u, d in users.items() if d['profile'] == 'good_user']
                user_id = random.choice(valid_users) if valid_users else random.choice(list(users.keys()))
            elif conv_type == 'mixed':
                user_id = random.choice(list(users.keys()))
            else:
                valid_users = [u for u, d in users.items() if d['profile'] in ['borderline', 'troll']]
                user_id = random.choice(valid_users) if valid_users else random.choice(list(users.keys()))
                
            profile = users[user_id]['profile']
            
            # Select text heavily biased by user profile
            if profile == 'good_user':
                text = random.choice(safe_texts) if random.random() < 0.98 else random.choice(borderline_texts)
            elif profile == 'borderline':
                text = random.choice(borderline_texts) if random.random() < 0.7 else random.choice(toxic_texts)
            else:
                text = random.choice(toxic_texts) if random.random() < 0.8 else random.choice(borderline_texts)
                
            msg = {'conversation_id': conv_id, 'message_id': msg_id, 'user_id': user_id, 'user_profile': profile, 'text': text, 'timestamp': msg_id}
            all_messages.append(msg)
            users[user_id]['messages'].append(msg)

    # 4. Score with Multilingual Model
    print("\nScoring messages via textdetox/xlmr-large-toxicity-classifier...")
    batch_size = 32
    scored_messages = []
    
    start_time = time.time()
    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i+batch_size]
        texts = [m['text'] for m in batch]
        
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Softmax to get probability of Toxic class (Index 1)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()
            
        for msg, score in zip(batch, scores):
            msg['toxicity_score'] = float(score)
            msg['toxic'] = 1 if score > 0.5 else 0
            scored_messages.append(msg)
            users[msg['user_id']]['scores'].append(float(score))
            
    print(f"Scoring complete in {time.time() - start_time:.2f}s")

    # 5. Build User Dictionary
    user_lookup = {}
    for uid, udata in users.items():
        if not udata['scores']: continue
        scores = udata['scores']
        user_lookup[uid] = {
            "avg_toxicity": float(np.mean(scores)),
            "total_messages": len(scores),
            "toxic_messages": sum(1 for s in scores if s > 0.5),
            "profile": udata['profile'],
            "join_days_ago": udata['join_days_ago']
        }

    # 6. Save Data
    setup_directories()
    df = pd.DataFrame(scored_messages)
    df.to_csv("data/processed/conversations_with_users.csv", index=False)
    with open("data/processed/user_lookup.json", "w") as f:
        json.dump(user_lookup, f, indent=4)
        
    print(f"\n✅ Dataset updated! (Toxic: {df['toxic'].sum()} / {len(df)})")

if __name__ == "__main__":
    generate_synthetic_data(num_conversations=500)