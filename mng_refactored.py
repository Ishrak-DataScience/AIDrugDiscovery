# -*- coding: utf-8 -*-
"""
NMG_ISHRAK_Cleaned.py
Refactored for clarity, efficiency, and removal of redundant training steps.
"""

import os
import torch
import pandas as pd
import time
from tqdm import tqdm

# Third-party imports
# Ensure these are installed: pip install torch transformers datasets rdkit pandas
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer, 
    pipeline,
    set_seed
)
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
set_seed(42)

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
INPUT_FILENAME = "Components-smiles-cactvs.smi"  # Change this to your specific file path
CLEAN_DATA_FILE = "smiles.txt"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./smiles_gpt2"

# Optional: specific for Google Colab file uploading
if not os.path.exists(INPUT_FILENAME):
    try:
        from google.colab import files
        print("Upload your SMILES file:")
        uploaded = files.upload()
        INPUT_FILENAME = list(uploaded.keys())[0]
    except ImportError:
        print(f"⚠️ File {INPUT_FILENAME} not found and not in Colab.")

# ==========================================
# 2. Data Processing (Clean & Validate)
# ==========================================
print(f"\n🧪 Processing {INPUT_FILENAME}...")

smiles_list = []
# Read file
try:
    with open(INPUT_FILENAME, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Assumes SMILES is the first column
            first_col = line.split()[0]
            smiles_list.append(first_col)
except FileNotFoundError:
    print("❌ Input file not found. Please check the path.")
    smiles_list = []

print(f"Total lines read: {len(smiles_list)}")

# Validate with RDKit
valid_smiles = []
seen = set()

print("Validating SMILES with RDKit...")
for s in tqdm(smiles_list):
    if s in seen:
        continue
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        valid_smiles.append(s)
        seen.add(s)

print(f"✅ Kept {len(valid_smiles)} valid/unique SMILES out of {len(smiles_list)}.")

# Save processed data for the model
if valid_smiles:
    pd.Series(valid_smiles).to_csv(CLEAN_DATA_FILE, index=False, header=False)
    print(f"📁 Saved training data to {CLEAN_DATA_FILE}")
else:
    raise ValueError("No valid SMILES found. Aborting.")

# ==========================================
# 3. Model & Tokenizer Setup
# ==========================================
print(f"\n🤖 Loading {MODEL_NAME} model and tokenizer...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # GPT-2 needs this fix
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)

# ==========================================
# 4. Strategy: Transfer Learning (Unfreezing)
# ==========================================
# Freeze ALL layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze ONLY the last transformer block (Layer 11 for GPT-2 small)
layer_to_unfreeze = 11
try:
    for param in model.transformer.h[layer_to_unfreeze].parameters():
        param.requires_grad = True
    print(f"🔥 Unfrozen transformer layer: {layer_to_unfreeze}")
except IndexError:
    print("⚠️ Layer index out of bounds, checking model structure...")

# Unfreeze the Language Model Head (Output layer)
for param in model.lm_head.parameters():
    param.requires_grad = True
print("🔥 Unfrozen lm_head (output layer)")

# ==========================================
# 5. Dataset Preparation
# ==========================================
print("\n📚 Preparing dataset...")
dataset = load_dataset("text", data_files={"train": CLEAN_DATA_FILE})

def tokenization_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenization_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================
# 6. Training
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,              # Increase to 3-5 for better results
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Simulates larger batch size
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  # Use mixed precision if on GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

print("\n🏋️ Starting training...")
trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")

# ==========================================
# 7. Generation & Validation
# ==========================================
print("\n⚗️ Generating new molecules...")

# Load the Fine-Tuned Model
gen_pipeline = pipeline(
    "text-generation",
    model=OUTPUT_DIR,
    tokenizer=OUTPUT_DIR,
    device=0 if torch.cuda.is_available() else -1
)

seed_text = "C" # Starting atom
generated_outputs = gen_pipeline(
    seed_text,
    max_length=60,
    num_return_sequences=50, # Generate 50 molecules
    do_sample=True,
    top_k=50
)

# Validate generated output
valid_generated = []
print("Validating generated molecules...")

for output in generated_outputs:
    # Extract text and clean whitespace
    smi = output["generated_text"].strip().split()[0]
    
    # Check validity
    if Chem.MolFromSmiles(smi):
        valid_generated.append(smi)

# Remove duplicates
valid_generated = list(dict.fromkeys(valid_generated))

print(f"✨ Generated {len(valid_generated)} valid unique SMILES.")

# Save results
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_csv = f"generated_valid_{timestamp}.csv"
pd.Series(valid_generated, name="SMILES").to_csv(output_csv, index=False)
print(f"💾 Saved generated molecules to {output_csv}")