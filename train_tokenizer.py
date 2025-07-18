#!/usr/bin/env python3
"""
Training script for order tokenization model
Converts text input to tokenized words for order processing AI
"""

import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from accelerate import Accelerator
from functools import partial

def validate_and_fix_json(text):
    """Validate and attempt to fix JSON output"""
    try:
        # Try to parse as-is
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        fixed_text = text.strip()
        
        # Add braces if missing
        if not fixed_text.startswith('{'):
            fixed_text = '{' + fixed_text
        if not fixed_text.endswith('}'):
            fixed_text = fixed_text + '}'
        
        # Try to parse again
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            # If still fails, return None
            return None

def main():
    print("Starting order tokenization training...")
    
    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    
    # Load data from TSV file and preprocess
    def load_and_tokenize_data(file_path):
        """Load the TSV file and prepare data for tokenization"""
        # Load the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        
        # Extract the "text" column for inputs
        inputs = [text for text in df["text"]]
        
        # Combine target columns into a single JSON-like structure for each row
        targets = []
        for _, row in df.iterrows():
            # Handle NaN values and convert pandas types to JSON-serializable types
            target_dict = {
                "customer_name": str(row["customer_name"]) if pd.notna(row["customer_name"]) else None,
                "order_type": str(row["order_type"]) if pd.notna(row["order_type"]) else None,
                "total_number_of_different_items": int(row["total_number_of_different_items"]) if pd.notna(row["total_number_of_different_items"]) else None,
                "order_items_name": str(row["order_items_name"]) if pd.notna(row["order_items_name"]) else None,
                "order_items_quantity": str(row["order_items_quantity"]) if pd.notna(row["order_items_quantity"]) else None,
                "order_items_modifications": str(row["order_items_modifications"]) if pd.notna(row["order_items_modifications"]) else None,
                "order_notes": str(row["order_notes"]) if pd.notna(row["order_notes"]) else None
            }
            # Ensure proper JSON formatting with consistent spacing
            json_target = json.dumps(target_dict, ensure_ascii=False, separators=(',', ':'))
            # Validate that the JSON is properly formatted
            try:
                json.loads(json_target)  # Validate JSON
                targets.append(json_target)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON target generated: {json_target}")
                print(f"Error: {e}")
                # Use a default valid JSON structure
                default_dict = {k: None for k in target_dict.keys()}
                targets.append(json.dumps(default_dict, ensure_ascii=False, separators=(',', ':')))
        
        return inputs, targets

    # Improved tokenization function with proper label handling
    def tokenize_data(inputs, targets):
        """Tokenize inputs and targets for the model with proper label processing"""
        # Add task prefix to inputs
        prefixed_inputs = ["extract order: " + text for text in inputs]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            prefixed_inputs, 
            max_length=512, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize targets (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=256, 
                truncation=True, 
                padding="max_length",
                return_tensors="pt"
            )
        
        # Replace padding token id with -100 for loss calculation
        labels_input_ids = labels["input_ids"]
        labels_input_ids[labels_input_ids == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels_input_ids
        
        # Convert to regular Python lists for Dataset compatibility
        model_inputs = {k: v.tolist() for k, v in model_inputs.items()}
        
        return model_inputs

    # Load and tokenize data
    file_path = "data/sample_order_data.tsv"  # Change to your TSV file path
    
    try:
        inputs, targets = load_and_tokenize_data(file_path)
        tokenized_data = tokenize_data(inputs, targets)
        print("Data loaded and tokenized successfully!")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize Accelerator
    accelerator = Accelerator()
    
    print(f"Accelerator initialized successfully.")
    print(f"Device: {accelerator.device}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Number of processes: {accelerator.num_processes}")
    
    # Create dataset
    dataset = Dataset.from_dict(tokenized_data)
    print(f"Dataset created with {len(dataset)} examples")
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print("Model loaded successfully")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        predict_with_generate=True,
        generation_max_length=256,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Prepare for training with accelerator
    model, trainer = accelerator.prepare(model, trainer)
    
    print("Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    
    print("Training completed and model saved!")
    
    # Test the model with a sample input
    test_input = "Hello, my name is John Smith. I'd like to order 2 large pizzas with extra cheese and 3 diet cokes please."
    
    print(f"\nTesting model with input: {test_input}")
    
    # Load the saved model for testing
    try:
        test_model = T5ForConditionalGeneration.from_pretrained("./trained_model")
        test_tokenizer = T5TokenizerFast.from_pretrained("./trained_model")
        
        # Move model to CPU for testing to avoid MPS issues
        test_model = test_model.to("cpu")
        
        # Tokenize test input
        input_text = "extract order: " + test_input
        inputs = test_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate output with improved parameters for JSON consistency
        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_length=256,
                num_beams=8,  # Increased beam search
                early_stopping=True,
                do_sample=False,
                temperature=0.1,  # Lower temperature for more consistent output
                repetition_penalty=1.1,  # Reduce repetition
                length_penalty=1.0,  # Neutral length penalty
                pad_token_id=test_tokenizer.pad_token_id,
                eos_token_id=test_tokenizer.eos_token_id
            )
        
        # Decode output
        decoded_output = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest Input: {test_input}")
        print(f"Model Output: {decoded_output}")
        
        # Try to parse as JSON with validation utility
        parsed_output = validate_and_fix_json(decoded_output)
        if parsed_output is not None:
            print("✅ Valid JSON output!")
            print(json.dumps(parsed_output, indent=2))
        else:
            print("❌ Output is not valid JSON and could not be fixed")
            print(f"Raw output: {decoded_output}")
            
    except Exception as e:
        print(f"⚠️ Error testing model: {e}")
        print("Model training completed successfully, but testing failed.")

if __name__ == "__main__":
    main()
