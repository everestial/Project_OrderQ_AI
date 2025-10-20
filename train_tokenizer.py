#!/usr/bin/env python3
"""
Training script for order tokenization model
Converts text input to tokenized words for order processing AI

KNOWLEDGE DOCS(about RAG vs Fine Tuning):

# good overview
- https://medium.com/@tahirbalarabe2/retrieval-augmented-generation-vs-fine-tuning-enhancing-llms-697e7a0cf7e0 

# this one has a chatbot on it's webpage, may be OrderQ can have something similar
- https://www.glean.com/blog/retrieval-augemented-generation-vs-fine-tuning 

# has working example of RAG wit FineTuning
- https://cobusgreyling.medium.com/fine-tuning-llms-with-retrieval-augmented-generation-rag-c66e56aec858 

# good overview from prompting guide and RAG
-https://www.promptingguide.ai/research/rag 

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

# RAG system imports
from rag_vector_index import MenuRAGIndex

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
            
# Helper function to parse pipe-delimited strings into lists
def parse_pipe_delimited(value, data_type='str'):
    """Convert pipe-delimited strings to proper lists"""
    if pd.isna(value) or str(value).strip() == '':
        return None
                
    items = str(value).split('|')
    items = [item.strip() for item in items if item.strip()]  # Remove empty items
                
    if not items:  # If no valid items after processing
        return None
                    
    if data_type == 'int':
        try:
            return [int(item) for item in items]
        except ValueError:
            # Fallback to string if conversion fails
            return items
    else:
        return items

def main():
    print("Starting order tokenization training...")
    
    # Initialize RAG system (not used during training, but helpful for testing after training)
    print("Initializing RAG menu index...")
    rag_index = MenuRAGIndex()

    ## Checking if RAG index is available, if not we will need to build RAG index.
    if not rag_index.load_index():
        # print("Building RAG index...")
        # rag_index.build_index()
        # print("RAG system ready!")
        print("RAG system is not available! (Note: RAG is not used during training, only for post-training testing)")
    else: print("RAG system is available!")    
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
        
        # Extract the "prompt" column for inputs (updated from "text")
        inputs = [prompt for prompt in df["prompt"]]

        # including the CUISINE and SOURCE value, which can help us point the prompt to appropriate RAG index
        # TODO - CHECK - these may not be that much helpful in custom/fine tuning the LLM ??, but more valuable for pin pointing data in RAG indexes ??
        cuisines = [cuisine for cuisine in df["cuisine"]]
        sources = [source for source in df["source"]]
        
        # breakpoint()
        
        # Combine target columns into a single JSON-like structure for each row
        targets = []
        for idx, (_, row) in enumerate(df.iterrows()):
            # Handle NaN values and convert pandas types to JSON-serializable types            
            # Process order items as structured lists instead of pipe-delimited strings
            order_items_name = parse_pipe_delimited(row["order_items_name"], 'str')
            order_items_quantity = parse_pipe_delimited(row["order_items_quantity"], 'int')
            
            # Updated - TODO: now we have modifications_plus and modifications_minus
            # Parse new modification fields (plus and minus separately)
            order_items_modifications_plus = parse_pipe_delimited(row["order_items_modifications_plus"], 'str')
            order_items_modifications_minus = parse_pipe_delimited(row["order_items_modifications_minus"], 'str')
            
            # Handle order_notes - can be either single string or list
            order_notes = row["order_notes"] if pd.notna(row["order_notes"]) else None
            # Updated - TODO/Check - I think order_notes are also pipe delimited, so we have notes for each item.

            # TODO: fix such that order_notes has a list of string or strings
            if order_notes and '|' in str(order_notes):
                order_notes = parse_pipe_delimited(order_notes, 'str')
            elif order_notes:
                order_notes = str(order_notes).strip() or None
            
            # Handle global_note
            global_note = str(row["global_note"]) if pd.notna(row["global_note"]) else None
            
            # Updated - TODO: PROCEED SCORE - Multi-task Learning Implementation (RECOMMENDED)
            # Add legitimacy validation as part of the same model output
            # 
            # Examples with proceed_score as first field:
            #   Input: "I want 2 pizzas" 
            #   Output: {"proceed_score": 1, "customer_name": null, "order_items_name": ["pizza"], ...}
            #   
            #   Input: "Hello how are you?"
            #   Output: {"proceed_score": 0, "customer_name": null, "order_items_name": null, ...}
            # 
            # Implementation Steps:
            # 1. Add 'proceed_score' column to training data TSV (0 or 1) ✓ DONE
            # 2. Include proceed_score as first field in JSON output during training ✓ IMPLEMENTING
            # 3. Model learns both tasks: order extraction AND legitimacy validation
            # 4. During inference, check proceed_score first - if 0, ignore other fields
            # 
            # Benefits:
            # - Single model handles both tasks
            # - Shared context improves both tasks
            # - Efficient inference (one forward pass)
            # - No additional deployment complexity
            
            # NOTE: proceed_score is exclusivelyl either 0 or 1
            proceed_score = int(row["proceed_score"]) if pd.notna(row["proceed_score"]) else 0
            
            #TODO: update code to make it more simple, because we will apply data filtering and validation upstream using a separate script
                # only data that are valid will come over here
                # could we add typehints over here?
            target_dict = {
                "proceed_score": proceed_score,  # Added as FIRST field - this is part of multi-task learning # type: 0 or 1 
                "customer_name": str(row["customer_name"]) if pd.notna(row["customer_name"]) else None,
                "order_type": str(row["order_type"]) if pd.notna(row["order_type"]) else None, # only 4 values: DineIn, TakeOut, Delivery, PickUp
                "total_number_of_different_items": int(row["total_number_of_different_items"]) if pd.notna(row["total_number_of_different_items"]) else None, # fix: interger value, if no items the counts should be 0
                "order_items_name": order_items_name,
                "order_items_quantity": order_items_quantity, # integer value
                # Updated - TODO: add - modifications_plus and modifications_minus 
                "order_items_modifications_plus": order_items_modifications_plus, # list of strings
                "order_items_modifications_minus": order_items_modifications_minus, # list of strings
                "order_notes": order_notes, # list of strings
                "global_note": global_note # string
            }

            # TODO: validation or a method for model to learn during training
                # we will have to have a method where the value of "total_number_of_different_items" should
                # be equal to value of len("order_items_name")
                # either here or somewhere else in the model training process

            # Ensure proper JSON formatting with consistent spacing
            json_target = json.dumps(target_dict, ensure_ascii=False, separators=(',', ':'))

            # TODO: when we are building json_target,  if the target cannot be formatted to json
                # both the input and it's target value should be taken out and then put in a separate file, so it can be looked at for potential issues 
            # Validate that the JSON is properly formatted
            try:
                json.loads(json_target)  # Validate JSON
                targets.append(json_target)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON target generated: {json_target}")
                print(f"Error: {e}")

                # NOTE: This is a good way to handle cases when the expected target cannot be convert to JSON
                # TODO - FIX - In such case it migh be just good to ignore such inputs and target??
                # Use a default valid JSON structure
                default_dict = {k: None for k in target_dict.keys()}
                targets.append(json.dumps(default_dict, ensure_ascii=False, separators=(',', ':')))

        # breakpoint()
        return inputs, targets, cuisines, sources
    

    # Improved tokenization function with proper label handling
    def tokenize_data(inputs, targets):
        """Tokenize inputs and targets for the model with proper label processing"""
        # Enhanced task prefix to provide clear context to the LLM model about expected output structure
        # Updated - TODO: When implementing proceed_score, update to:
        expected_keys = "proceed_score, customer_name, order_type, total_number_of_different_items, order_items_name, order_items_quantity, order_items_modifications_plus, order_items_modifications_minus, order_notes, global_note"
        task_description = f"Extract order information as JSON with keys: {expected_keys}."
        # TODO/Check - would adding \n help in => 
            # task_description = f"extract order information as JSON with keys: {expected_keys}. Input: "
        # task_description = f"extract order information as JSON with keys: {expected_keys}. Input: "

        
        # TODO: RESTAURANT TAG CONTEXT - Add cuisine/restaurant type context
        # NOTE: is it better to include it as uppercase PREFIX or PREFIXES or name of the RAG-Index which will be called from somewhere else?
        # Add restaurant context to improve model accuracy for specific cuisines
        # Examples:
        #   "[INDIAN] extract order information as JSON..." 
        #   "[ITALIAN] extract order information as JSON..."
        #   "[NEPALI,INDIAN] extract order information as JSON..." (multiple tags)
        # 
        # Implementation:
        # 1. Add restaurant_tags column to training data TSV (e.g., "indian|italian")
        # 2. Parse tags and add to prefix during training
        # 3. During inference, restaurant context provided by:
        #    - Website domain detection
        #    - Kiosk configuration
        #    - User selection
        # 
        # restaurant_tags = parse_pipe_delimited(row["restaurant_tags"]) if 'restaurant_tags' in df.columns else None
        # if restaurant_tags:
        #     tag_prefix = f"[{','.join(restaurant_tags).upper()}] "
        #     task_description = tag_prefix + task_description
        
        # Simple input prefixing for training (no RAG context during training)
        # prefixed_inputs = [task_description + text for text in inputs]
        prefixed_inputs = [f"CUSTOMER ORDER: {text}\nTASK: {task_description}" for text in inputs]
        
                    # prefixed_inputs.append(f"""CUSTOMER ORDER: ... + TASK: .... """)
        # breakpoint()
        
        # Note: This structured prefixing helps the model understand:
        # 1. Output should be JSON format
        # 2. Specific keys expected in the output
        # 3. Clear separation between instruction and input text
        
        # TODO: FUTURE - We will have to add new contexts and new top level training
        # Context 02: Find is legit Order Score: with expected values 0, 1
        # if the "Proceed Score" for the input (given by user) is 0, we will create no output
        # for this we may have to build another model, that works as filter, before the input is sent to "extract order" model
            # or make it part of the same model training?
        
        # Context 03: provide context if the order is from Indian or Italian or Japanese or Nepali Restaurant
        # this context could be provided as tags "indian", "nepali", "italian" (multiple allowed)
        # and added during training
        # but, when using the model, this context will be provided based on which restaurant website or kiosk is used for sending inputs
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
    # file_path = "data/sample_order_data.tsv"  # Change to your TSV file path of your choice
    file_path = "data/sample_order_data_v02_cleaned.tsv"  # Change to your TSV file path of your choice

    # breakpoint()
    
    try:
        inputs, targets, cuisines, sources = load_and_tokenize_data(file_path)
        tokenized_data = tokenize_data(inputs, targets)

        print("Data loaded and tokenized successfully!")

        # Just for self note:
        # (Pdb) tokenized_data.keys()
        # dict_keys(['input_ids', 'attention_mask', 'labels'])
        # (Pdb) len(tokenized_data['input_ids'])
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    breakpoint()

    # Initialize Accelerator
    accelerator = Accelerator()
    
    print(f"Accelerator initialized successfully.")
    print(f"Device: {accelerator.device}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Number of processes: {accelerator.num_processes}")
    
    # Create dataset
    dataset = Dataset.from_dict(tokenized_data)
    print(f"Dataset created with {len(dataset)} examples")
    # breakpoint()
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print("Model loaded successfully")
    
    # TODO: PEFT/LoRA Integration (Parameter-Efficient Fine-Tuning)
    # Add LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
    # This should be added here, before training arguments
    # 
    # from peft import LoraConfig, get_peft_model, TaskType
    # 
    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_2_SEQ_LM,  # T5 is seq2seq
    #     inference_mode=False,
    #     r=16,  # rank
    #     lora_alpha=32,  # scaling parameter
    #     lora_dropout=0.1,  # dropout probability
    #     target_modules=["q", "v"]  # which layers to apply LoRA to
    # )
    # model = get_peft_model(model, lora_config)
    # 
    # Benefits:
    # - Reduces memory usage during training
    # - Faster training with fewer parameters to update
    # - Smaller checkpoint files (only LoRA weights saved)
    # - Can create multiple specialized adapters for different restaurants
    
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
    breakpoint()
    
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
        
        # Fix task description to match training format
        expected_keys = "proceed_score, customer_name, order_type, total_number_of_different_items, order_items_name, order_items_quantity, order_items_modifications_plus, order_items_modifications_minus, order_notes, global_note"
        task_description = f"Extract order information as JSON with keys: {expected_keys}."
        
        # For testing, we can optionally use RAG context (as it would be used during inference)
        try:
            # TODO: Update RAG indexing to pull specific data based on "restaurant_name" first, then "cuisine"
            # TODO: Update RAG index to use menu_database_v3.json with different data structure
            # TODO: If restaurant name not found, pick 3 restaurant menus with matching cuisine
            
            # Try restaurant-name-first approach, then fallback to cuisine
            restaurant_name = "Little India"  # Example - this should come from user context/session
            
            # First try searching by specific restaurant name
            if hasattr(rag_index, 'search_by_restaurant_name'):
                rag_context = rag_index.get_menu_context_for_order(
                    test_input, 
                    restaurant_name=restaurant_name, 
                    cuisine="indian",  # fallback cuisine
                    max_items=3 # TODO - CHECK - is this look for 1 item to match 3 max items or overall trying to return 3 items max
                )
            else:
                # Fallback to cuisine-only search if restaurant search not available
                rag_context = rag_index.get_menu_context_for_order(
                    test_input, 
                    cuisine="indian", # TODO - CHECK - in this case would it search for all menu with cuisine = "indian" or only the first one
                    max_items=3
                )
            
            # Use RAG-enhanced format for testing (simulating inference)
            input_text = (f"MENU CONTEXT:\n"
                         f"{rag_context}\n\n"
                         f"CUSTOMER ORDER: {test_input}\nTASK: {task_description}")
        except Exception as e:
            print(f"Warning: Could not get RAG context for test: {e}")
            # Fallback to training format (no RAG context)
            input_text = f"CUSTOMER ORDER: {test_input}\nTASK: {task_description}"
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
