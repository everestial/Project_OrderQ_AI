#!/usr/bin/env python3
"""
Order Processing Script for OrderQ AI

This script loads the trained T5 model and processes natural language restaurant orders
to generate structured JSON output, handling JSON formatting issues. It can also process
orders from a text file and save output to a TSV file.

Usage:
    python process_order.py
"""

import json
import torch
import re
import logging
import pandas as pd
import argparse
import sys
from pathlib import Path
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from typing import Dict, Optional, List

# RAG system imports
from rag_vector_index import MenuRAGIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize model, tokenizer, and RAG system
def initialize_model(model_path: str = "./trained_model"):
    """Load the trained model, tokenizer, and RAG index."""
    logger.info("Loading model and tokenizer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    
    # Initialize RAG system
    logger.info("Initializing RAG menu index...")
    rag_index = MenuRAGIndex()
    if not rag_index.load_index():
        logger.warning("RAG index not found, building new one...")
        rag_index.build_index()
    logger.info("RAG system ready!")
    
    # TODO: PEFT/LoRA Model Loading
    # If model was trained with LoRA, load the adapter weights:
    # 
    # from peft import PeftModel
    # model = PeftModel.from_pretrained(model, model_path)
    # 
    # This enables loading LoRA adapter weights on top of base model
    
    logger.info("Model, tokenizer, and RAG index loaded.")
    return model, tokenizer, device, rag_index

# Validate and fix JSON output
def validate_and_fix_json(text: str) -> Optional[Dict]:
    """Validate and fix JSON output."""
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass
    
    fixed_text = text.strip()
    if not fixed_text.startswith('{'):
        fixed_text = '{' + fixed_text
    if not fixed_text.endswith('}'):
        fixed_text = fixed_text + '}'

    try:
        fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
        fixed_text = fixed_text.replace("'", '"')
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
        result = json.loads(fixed_text)
        return result
    except json.JSONDecodeError:
        return None

# Process a single order with RAG context
def process_order(model, tokenizer, device, rag_index, order_text: str, cuisine: str = None, restaurant_tags: List[str] = None) -> Dict[str, Optional[str]]:
    logger.info(f"Processing order: {order_text[:100]}...")
    
    # TODO: PROCEED SCORE VALIDATION - Multi-task Output Processing
    # After model inference, check proceed_score from JSON output:
    # 
    # if parsed_json and 'proceed_score' in parsed_json:
    #     if parsed_json['proceed_score'] == 0:
    #         return {"status": "skipped", "reason": "Not a legitimate order"}
    # 
    # This approach:
    # - Uses the same model for both tasks
    # - proceed_score comes from model output, not pre-processing
    # - Model can make contextual decisions about legitimacy
    # - Single inference call handles everything
    
    # Use RAG-enhanced processing that matches training format
    expected_keys = "proceed_score, customer_name, order_type, total_number_of_different_items, order_items_name, order_items_quantity, order_items_modifications_plus, order_items_modifications_minus, order_notes, global_note"
    task_description = f"extract order information as JSON with keys: {expected_keys}. Input: "
    
    # Generate RAG context for this order (matching training format)
    try:
        rag_context = rag_index.get_menu_context_for_order(order_text, cuisine, max_items=3)
        
        # Create RAG-enhanced input (matching training format)
        input_text = f"""MENU CONTEXT:
{rag_context}

CUSTOMER ORDER:
{order_text}

TASK: Extract order information as JSON with keys: {expected_keys}"""
        
        logger.info(f"RAG context generated for cuisine: {cuisine}")
        logger.debug(f"RAG context: {rag_context}")
        
    except Exception as e:
        logger.warning(f"RAG context generation failed: {e}. Using basic processing.")
        # Fallback to basic processing without RAG
        input_text = task_description + order_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=8, early_stopping=True)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Raw model output: {raw_output}")
    parsed_json = validate_and_fix_json(raw_output)
    breakpoint()

    if parsed_json:
        return {"status": "success", "result": json.dumps(parsed_json)}
    else:
        return {"status": "error", "raw_output": raw_output}

# Process orders from a file and save output to a TSV file
def process_orders_from_file(input_file: str, output_file: str, model, tokenizer, device, rag_index, cuisine: str = None, restaurant_tags: List[str] = None):
    logger.info(f"Reading orders from {input_file}")
    with open(input_file, 'r') as f:
        order_texts = [line.strip() for line in f if line.strip()]
    
    results = []
    for order_text in order_texts:
        result = process_order(model, tokenizer, device, rag_index, order_text, cuisine, restaurant_tags)
        results.append(result)
    breakpoint()

    df = pd.DataFrame(results)
    logger.info(f"Saving results to {output_file}")
    df.to_csv(output_file, sep='\t', index=False)

# Main function with command line interface
def main():
    parser = argparse.ArgumentParser(
        description="Process restaurant orders using OrderQ AI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python process_order.py -i data/orders.txt -o data/results.tsv
  python process_order.py --model custom_model/ --input orders.txt
  python process_order.py --single "I want 2 pizzas and 3 cokes"
  python process_order.py -s "Dal bhat and momo" -r nepali,indian
  python process_order.py -i orders.txt -r italian --verbose
        """
    )
    
    parser.add_argument(
        "-m", "--model", 
        default="./trained_model",
        help="Path to the trained model directory (default: ./trained_model)"
    )
    
    parser.add_argument(
        "-i", "--input",
        help="Input file containing orders (one per line)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output TSV file for results (default: processed_orders.tsv)"
    )
    
    parser.add_argument(
        "-s", "--single",
        help="Process a single order text directly"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "-r", "--restaurant-tags",
        help="Comma-separated restaurant/cuisine tags (e.g., indian,nepali)"
    )
    
    parser.add_argument(
        "-c", "--cuisine",
        help="Cuisine type for RAG context (e.g., indian, american, italian)"
    )
    
    args = parser.parse_args()
    
    # Parse restaurant tags
    restaurant_tags = None
    if args.restaurant_tags:
        restaurant_tags = [tag.strip() for tag in args.restaurant_tags.split(',') if tag.strip()]
        logger.info(f"Using restaurant tags: {restaurant_tags}")
    
    # Parse cuisine type
    cuisine = args.cuisine
    if cuisine:
        logger.info(f"Using cuisine type for RAG: {cuisine}")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.single and not args.input:
        parser.error("Either --input file or --single order text must be provided")
    
    if not Path(args.model).exists():
        logger.error(f"Model path does not exist: {args.model}")
        sys.exit(1)
    
    # Initialize model
    try:
        model, tokenizer, device, rag_index = initialize_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Process single order
    if args.single:
        logger.info(f"Processing single order: {args.single}")
        result = process_order(model, tokenizer, device, rag_index, args.single, cuisine, restaurant_tags)
        
        if result["status"] == "success":
            print("✅ Successfully processed order:")
            parsed_json = json.loads(result["result"])
            print(json.dumps(parsed_json, indent=2))
        else:
            print("❌ Failed to process order:")
            print(f"Raw output: {result.get('raw_output', 'No output')}")
        return
    
    # Process file
    if args.input:
        if not Path(args.input).exists():
            logger.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        
        # Set default output filename if not provided
        output_file = args.output or f"processed_{Path(args.input).stem}.tsv"
        
        logger.info(f"Processing orders from: {args.input}")
        logger.info(f"Output will be saved to: {output_file}")
        
        try:
            process_orders_from_file(args.input, output_file, model, tokenizer, device, rag_index, cuisine, restaurant_tags)
            logger.info("Order processing completed successfully!")
        except Exception as e:
            logger.error(f"Failed to process orders: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()

