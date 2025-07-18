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
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from typing import Dict, Optional, List

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

# Initialize model and tokenizer
def initialize_model(model_path: str = "./trained_model"):
    """Load the trained model and tokenizer."""
    logger.info("Loading model and tokenizer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded.")
    return model, tokenizer, device

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

# Process a single order
def process_order(model, tokenizer, device, order_text: str) -> Dict[str, Optional[str]]:
    logger.info(f"Processing order: {order_text[:100]}...")
    input_text = "extract order: " + order_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=8, early_stopping=True)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Raw model output: {raw_output}")
    parsed_json = validate_and_fix_json(raw_output)
    if parsed_json:
        return {"status": "success", "result": json.dumps(parsed_json)}
    else:
        return {"status": "error", "raw_output": raw_output}

# Process orders from a file and save output to a TSV file
def process_orders_from_file(input_file: str, output_file: str, model, tokenizer, device):
    logger.info(f"Reading orders from {input_file}")
    with open(input_file, 'r') as f:
        order_texts = [line.strip() for line in f if line.strip()]
    
    results = []
    for order_text in order_texts:
        result = process_order(model, tokenizer, device, order_text)
        results.append(result)
    
    df = pd.DataFrame(results)
    logger.info(f"Saving results to {output_file}")
    df.to_csv(output_file, sep='\t', index=False)

# Main function for demonstration
def main():
    model_path = "./trained_model"
    input_file = "data/orders.txt"
    output_file = "data/processed_orders.tsv"

    model, tokenizer, device = initialize_model(model_path)
    process_orders_from_file(input_file, output_file, model, tokenizer, device)
    logger.info("Order processing completed.")

if __name__ == "__main__":
    main()

