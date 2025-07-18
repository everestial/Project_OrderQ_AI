# OrderQ AI - Order Processing System

## Project Overview

OrderQ AI is a natural language processing system that converts natural language restaurant orders into structured JSON format. The system uses a fine-tuned T5 model to extract structured information from customer orders.

## Key Features

- **Natural Language Processing**: Converts free-form order text into structured data
- **Customer Information Extraction**: Identifies customer names and order types
- **Item Parsing**: Extracts item names, quantities, and modifications
- **JSON Output**: Produces well-formatted JSON for downstream processing
- **Robust Post-Processing**: Handles and fixes common model output issues

## Model Architecture

- **Base Model**: T5-base (Text-to-Text Transfer Transformer)
- **Task**: Sequence-to-sequence generation
- **Training Data**: TSV file with natural language orders and structured targets
- **Input Format**: "extract order: [natural language order]"
- **Output Format**: JSON with order structure

## Data Structure

The system extracts the following information from orders:

```json
{
  "customer_name": "John Smith",
  "order_type": "delivery",
  "total_number_of_different_items": 2,
  "order_items_name": "pizza|diet cokes",
  "order_items_quantity": "2|3",
  "order_items_modifications": "large|extra cheese|",
  "order_notes": null
}
```

## Key Files

- `train_tokenizer.py` / `train_tokenizer.ipynb`: Main training script with proper data handling (both Python script and Jupyter notebook versions)
- `process_order.py` / `process_order.ipynb`: Production-ready order processing with robust JSON post-processing and file I/O capabilities
- `demo_complete.py` / `demo_complete.ipynb`: Complete demonstration script showcasing the entire OrderQ AI workflow
- `data/sample_order_data.tsv`: Training data in TSV format
- `data/orders.txt`: Sample input file for batch processing
- `data/processed_orders.tsv`: Output file with processed order results
- `data/processed_orders_02.tsv`: Additional processed output file

## Detailed File Descriptions

### `process_order.py` - Production Order Processing

This is the core production module that handles real-world order processing using a function-based approach. It provides robust interfaces for converting natural language orders into structured JSON and supports file-based batch processing.

**Key Features:**
- **Function-Based Design**: Clean, modular functions for easy integration
- **File I/O Support**: Read orders from text files and save results to TSV format
- **JSON Post-Processing**: Automatic fixing of common model output issues including:
  - Missing curly braces `{}`
  - Malformed patterns and invalid JSON structures
- **Error Handling**: Comprehensive error handling with detailed error reporting
- **Batch Processing**: Support for processing multiple orders efficiently
- **Production Ready**: Optimized for real-world deployment with proper resource management

**Main Functions:**
- `initialize_model(model_path)`: Load the trained model and tokenizer
- `process_order(model, tokenizer, device, order_text)`: Process a single order
- `process_orders_from_file(input_file, output_file, model, tokenizer, device)`: Batch process from file
- `validate_and_fix_json(text)`: Fix malformed JSON outputs

**Usage Example:**
```python
from process_order import initialize_model, process_order

# Initialize once
model, tokenizer, device = initialize_model('./trained_model')

# Process orders
result = process_order(model, tokenizer, device, "Hi, I'm John. I want 2 pizzas for delivery.")
if result['status'] == 'success':
    parsed_result = json.loads(result['result'])
    print(f"Customer: {parsed_result['customer_name']}")
    print(f"Items: {parsed_result['order_items_name']}")
```

**File Processing:**
```python
# Process orders from file
process_orders_from_file('data/orders.txt', 'data/processed_orders.tsv', model, tokenizer, device)
```

### `demo_complete.py` - Complete System Demonstration

This comprehensive demo script showcases the entire OrderQ AI system workflow from model loading to processing various types of orders.

**Key Features:**
- **Complete Workflow Demo**: Shows the full pipeline from model initialization to final results
- **Multiple Order Types**: Tests various order scenarios including:
  - Pizza delivery orders
  - Takeout orders
  - Dine-in orders
  - Complex multi-item orders
  - Simple orders
- **Performance Metrics**: Displays success rates and processing statistics
- **Error Demonstration**: Shows how the system handles both successful and failed processing
- **Production Usage Example**: Provides code examples for real-world integration
- **User-Friendly Output**: Formatted output with emojis and clear section separators

**Demo Order Types:**
1. **Pizza Delivery**: Complex order with multiple items and modifications
2. **Takeout Order**: Simple order with specific modifications
3. **Dine-in Order**: Order with specific service type detection
4. **Complex Order**: Multiple items with various modifications
5. **Simple Order**: Basic order to test minimal input handling

**Output Information:**
- Processing status for each order
- Extracted customer information
- Item details and modifications
- Success/failure rates
- Complete JSON output for each order
- Production integration examples

**Usage:**
```bash
python demo_complete.py
```

**Sample Output:**
```
🍕 OrderQ AI - Complete Demonstration
==================================================
✅ Found trained model

📝 Order 1: Pizza Delivery
✅ Processing successful!
🎉 Extracted Information:
   Customer: John Smith
   Order Type: delivery
   Items: pizza, diet cokes
   Quantities: 2, 3
```

## Training Process

1. **Data Loading**: Loads TSV data with proper handling of NaN values and pandas data types
2. **Tokenization**: Prepares input/output pairs for T5 model
3. **Training**: Fine-tunes T5-base for 3 epochs with proper hyperparameters
4. **Model Saving**: Saves the trained model and tokenizer

## Post-Processing

The system includes robust post-processing to handle common model output issues:

- **JSON Formatting**: Adds missing curly braces to incomplete JSON
- **Malformed Patterns**: Fixes issues like `"field": "value": "another_value"`
- **Error Handling**: Gracefully handles parsing failures

## Usage

### Training a New Model

```bash
python train_tokenizer.py
```

### Processing Orders

```python
from process_order import initialize_model, process_order
import json

# Initialize model once
model, tokenizer, device = initialize_model('./trained_model')

# Process a single order
order_text = "Hello, my name is John Smith. I'd like to order 2 large pizzas with extra cheese and 3 diet cokes please."
result = process_order(model, tokenizer, device, order_text)
print(json.dumps(result, indent=2))
```

### Processing Orders from File

```python
from process_order import initialize_model, process_orders_from_file

# Initialize model
model, tokenizer, device = initialize_model('./trained_model')

# Process orders from file and save to TSV
process_orders_from_file('data/orders.txt', 'data/processed_orders.tsv', model, tokenizer, device)
```

### Running the Demo

```bash
python process_order.py
```

### Complete System Demo

```bash
python demo_complete.py
```

## Results

The system successfully processes various order types:

### Example 1: Pizza Order
**Input**: "Hello, my name is John Smith. I'd like to order 2 large pizzas with extra cheese and 3 diet cokes please."

**Output**:
```json
{
  "customer_name": "John Smith",
  "order_type": "delivery",
  "total_number_of_different_items": 2,
  "order_items_name": "pizza|diet cokes",
  "order_items_quantity": "2|3",
  "order_items_modifications": "large|extra cheese|",
  "order_notes": null
}
```

### Example 2: Takeout Order
**Input**: "Hi, this is Sarah Johnson. I need 1 burger with no onions and 2 coffees for takeout."

**Output**:
```json
{
  "customer_name": "Sarah Johnson",
  "order_type": "takeout",
  "total_number_of_different_items": 2,
  "order_items_name": "burger|coffees",
  "order_items_quantity": "1|2",
  "order_items_modifications": "|no onions|",
  "order_notes": null
}
```

## Technical Improvements Made

1. **Fixed Data Type Issues**: Resolved pandas data type serialization problems
2. **Improved NaN Handling**: Proper conversion of NaN values to JSON null
3. **Enhanced Post-Processing**: Robust JSON fixing for malformed outputs
4. **Better Error Handling**: Graceful failure handling with detailed error information

## Performance

- **Training Time**: ~7.5 minutes on MPS (Apple Silicon)
- **Model Size**: T5-base (~220M parameters)
- **Success Rate**: High success rate with post-processing fixes
- **Processing Speed**: Fast inference suitable for real-time applications

## Future Enhancements

- **Validation**: Add data validation for extracted fields
- **Confidence Scoring**: Include confidence scores for extractions
- **Menu Integration**: Add menu item validation
- **Multi-language Support**: Extend to other languages
- **API Wrapper**: Create REST API for easy integration

## Dependencies

- transformers
- torch
- pandas
- numpy
- datasets
- accelerate

The system demonstrates successful fine-tuning of a T5 model for structured information extraction from natural language orders, with robust post-processing to handle real-world model outputs.

# Project OrderQ AI

An AI-powered project for order management and queue optimization.

## Project Setup

### Virtual Environment
This project uses a Python virtual environment to manage dependencies.

#### Setup Instructions
1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Jupyter Development Environment
The project is set up with Jupyter Lab/Notebook for interactive development.

#### Starting Jupyter
```bash
# Activate virtual environment first
source venv/bin/activate

# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

#### Custom Kernel
A custom Jupyter kernel "Project OrderQ AI" has been created for this project, which uses the project's virtual environment.

## Project Structure
```
Project_OrderQ_AI/
├── venv/                 # Virtual environment (not in git)
├── .gitignore           # Git ignore file
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Getting Started
1. Clone this repository
2. Set up the virtual environment as described above
3. Start Jupyter Lab/Notebook
4. Begin development in the interactive environment

## Contributing
Please ensure you're working within the virtual environment and that all dependencies are properly documented in `requirements.txt`.
