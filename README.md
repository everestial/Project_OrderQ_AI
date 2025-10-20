# OrderQ AI - Order Processing System

## Project Overview

OrderQ AI is a comprehensive natural language processing system that converts natural language restaurant orders into structured JSON format. The system combines:
- **Fine-tuned T5 model** for order extraction and legitimacy scoring
- **RAG (Retrieval-Augmented Generation)** for menu-aware order processing
- **Restaurant-name-first search** with cuisine fallback for multi-restaurant support

## Key Features

- **Natural Language Processing**: Converts free-form order text into structured data
- **Multi-task Learning**: Order extraction + legitimacy scoring (proceed_score)
- **RAG-Enhanced Processing**: Menu-aware order processing with context
- **Restaurant-First Search**: Prioritizes specific restaurant menus, falls back to cuisine
- **Customer Information Extraction**: Identifies customer names and order types
- **Advanced Item Parsing**: Extracts item names, quantities, modifications (plus/minus)
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
  "proceed_score": 1,
  "customer_name": "John Smith",
  "order_type": "delivery",
  "total_number_of_different_items": 2,
  "order_items_name": ["pizza", "diet cokes"],
  "order_items_quantity": [2, 3],
  "order_items_modifications_plus": [["large", "extra cheese"], []],
  "order_items_modifications_minus": [[], []],
  "order_notes": ["crispy crust", null],
  "global_note": "Please deliver to front door"
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

## RAG Index Setup

### Step 1: Prepare Menu Database
Ensure you have a properly formatted JSON file (`data/menu_database_v3.json`):
```json
[
  {
    "restro_id": "REST_20180",
    "restro_name": "Little India", 
    "restro_cuisine": "INDIAN",
    "menu": [
      {
        "item_id": "ITM_10623",
        "name": "Chicken Tikka Masala",
        "aliases": ["tikka", "chicken tikka"],
        "description": "Grilled chicken in spiced curry sauce",
        "price": 19.81,
        "category": "main_courses",
        "dietary_tags": ["spicy"],
        "ingredients": ["chicken", "tomato", "spices"]
      }
    ]
  }
]
```

**Test the menu database structure:**
```bash
python rag_menu_data.py
```
This will show available restaurants, cuisines, and demo restaurant-specific searches.

### Step 2: Build RAG Vector Index
```python
from rag_vector_index import MenuRAGIndex

# Initialize RAG system
rag_index = MenuRAGIndex()

# Build index from menu database
rag_index.build_index()  # Reads from data/menu_database_v3.json

# Index is automatically saved to data/menu_rag_index_v3.pkl
```

**Test the RAG vector index:**
```bash
python rag_vector_index.py
```
This will build the index (if needed) and demo restaurant-first searches.

### Step 3: RAG Search Capabilities

**Restaurant-name-first search:**
```python
from rag_vector_index import MenuRAGIndex

# Initialize RAG system
rag_index = MenuRAGIndex()
rag_index.load_index()  # or build_index() if not exists

# Restaurant-specific search (preferred approach)
context = rag_index.get_menu_context_for_order(
    "I want chicken tikka with naan",
    restaurant_name="Little India",  # Searches here first
    cuisine="indian",                # Falls back to this
    max_items=3
)
print(context)
# Output:
# MENU ITEMS from Little India:
# - Chicken Tikka Masala (Indian) at Little India: Grilled chicken in spiced curry sauce [$.19.81] (score: 0.856)
# - Garlic Naan (Indian) at Little India: Soft flatbread with garlic and herbs [$4.88] (score: 0.743)
```

**Cuisine-based search (fallback):**
```python
# When restaurant not found, searches by cuisine across multiple restaurants
context = rag_index.get_menu_context_for_order(
    "spicy chicken with rice",
    restaurant_name="Unknown Restaurant",  # Not found
    cuisine="indian",                      # Falls back to this
    max_items=3
)
print(context)
# Output:
# MENU ITEMS (INDIAN cuisine):
# - Chicken Vindaloo (Indian) at Little India: Spicy Goan curry with chicken [$17.99] (score: 0.892)
# - Chicken Tikka Masala (Indian) at Little India: Grilled chicken in spiced curry sauce [$19.81] (score: 0.834)
# - Lamb Biryani (Indian) at Little India: Aromatic basmati rice layered with spiced lamb [$21.50] (score: 0.756)
```

**Direct restaurant search:**
```python
# Search within specific restaurant only
results = rag_index.search_by_restaurant_name("Little India", "chicken", top_k=3)
for item, score in results:
    print(f"{item['name']}: {item['description']} (${item['price']:.2f}) - Score: {score:.3f}")
# Output:
# Chicken Tikka Masala: Grilled chicken in spiced curry sauce ($19.81) - Score: 0.856
# Butter Chicken: Tender chicken in creamy tomato sauce ($19.96) - Score: 0.823
# Tandoori Wings: Chicken wings marinated in tandoori spices ($10.61) - Score: 0.789
```

**Cuisine search across restaurants:**
```python
# Search by cuisine across multiple restaurants (max 3 restaurants)
results = rag_index.search_by_cuisine("indian", "vegetarian curry", top_k=3)
for item, score in results:
    print(f"{item['name']} at {item['restaurant_name']}: ${item['price']:.2f} - Score: {score:.3f}")
# Output:
# Palak Paneer at Little India: $16.48 - Score: 0.878
# Dal Makhani at Little India: $15.08 - Score: 0.845
# Paneer Tikka at Spice Garden: $14.99 - Score: 0.812
```

### RAG Data Flow Architecture

🔄 **The Complete Data Flow:**
```
menu_database_v3.json 
    ↓
rag_menu_data.py (loads & organizes data)
    ↓  
rag_vector_index.py (creates embeddings)
    ↓
RAG Index (saved as .pkl file)
    ↓
Inference (provides context to trained model)
```

🎯 **Key Point:**
`rag_menu_data.py` is the **data layer** that:
- Loads the JSON menu database
- Organizes it for efficient RAG operations  
- Provides restaurant-first search capabilities
- Enables the RAG system to work with the v3 database structure

**It's essential for RAG functionality but completely separate from LLM training!**

❌ **Where it is NOT used:**

1. **LLM Training** (`train_tokenizer.py`):
   • Training uses only the TSV data file
   • No menu database involved during training  
   • Model learns extraction patterns, not specific menu items

### Training vs Inference Separation

**During Training:** 
- ✅ Uses TSV training data only
- ❌ No RAG context included
- 🎯 Model learns extraction patterns, not specific menu items

**During Inference:**
- ✅ Uses trained model + RAG context
- ✅ RAG provides menu-aware context from restaurant database
- 🎯 Best of both worlds: learned patterns + current menu data

## Training Process

1. **Data Loading**: Loads TSV data with proper handling of NaN values and pandas data types
2. **Tokenization**: Prepares input/output pairs for T5 model (no RAG context during training)
3. **Training**: Fine-tunes T5-base for order extraction + legitimacy scoring
4. **Model Saving**: Saves the trained model and tokenizer

## Post-Processing

The system includes robust post-processing to handle common model output issues:

- **JSON Formatting**: Adds missing curly braces to incomplete JSON
- **Malformed Patterns**: Fixes issues like `"field": "value": "another_value"`
- **Error Handling**: Gracefully handles parsing failures

## TESTING and USAGE

### Quick Test Suite

**1. Test Menu Database:**
```bash
python rag_menu_data.py
```
Expected output: List of restaurants, cuisines, and demo searches

**2. Test RAG Vector Index:**
```bash
python rag_vector_index.py  
```
Expected output: Index building progress, restaurant-specific searches, cuisine fallback demos

**3. Test Training Data Loading:**
```bash
python -c "from train_tokenizer import main; main()" # Will stop at breakpoint for inspection
```

### Production Workflow

**Step 1: Build RAG Index (one-time setup)**
```python
from rag_vector_index import MenuRAGIndex

rag_index = MenuRAGIndex()
rag_index.build_index()  # Takes ~2-3 minutes, saves to data/menu_rag_index_v3.pkl
```

**Step 2: Train Model (if needed)**
```bash
python train_tokenizer.py  # Takes ~10-15 minutes on MPS
```

**Step 3: Process Orders with RAG Context**
```python
from process_order import initialize_model, process_order_with_rag
from rag_vector_index import MenuRAGIndex

# Initialize systems
model, tokenizer, device = initialize_model('./trained_model')
rag_index = MenuRAGIndex()
rag_index.load_index()

# Process order with restaurant-first RAG context
result = process_order_with_rag(
    model, tokenizer, device, rag_index,
    "Hi, I'm John. I want chicken tikka and naan from Little India",
    restaurant_name="Little India",
    cuisine="indian"
)
```

### Example Outputs

**Restaurant-specific order:**
```json
{
  "proceed_score": 1,
  "customer_name": "John", 
  "order_type": null,
  "total_number_of_different_items": 2,
  "order_items_name": ["Chicken Tikka Masala", "Garlic Naan"],
  "order_items_quantity": [1, 1],
  "order_items_modifications_plus": [[], []],
  "order_items_modifications_minus": [[], []],
  "order_notes": [null, null],
  "global_note": null
}
```

**Invalid order (proceed_score = 0):**
```json
{
  "proceed_score": 0,
  "customer_name": null,
  "order_type": null, 
  "total_number_of_different_items": null,
  "order_items_name": null,
  "order_items_quantity": null,
  "order_items_modifications_plus": null,
  "order_items_modifications_minus": null,
  "order_notes": null,
  "global_note": null
}
```

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

### Immediate Improvements
- **Validation**: Add data validation for extracted fields
- **Confidence Scoring**: Include confidence scores for extractions
- **Multi-language Support**: Extend to other languages
- **API Wrapper**: Create REST API for easy integration

### Chatbot/Agentic AI Architecture (Future Implementation)

**Yes, `rag_menu_data.py` would be ESSENTIAL for chatbot implementation!**

#### How RAG Components Enable Chatbots:

**1. Menu Queries & Recommendations:**
```python
# Customer: "What chicken dishes do you have?"
from rag_menu_data import search_items_by_restaurant
chicken_items = search_items_by_restaurant("Little India", "chicken")
# Chatbot responds with available chicken dishes, prices, descriptions
```

**2. Menu Validation:**
```python
# Customer: "I want chicken tikka"
# Chatbot validates item exists in restaurant's menu
rag_context = rag_index.get_menu_context_for_order("chicken tikka", restaurant_name="Little India")
if rag_context: # Item found
    proceed_with_order()
```

**3. Dynamic Menu Updates:**
```python
# Update menu database without retraining model
# Chatbot immediately knows about new items, prices, availability
```

#### Chatbot Architecture Components:

**Core Components:**
- **OrderQ AI Model** (current) - Order extraction & legitimacy scoring
- **RAG System** (current) - Menu awareness & context
- **Dialogue Manager** (future) - Conversation flow & state management
- **Intent Classifier** (future) - Understanding user intentions beyond orders
- **Response Generator** (future) - Natural conversation responses

**Chatbot Data Flow:**
```
User Input → Intent Classification → Route to Handler:
├── Order Intent → OrderQ AI + RAG → Structured Order
├── Menu Query → RAG Search → Menu Information  
├── Greeting/Help → Template Response
└── Clarification → Context-aware Response
```

**Implementation Phases:**

**Phase 1: Enhanced Order Processing**
- Integrate current OrderQ AI + RAG for order extraction
- Add menu validation and recommendations
- Implement basic clarification questions

**Phase 2: Conversational Interface**
- Add dialogue state management
- Implement intent classification (order, query, help, etc.)
- Create response templates and natural language generation

**Phase 3: Agentic Capabilities**
- Multi-turn conversation handling
- Proactive recommendations based on order history
- Integration with POS/payment systems
- Personalization and memory

**Phase 4: Advanced AI Agent**
- Context-aware upselling
- Dynamic pricing integration
- Multi-restaurant coordination
- Advanced natural language understanding

#### Technologies for Chatbot Extension:
- **Rasa/Dialogflow**: Conversation management
- **LangChain**: LLM orchestration and memory
- **FastAPI**: Real-time API endpoints
- **WebSocket**: Real-time chat interface
- **Redis**: Session and state management

**Note**: The current OrderQ AI + RAG system provides the **foundation** for a restaurant ordering chatbot. The RAG menu system becomes the **knowledge base** that enables menu-aware conversations!

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
