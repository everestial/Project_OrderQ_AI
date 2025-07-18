#!/usr/bin/env python3
"""
Complete demonstration of the OrderQ AI system
Shows training, testing, and production usage
"""

import json
import os
from process_order import initialize_model, process_order

def main():
    print("🍕 OrderQ AI - Complete Demonstration")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("./trained_model"):
        print("❌ Trained model not found!")
        print("Please run: python train_tokenizer.py")
        return
    
    print("✅ Found trained model")
    
    # Initialize the model
    print("\n🔧 Initializing model...")
    model, tokenizer, device = initialize_model("./trained_model")
    
    # Demo orders covering different scenarios
    demo_orders = [
        {
            "name": "Pizza Delivery",
            "text": "Hello, my name is John Smith. I'd like to order 2 large pizzas with extra cheese and 3 diet cokes for delivery please."
        },
        {
            "name": "Takeout Order",
            "text": "Hi, this is Sarah Johnson. I need 1 burger with no onions and 2 coffees for takeout."
        },
        {
            "name": "Dine-in Order",
            "text": "Good evening, I'm Mike Davis. Can I get 3 chicken sandwiches with extra pickles and 1 salad for dine-in?"
        },
        {
            "name": "Complex Order",
            "text": "Hi, Anna Wilson here. I want 4 large pizzas with extra cheese and pepperoni, 2 diet cokes, 3 regular cokes, and 1 salad with no tomatoes for delivery."
        },
        {
            "name": "Simple Order",
            "text": "Hello, Tom Brown. Just 2 burgers please."
        }
    ]
    
    print(f"\n🎯 Processing {len(demo_orders)} demo orders...")
    print("=" * 50)
    
    successful_orders = 0
    
    for i, order in enumerate(demo_orders, 1):
        print(f"\n📝 Order {i}: {order['name']}")
        print(f"Input: {order['text']}")
        print("-" * 40)
        
        # Process the order
        result = process_order(model, tokenizer, device, order['text'])
        
        # Check if successful
        if result['status'] == 'success':
            successful_orders += 1
            print("✅ Processing successful!")
            print(f"🎉 Extracted Information:")
            
            # Parse the JSON result
            parsed_result = json.loads(result['result'])
            
            print(f"   Customer: {parsed_result.get('customer_name', 'Unknown')}")
            print(f"   Order Type: {parsed_result.get('order_type', 'Unknown')}")
            print(f"   Items: {parsed_result.get('order_items_name', 'Unknown').replace('|', ', ')}")
            print(f"   Quantities: {parsed_result.get('order_items_quantity', 'Unknown').replace('|', ', ')}")
            
            modifications = parsed_result.get('order_items_modifications', '')
            if modifications and modifications != '':
                mods = modifications.split('|')
                mods_clean = [mod for mod in mods if mod.strip()]
                if mods_clean:
                    print(f"   Modifications: {', '.join(mods_clean)}")
        else:
            print("❌ Processing failed!")
            print(f"   Error: {result.get('raw_output', 'Unknown error')}")
        
        print(f"\n📋 Full JSON Output:")
        print(json.dumps(result, indent=2))
        print("=" * 50)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"Total Orders: {len(demo_orders)}")
    print(f"Successful: {successful_orders}")
    print(f"Failed: {len(demo_orders) - successful_orders}")
    print(f"Success Rate: {successful_orders/len(demo_orders)*100:.1f}%")
    
    # Production usage example
    print(f"\n🚀 Production Usage Example:")
    print("=" * 50)
    
    print("""
# Example usage in production:

from process_order import initialize_model, process_order

# Initialize once
model, tokenizer, device = initialize_model('./trained_model')

# Process orders as they come in
def handle_customer_order(order_text):
    result = process_order(model, tokenizer, device, order_text)
    
    if result['status'] == 'success':
        # Success - use the structured data
        parsed_result = json.loads(result['result'])
        customer_name = parsed_result['customer_name']
        order_type = parsed_result['order_type']
        items = parsed_result['order_items_name'].split('|')
        quantities = [int(q) for q in parsed_result['order_items_quantity'].split('|')]
        
        # Process the order...
        return {"status": "success", "data": parsed_result}
    else:
        # Handle error
        return {"status": "error", "message": result.get('raw_output', 'Unknown error')}

# Example call
# response = handle_customer_order("Hi, I'm John. I want 2 pizzas.")
    """)
    
    print("\n🎉 Demo completed successfully!")
    print("The OrderQ AI system is ready for production use!")

if __name__ == "__main__":
    main()
