#!/usr/bin/env python3
"""
RAG Menu Database v3.0
Updated menu data loader for the new v3 database structure
Supports restaurant-name-first search with cuisine fallback
"""

import json
from typing import Dict, List, Any, Optional

def load_menu_database_v3(file_path: str = "data/menu_database_v3.json") -> Dict[str, Any]:
    """
    Load the v3 menu database from JSON file
    
    Args:
        file_path: Path to the v3 menu database JSON file
        
    Returns:
        Dictionary with restaurants organized by restaurant name and cuisine
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            restaurants = json.load(f)
        
        # Organize data for efficient lookup
        organized_data = {
            'restaurants': {},  # restro_name -> restaurant data
            'by_cuisine': {},   # cuisine -> list of restaurants
            'all_items': []     # flattened list of all menu items with restaurant info
        }
        
        for restaurant in restaurants:
            restro_name = restaurant['restro_name']
            restro_cuisine = restaurant['restro_cuisine'].lower()
            restro_id = restaurant['restro_id']
            
            # Store restaurant data
            organized_data['restaurants'][restro_name] = restaurant
            
            # Group by cuisine for fallback search
            if restro_cuisine not in organized_data['by_cuisine']:
                organized_data['by_cuisine'][restro_cuisine] = []
            organized_data['by_cuisine'][restro_cuisine].append(restaurant)
            
            # Flatten menu items with restaurant context
            for item in restaurant.get('menu', []):
                # Add restaurant context to each item
                enhanced_item = item.copy()
                enhanced_item['restaurant_name'] = restro_name
                enhanced_item['restaurant_id'] = restro_id
                enhanced_item['cuisine'] = restro_cuisine
                organized_data['all_items'].append(enhanced_item)
        
        print(f"Loaded {len(organized_data['restaurants'])} restaurants with {len(organized_data['all_items'])} menu items")
        return organized_data
        
    except FileNotFoundError:
        print(f"Error: Menu database file '{file_path}' not found")
        return {'restaurants': {}, 'by_cuisine': {}, 'all_items': []}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in menu database file: {e}")
        return {'restaurants': {}, 'by_cuisine': {}, 'all_items': []}

def get_all_menu_items_v3() -> List[Dict[str, Any]]:
    """
    Get all menu items from the v3 database with restaurant context
    
    Returns:
        List of menu items with restaurant information added
    """
    database = load_menu_database_v3()
    return database['all_items']

def get_restaurant_menu(restaurant_name: str) -> Optional[Dict[str, Any]]:
    """
    Get menu for a specific restaurant
    
    Args:
        restaurant_name: Name of the restaurant
        
    Returns:
        Restaurant data with menu, or None if not found
    """
    database = load_menu_database_v3()
    return database['restaurants'].get(restaurant_name)

def get_restaurants_by_cuisine(cuisine: str, max_restaurants: int = 3) -> List[Dict[str, Any]]:
    """
    Get restaurants by cuisine type
    
    Args:
        cuisine: Cuisine type (e.g., 'indian', 'american', 'italian')
        max_restaurants: Maximum number of restaurants to return
        
    Returns:
        List of restaurant data
    """
    database = load_menu_database_v3()
    restaurants = database['by_cuisine'].get(cuisine.lower(), [])
    return restaurants[:max_restaurants]

def search_items_by_restaurant(restaurant_name: str, query: str = None) -> List[Dict[str, Any]]:
    """
    Search menu items within a specific restaurant
    
    Args:
        restaurant_name: Name of the restaurant
        query: Optional search query to filter items
        
    Returns:
        List of matching menu items
    """
    restaurant_data = get_restaurant_menu(restaurant_name)
    if not restaurant_data:
        return []
    
    menu_items = restaurant_data.get('menu', [])
    
    # If no query, return all items
    if not query:
        return menu_items
    
    # Simple text search in name, aliases, and description
    query_lower = query.lower()
    matching_items = []
    
    for item in menu_items:
        # Check name
        if query_lower in item['name'].lower():
            matching_items.append(item)
            continue
            
        # Check aliases
        if any(query_lower in alias.lower() for alias in item.get('aliases', [])):
            matching_items.append(item)
            continue
            
        # Check description
        if query_lower in item.get('description', '').lower():
            matching_items.append(item)
            continue
    
    return matching_items

def get_schema_version() -> str:
    """Return the schema version"""
    return "3.0"

# For compatibility with existing code
MENU_DATABASE = load_menu_database_v3()

if __name__ == "__main__":
    # Demo the v3 database functionality
    print("=== Menu Database v3.0 Demo ===")
    
    database = load_menu_database_v3()
    print(f"\nTotal restaurants: {len(database['restaurants'])}")
    print(f"Total cuisines: {len(database['by_cuisine'])}")
    print(f"Total menu items: {len(database['all_items'])}")
    
    # Show restaurant names
    print("\nRestaurants available:")
    for restaurant_name in database['restaurants'].keys():
        restaurant = database['restaurants'][restaurant_name]
        print(f"  - {restaurant_name} ({restaurant['restro_cuisine']})")
    
    # Demo restaurant-specific search
    print(f"\n=== Restaurant-Specific Search Demo ===")
    little_india_items = search_items_by_restaurant("Little India", "chicken")
    print(f"'chicken' items at Little India:")
    for item in little_india_items:
        print(f"  - {item['name']}: {item['description']} (${item['price']:.2f})")
    
    # Demo cuisine-based fallback
    print(f"\n=== Cuisine Fallback Demo ===")
    indian_restaurants = get_restaurants_by_cuisine("indian", max_restaurants=2)
    print(f"Indian restaurants (max 2):")
    for restaurant in indian_restaurants:
        print(f"  - {restaurant['restro_name']}: {len(restaurant['menu'])} items")