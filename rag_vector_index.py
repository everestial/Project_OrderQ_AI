#!/usr/bin/env python3
"""
RAG Vector Index System v3.0
Updated for menu_database_v3.json with restaurant-name-first search capability
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Import the database loader
from rag_menu_data import (
    get_all_menu_items_v3, 
    load_menu_database_v3, 
    get_restaurant_menu,
    search_items_by_restaurant,
    get_restaurants_by_cuisine,
    get_schema_version
)

class MenuRAGIndex:
    """
    RAG index for menu items using vector embeddings
    Enhanced for v3 database with restaurant-name-first search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG index with a sentence transformer model
        
        Args:
            model_name: Hugging Face sentence transformer model name
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.menu_items = []
        self.embeddings = None
        self.database = None
        self.index_file = "data/menu_rag_index_v3.pkl"
        
    def create_searchable_text(self, item: Dict[str, Any]) -> str:
        """
        Create a weighted text representation of menu item for embedding
        Prioritizes item names and aliases over descriptions
        
        Args:
            item: Menu item dictionary
            
        Returns:
            Weighted text string for embedding with name emphasis
        """
        # Primary identifiers (highest weight) - repeated multiple times
        name = item['name']
        aliases = item.get('aliases', [])
        
        # Build weighted text with repetition for emphasis
        weighted_parts = []
        
        # Repeat name 3 times for maximum weight
        weighted_parts.extend([name] * 3)
        
        # Repeat each alias 2 times for high weight
        for alias in aliases:
            weighted_parts.extend([alias] * 2)
        
        # Add category and restaurant info with medium weight
        category = item.get('category', '')
        restaurant_name = item.get('restaurant_name', '')
        cuisine = item.get('cuisine', '')
        
        if category:
            weighted_parts.append(f"{category} dish")
        if restaurant_name:
            weighted_parts.append(f"from {restaurant_name}")
        if cuisine:
            weighted_parts.append(f"{cuisine} cuisine")
        
        # Add description and other details with lower weight (single occurrence)
        description = item.get('description', '')
        if description:
            weighted_parts.append(description)
        
        # Add ingredients and dietary info with low weight
        ingredients = item.get('ingredients', [])
        dietary_tags = item.get('dietary_tags', [])
        if ingredients:
            weighted_parts.append(f"Contains: {', '.join(ingredients[:3])}")  # Limit to first 3
        if dietary_tags:
            weighted_parts.append(', '.join(dietary_tags))
        
        # Add spice level
        spice_level = item.get('spice_level', 'none')
        if spice_level and spice_level != 'none':
            weighted_parts.append(f"{spice_level} spice")
        
        return " ".join(weighted_parts)
    
    def build_index(self, save_to_file: bool = True) -> None:
        """
        Build the vector index from v3 menu database
        
        Args:
            save_to_file: Whether to save the index to disk
        """
        print(f"Building RAG index v3.0 (Schema v{get_schema_version()})...")
        
        # Load database and get all menu items
        self.database = load_menu_database_v3()
        self.menu_items = get_all_menu_items_v3()
        print(f"Processing {len(self.menu_items)} menu items from {len(self.database['restaurants'])} restaurants...")
        
        # Create searchable text for each item
        texts = [self.create_searchable_text(item) for item in self.menu_items]
        
        # Generate embeddings with weighted text
        print("Generating embeddings (with name and restaurant weighting)...")
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        if save_to_file:
            self.save_index()
        
        print("RAG index v3.0 built successfully!")
    
    def save_index(self) -> None:
        """Save the index to disk"""
        index_data = {
            'menu_items': self.menu_items,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'schema_version': get_schema_version(),
            'total_items': len(self.menu_items),
            'total_restaurants': len(self.database['restaurants']) if self.database else 0
        }
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index v3.0 saved to {self.index_file}")
    
    def load_index(self) -> bool:
        """
        Load the index from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.index_file):
            return False
        
        try:
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.menu_items = index_data['menu_items']
            self.embeddings = index_data['embeddings']
            
            # Load database for restaurant searches
            self.database = load_menu_database_v3()
            
            # Check if model matches
            if index_data.get('model_name') != self.model_name:
                print(f"Warning: Index was built with {index_data.get('model_name')}, "
                      f"but current model is {self.model_name}")
            
            # Show version info
            stored_schema = index_data.get('schema_version', '1.0')
            current_schema = get_schema_version()
            if stored_schema != current_schema:
                print(f"Note: Index uses schema v{stored_schema}, current schema is v{current_schema}")
            
            print(f"Loaded {index_data.get('total_items', len(self.menu_items))} items from "
                  f"{index_data.get('total_restaurants', 0)} restaurants")
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search_by_restaurant_name(self, restaurant_name: str, query: str = None, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for items from a specific restaurant, optionally with query matching
        
        Args:
            restaurant_name: Name of the restaurant to search within
            query: Optional search query for filtering items
            top_k: Number of results to return
            
        Returns:
            List of (item, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Filter items by restaurant name first
        restaurant_items = []
        restaurant_indices = []
        
        for idx, item in enumerate(self.menu_items):
            if item.get('restaurant_name', '').lower() == restaurant_name.lower():
                restaurant_items.append(item)
                restaurant_indices.append(idx)
        
        if not restaurant_items:
            return []  # No items found for this restaurant
        
        # If no query provided, return all items from restaurant with neutral scores
        if not query:
            return [(item, 1.0) for item in restaurant_items[:top_k]]
        
        # Search within restaurant items using query
        query_embedding = self.encoder.encode([query])
        restaurant_embeddings = self.embeddings[restaurant_indices]
        
        similarities = cosine_similarity(query_embedding, restaurant_embeddings)[0]
        
        # Sort by similarity
        sorted_pairs = sorted(zip(restaurant_items, similarities), key=lambda x: x[1], reverse=True)
        
        return [(item, float(score)) for item, score in sorted_pairs[:top_k]]
    
    def search_by_cuisine(self, cuisine: str, query: str = None, top_k: int = 5, max_restaurants: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for items by cuisine, optionally filtered by query
        
        Args:
            cuisine: Cuisine type to search within
            query: Optional search query
            top_k: Number of results to return
            max_restaurants: Maximum number of different restaurants to include
            
        Returns:
            List of (item, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Filter items by cuisine
        cuisine_items = []
        cuisine_indices = []
        restaurants_seen = set()
        
        for idx, item in enumerate(self.menu_items):
            if item.get('cuisine', '').lower() == cuisine.lower():
                restaurant_name = item.get('restaurant_name', '')
                
                # Limit number of different restaurants
                if len(restaurants_seen) < max_restaurants or restaurant_name in restaurants_seen:
                    cuisine_items.append(item)
                    cuisine_indices.append(idx)
                    restaurants_seen.add(restaurant_name)
        
        if not cuisine_items:
            return []
        
        # If no query, return diverse items from cuisine
        if not query:
            return [(item, 1.0) for item in cuisine_items[:top_k]]
        
        # Search within cuisine items using query
        query_embedding = self.encoder.encode([query])
        cuisine_embeddings = self.embeddings[cuisine_indices]
        
        similarities = cosine_similarity(query_embedding, cuisine_embeddings)[0]
        
        # Sort by similarity
        sorted_pairs = sorted(zip(cuisine_items, similarities), key=lambda x: x[1], reverse=True)
        
        return [(item, float(score)) for item, score in sorted_pairs[:top_k]]
    
    def get_menu_context_for_order(self, order_text: str, restaurant_name: str = None, cuisine: str = None, max_items: int = 5) -> str:
        """
        Get relevant menu context for an order processing task
        Uses restaurant-name-first approach, then cuisine fallback
        
        Args:
            order_text: Customer's order text
            restaurant_name: Preferred restaurant name (searched first)
            cuisine: Fallback cuisine filter if restaurant not found
            max_items: Maximum number of menu items to include in context
            
        Returns:
            Formatted context string with relevant menu items
        """
        results = []
        
        # Try restaurant-name-first approach
        if restaurant_name:
            results = self.search_by_restaurant_name(restaurant_name, order_text, top_k=max_items)
            if results:
                context_parts = [f"MENU ITEMS from {restaurant_name}:"]
            else:
                print(f"No items found for restaurant '{restaurant_name}', falling back to cuisine search")
        
        # Fallback to cuisine search if restaurant search failed or wasn't requested
        if not results and cuisine:
            results = self.search_by_cuisine(cuisine, order_text, top_k=max_items, max_restaurants=3)
            if results:
                context_parts = [f"MENU ITEMS ({cuisine.upper()} cuisine):"]
        
        # Final fallback to general search
        if not results:
            # General search across all items (existing functionality)
            query_embedding = self.encoder.encode([order_text])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in sorted_indices[:max_items]:
                item = self.menu_items[idx]
                similarity = similarities[idx]
                results.append((item, float(similarity)))
            
            context_parts = ["RELEVANT MENU ITEMS (general search):"]
        
        if not results:
            return "No relevant menu items found."
        
        # Format the context
        for item, score in results:
            cuisine_display = item.get('cuisine', 'unknown').title()
            restaurant_name_display = item.get('restaurant_name', 'Unknown Restaurant')
            
            context_parts.append(
                f"- {item['name']} ({cuisine_display}) at {restaurant_name_display}: "
                f"{item.get('description', '')} [${item.get('price', 0):.2f}] (score: {score:.3f})"
            )
        
        return "\n".join(context_parts)

def demo_v3_rag_system():
    """Demonstrate the v3 RAG system functionality"""
    print("=== RAG Menu System v3.0 Demo ===\n")
    
    # Initialize and build index
    rag_index = MenuRAGIndex()
    
    # Try to load existing index, otherwise build new one
    if not rag_index.load_index():
        print("Building new v3 index...")
        rag_index.build_index()
    
    # Demo restaurant-specific search
    print("\n=== Restaurant-Specific Search ===")
    test_order = "I want chicken tikka with naan bread"
    context = rag_index.get_menu_context_for_order(
        test_order, 
        restaurant_name="Little India", 
        cuisine="indian",
        max_items=3
    )
    print(f"Order: '{test_order}'")
    print(f"Context:\n{context}")
    
    # Demo cuisine fallback
    print(f"\n=== Cuisine Fallback (unknown restaurant) ===")
    context = rag_index.get_menu_context_for_order(
        test_order, 
        restaurant_name="Unknown Restaurant", 
        cuisine="indian",
        max_items=3
    )
    print(f"Order: '{test_order}'")
    print(f"Context:\n{context}")

if __name__ == "__main__":
    demo_v3_rag_system()