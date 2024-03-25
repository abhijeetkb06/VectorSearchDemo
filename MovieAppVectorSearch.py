# Author: Abhijeet Behera
# Date: 2024-03-12
# Description: Updated demo to show movie images in search results using MongoDB Atlas Vector search and Streamlit.
import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json 

# Initialize the vectorization model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to vectorize text
def vectorize_text(text):
    return model.encode(text).tolist()

# Function to load or update sample data into MongoDB with 'poster_url'
def load_sample_data(client):
    db = client['movie_database']
    collection = db['movies']

    # Check if the collection already has data
    if collection.count_documents({}) == 0:
        
        with open('data/MovieSample.json', 'r') as sample_data:
            sample_movies = json.load(sample_data)

            # Vectorize descriptions and add to documents before inserting
            for movie in sample_movies:
                movie['vector'] = vectorize_text(movie['description'])

            collection.insert_many(sample_movies)
            st.success(f"Loaded {len(sample_movies)} sample movies into the database.")
    else:
        st.info("Sample data already loaded.")

# MongoDB connection and search functionality
def search_movies(client, query_vector):
    db = client['movie_database']
    collection = db['movies']
    search_results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vectorSearchIndex",
                "path": "vector",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5
            }
        },
        {
            "$project": {
                "title": 1,
                "description": 1,
                "genre": 1,
                "poster_url": 1,  # Ensure to project 'poster_url'
                "_id": 0,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])
    return list(search_results)

# Streamlit UI components
def main():
    st.title("Movie Search App Powered by Vector Search")

    # MongoDB Atlas Connection Configuration
    connection_string = st.text_input("MongoDB Atlas connection string:")

    if connection_string:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        try:
            # Test connection
            client.server_info()
            st.markdown("<span style='color:green'>Connected</span>", unsafe_allow_html=True)

            # Load or update sample data
            load_sample_data(client)

        except Exception as e:
            st.error(f"Failed to connect to MongoDB Atlas: {e}")
            return

        # Movie Search Section
        query = st.text_input("Enter search terms related to the movie:")

        if query:
            query_vector = vectorize_text(query)
            results = search_movies(client, query_vector)

            if results:
                for result in results:
                    st.subheader(result['title'])
                    st.image(result['poster_url'], width=200)  # Display movie poster
                    st.write(f"Description: {result['description']}")
                    st.write(f"Genres: {', '.join(result['genre'])}")
                    st.write(f"Search Score: {result.get('score', 'Not available')}")
                    st.markdown("---")
            else:
                st.write("No movies found matching your search criteria.")
    else:
        st.write("Please enter your MongoDB Atlas connection string to proceed.")

if __name__ == "__main__":
    main()
