# Author: Abhijeet Behera
# Date: 2024-03-12
# Description: This is a demo of vector search using Mongo Atlas Vector search.
import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json 

# Initialize the vectorization model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to vectorize text
def vectorize_text(text):
    return model.encode(text).tolist()

# Function to load sample data into MongoDB
def load_sample_data(client):
    db = client['movie_database']
    collection = db['movies']

    # Check if the collection already has data
    if collection.count_documents({}) == 0:
        
        with open('data/MovieSample.json', 'r') as sample_data:
            # Sample movie data
            sample_movies = json.load(sample_data)

            # Vectorize descriptions and add to documents before inserting
            for movie in sample_movies:
                movie['vector'] = vectorize_text(movie['description'])

            collection.insert_many(sample_movies)
            st.success(f"Loaded {len(sample_movies)} sample movies into the database.")
            return sample_movies
    else:
        st.info("Sample data already loaded.")
        return list(collection.find({}, {'_id': 0, 'vector': 0}).limit(20))  # Return a few sample documents without the vector field

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

            # Load sample data and display in UI
            sample_data = load_sample_data(client)
            for movie in sample_data:
                st.write(f"Title: {movie['title']}, Genre: {', '.join(movie['genre'])}")
        except Exception as e:
            st.error(f"Failed to connect to MongoDB Atlas: {e}")
            return

        # Movie Search Section
        query = st.text_input("Enter search terms related to the movie:")

        if query:
            query_vector = vectorize_text(query)
            results = search_movies(client, query_vector)

            # Filter the results in Python based on the score
            # filtered_results = [result for result in results if result.get('score', 0) > 0.70]

            # Check if filtered_results is empty
            if results:  # This line was corrected
                for result in results:
                    st.subheader(result['title'])
                    st.write(f"Description: {result['description']}")
                    st.write(f"Genres: {', '.join(result['genre'])}")
                    st.write(f"Search Score: {result.get('score', 'Not available')}")
                    st.markdown("---")
            else:
                st.write("No movies found matching your search score criteria.")
    else:
        st.write("Please enter your MongoDB Atlas connection string to proceed.")

if __name__ == "__main__":
    main()