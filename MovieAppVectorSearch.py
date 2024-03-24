# Author: Abhijeet Behera
# Date: 2024-03-12
# Description: This is a demo of vector search using Mongo Atlas Vector search.
import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

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
        # Sample movie data
        sample_movies = [
                {
                    "title": "Inception",
                    "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
                    "genre": ["Action", "Adventure", "Sci-Fi"]
                },
                {
                    "title": "The Shawshank Redemption",
                    "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                    "genre": ["Drama"]
                },
                {
                    "title": "The Godfather",
                    "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                    "genre": ["Crime", "Drama"]
                },
                {
                    "title": "Pulp Fiction",
                    "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
                    "genre": ["Crime", "Drama"]
                },
                {
                    "title": "The Dark Knight",
                    "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
                    "genre": ["Action", "Crime", "Drama"]
                },
                {
                    "title": "Forrest Gump",
                    "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.",
                    "genre": ["Drama", "Romance"]
                },
                {
                    "title": "Fight Club",
                    "description": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
                    "genre": ["Drama"]
                },
                {
                    "title": "Psycho",
                    "description": "A Phoenix secretary embezzles forty thousand dollars from her employer's client, goes on the run, and checks into a remote motel run by a young man under the domination of his mother.",
                    "genre": ["Horror", "Mystery", "Thriller"]
                },
                {
                    "title": "Parasite",
                    "description": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
                    "genre": ["Comedy", "Drama", "Thriller"]
                },
                {
                    "title": "La La Land",
                    "description": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.",
                    "genre": ["Comedy", "Drama", "Music"]
                },
                {
                    "title": "The Matrix",
                    "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
                    "genre": ["Action", "Sci-Fi"]
                },
                {
                    "title": "Titanic",
                    "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
                    "genre": ["Drama", "Romance"]
                },
                {
                    "title": "Jurassic Park",
                    "description": "During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok.",
                    "genre": ["Adventure", "Sci-Fi", "Thriller"]
                },
                {
                    "title": "The Silence of the Lambs",
                    "description": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.",
                    "genre": ["Crime", "Drama", "Thriller"]
                },
                {
                    "title": "Se7en",
                    "description": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.",
                    "genre": ["Crime", "Drama", "Mystery"]
                },
                {
                    "title": "It's a Wonderful Life",
                    "description": "An angel is sent from Heaven to help a desperately frustrated businessman by showing him what life would have been like if he had never existed.",
                    "genre": ["Drama", "Family", "Fantasy"]
                },
                {
                    "title": "Whiplash",
                    "description": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
                    "genre": ["Drama", "Music"]
                },
                {
                    "title": "The Prestige",
                    "description": "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.",
                    "genre": ["Drama", "Mystery", "Sci-Fi"]
                },
                {
                    "title": "Gladiator",
                    "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.",
                    "genre": ["Action", "Adventure", "Drama"]
                },
                {
                    "title": "Interstellar",
                    "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
                    "genre": ["Adventure", "Drama", "Sci-Fi"]
                }
        ]

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