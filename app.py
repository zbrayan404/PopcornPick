import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import stats
import pickle

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] {
        text-align: center;
    }
    button {
        height: auto;
        width: auto !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_data():

    # Load pickle file
    movies_selection = pickle.load(open('pickle/movies_rating.pkl', 'rb'))
    users_rating = pickle.load(open('pickle/users_rating.pkl', 'rb'))
    similarity = pickle.load(open('pickle/similarity.pkl', 'rb'))
    tfidf_model = pickle.load(open('pickle/tfidf_model.pkl','rb'))  
    tfidf_matrix = pickle.load(open('pickle/tfidf_matrix.pkl','rb')) 
    movies_meta = pickle.load(open('pickle/movies_meta.pkl','rb'))    

    # Create info_to_movieId dictionary
    info_to_movieId = dict(zip(movies_selection.apply(lambda movie: f"{movie['title']} ({movie['year']})", axis=1), movies_selection['id']))

    return movies_meta, movies_selection, info_to_movieId, users_rating, similarity, tfidf_model, tfidf_matrix

movies_meta, movies_selection, info_to_movieId, users_rating, similarity, tfidf_model, tfidf_matrix = load_data()

def similar_users(target_user, users_data):
    pearson_cor = {}
    for user_id, movies_df in users_data:
        movies_df = movies_df.sort_values(by='id')
        target_user = target_user.sort_values(by='id')
        
        movie_shared = target_user[target_user['id'].isin(movies_df['id'].tolist())]
        user_rating_list = movie_shared['rating'].tolist()
        users_rating_list = movies_df['rating'].tolist()
        
        correlation = stats.pearsonr(user_rating_list, users_rating_list)
        pearson_cor[user_id[0]] = correlation[0]
    
    return pd.DataFrame(list(pearson_cor.items()), columns=['userId', 'pearson_cor'])

def user_recommend_movies(selected_movies):
    # Filter users based on selected movies
    users = users_rating[users_rating['id'].isin(selected_movies['id'].tolist())]
    # Group users by userId
    users_id = users.groupby(['userId'])
    # Sort users by the number of selected movies rated in descending order
    users_id = sorted(users_id, key=lambda x: len(x[1]), reverse=True)
    # Select the top 100 users
    users_id = users_id[0:100]
    # Calculate Pearson correlation for similar users
    pearson_df = similar_users(selected_movies, users_id) 
    # Select the top 50 similar users based on Pearson correlation
    similar_users_df = pearson_df.sort_values(by='pearson_cor', ascending=False)[0:50]
    # Merge user ratings with similar users' information
    new_users_ratings = pd.merge(users_rating, similar_users_df, on='userId', how='inner')
    # Calculate the new rating based on Pearson correlation
    new_users_ratings['new_rating'] = new_users_ratings['pearson_cor'] * new_users_ratings['rating']
    # Aggregate ratings for recommended movies
    agg_ratings = new_users_ratings.groupby('id').agg(
        sum_rating=('new_rating', 'sum'),
        sum_pearson=('pearson_cor', 'sum'),
        number_of_ratings=('new_rating', 'count')
    ).reset_index()
    # Calculate weighted average rating for recommended movies
    agg_ratings['weighted_average'] = agg_ratings['sum_rating'] / agg_ratings['sum_pearson']
    # Merge recommended movies with movie selection information
    recommend_movies = pd.merge(movies_selection, agg_ratings, on='id', how='inner')
    # Filter recommended movies with a minimum number of ratings
    recommend_movies = recommend_movies[recommend_movies['number_of_ratings'] > 10]
    # Exclude already selected movies
    recommend_movies = recommend_movies[~recommend_movies['id'].isin(selected_movies_df['id'].tolist())]
    # Sort recommended movies by weighted average rating in descending order and select the top 10
    recommend_movies = recommend_movies.sort_values(by='weighted_average', ascending=False)[0:5]
    return recommend_movies

def content_recommend(movieId):
    # Check if the movieId is not present in the similarity index
    if movieId not in similarity.index:
        # Retrieve the movie details from movies_meta based on the given movieId
        movie = movies_meta[movies_meta['id'] == int(movieId)]
        # Check if the DataFrame 'movie' is not empty
        if not movie.empty:
            # Extract tags for the given movie
            tags = movie['tags']
            # Transform the tags using the TF-IDF model
            movie_matrix = tfidf_model.transform(tags)
            # Calculate similarity scores with other movies in the dataset
            similarity_scores = linear_kernel(movie_matrix, tfidf_matrix).flatten()
            # Create a DataFrame with movie ids and their similarity scores
            similarity_df = pd.DataFrame({'id': movies_meta['id'], 'similarity': similarity_scores})
            # Sort the movies based on similarity scores and get the top 5 recommended movies
            movies_list = similarity_df.sort_values(by='similarity', ascending=False)[1:6]['id'].tolist()
            # Return the recommended movies
            return movies_list
        else:
            print(f'Movie with id {movieId} not found in the dataset.')
        return
    # If the movieId is present in the similarity index
    movie_index = similarity.index.get_loc(movie)
    # Get the similarity scores for the input movie with other movies
    movies_list = list(similarity.iloc[movie_index])
    # Sort the movies based on similarity scores and get the top 5 recommended movies
    movies_list = sorted(list(enumerate(movies_list)), key=lambda x: x[1], reverse=True)[1:6]
    # Return the recommended movies
    return movies_list

def display_movie_rating(movieId, movie_info, rating, movies_column):
    with movies_column:
        st.write(movie_info)
        
        # Get the image path
        image_path = movies_meta[movies_meta['id'] == int(movieId)]['poster_path'].values[0]

        # Base URL for the images
        base_url = "https://image.tmdb.org/t/p/w500"

        # Constructing the full URL
        full_url = base_url + image_path

        # Display the image
        st.image(full_url)

        # Use the movie ID as the key for ratings
        rating_key = f"rating_{movieId}"
        new_rating = st.slider(f"Rating the Movie:", 0.0, 5.0, rating or 0.0, 0.1, key=rating_key)

        # Update the rating in the session state
        st.session_state.selected_movies[movieId] = (movie_info, new_rating)

        # Add a "Close" button to remove the selected movie
        if st.button(f"Close", movie_info):
            if movieId in st.session_state.selected_movies:
                del st.session_state.selected_movies[movieId]
                st.rerun()

def display_movie(movieId, movie_info, movies_column):
    with movies_column:
        st.write(movie_info)
        
        # Get the image path
        image_path = movies_meta[movies_meta['id'] == int(movieId)]['poster_path'].values[0]

        # Base URL for the images
        base_url = "https://image.tmdb.org/t/p/w500"

        # Constructing the full URL
        full_url = base_url + image_path

        # Display the image
        st.image(full_url)

def recommend_movies(movies, session):
    for index, movie in movies.iterrows():
        title = f"{movie['title']} ({int(movie['year'])})"
        movie_id = movie['id']
        session[movie_id] = title

# Initialize session_state if not present
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = {}

if "recommend_movies" not in st.session_state:
    st.session_state.recommend_movies = {}

# Display the selectbox
selected_info = st.sidebar.selectbox('Select a movie', movies_selection.apply(lambda movie: f"{movie['title']} ({movie['year']})", axis=1))

with st.sidebar:
    # Get the corresponding movieId when the "Recommend" button is clicked
    if st.button('Select'):

        if selected_info in info_to_movieId:
            selected_movieId = info_to_movieId[selected_info]

            # Check if the maximum number of boxes (5) is reached
            if len(st.session_state.selected_movies) < 5:
                # Add the selected movie to the session state with a tuple (movieId, rating)
                st.session_state.selected_movies[selected_movieId] = (selected_info, None)
    
    selected_movieId = info_to_movieId[selected_info]

    st.write(selected_info)

    image_path = movies_meta[movies_meta['id'] == int(selected_movieId)]['poster_path'].values[0]

    # Base URL for the images
    base_url = "https://image.tmdb.org/t/p/w500"

    # Constructing the full URL
    full_url = base_url + image_path

    # Display the image
    st.image(full_url)
    

# Display the selected movies and ratings in st.columns
st.title("Movie Recommendation")

# Create st.columns for movies and ratings
movies_column = st.columns(5)

# Dispaly Selected Movie
for i, (movieId, (movie_info, rating)) in enumerate(st.session_state.selected_movies.items()):
    display_movie_rating(movieId, movie_info, rating, movies_column[i % 5])

# Check if 5 movies are selected and all ratings are input to display the button for creating the DataFrame
if len(st.session_state.selected_movies) == 5:
    if st.button("Recommend"):
        # Create a DataFrame with selected movie IDs and ratings
        selected_movies_data = [(movieId, float(st.session_state.selected_movies[movieId][1])) for movieId in st.session_state.selected_movies.keys()]
        selected_movies_df = pd.DataFrame(selected_movies_data, columns=['id', 'rating'])

        st.session_state.recommend_movies = {}

        result = user_recommend_movies(selected_movies_df)
        st.write("User based Recommendation:")

        recommend_column = st.columns(5)
        recommend_movies(result,  st.session_state.recommend_movies)

        for i, (movieId, movie_info) in enumerate(st.session_state.recommend_movies.items()):
            if i < 5:
                display_movie(movieId, movie_info, recommend_column[i])

        for index, movie in selected_movies_df.iterrows():
            movie_id = movie['id']
            movie_list = content_recommend(movie_id)
            movies = movies_meta[movies_meta['id'].astype(int).isin(movie_list)]
            recommend_movies(movies,  st.session_state.recommend_movies)

        st.write("Content based Recommendation:")

        for i in range(selected_movies_df.shape[0]):
            movie_id = selected_movies_df.iloc[i]['id']
            movie = movies_selection[movies_selection['id'] == movie_id]
            movie_title = f"{movie['title'].iloc[0]} ({movie['year'].iloc[0]})"
            st.write(f"Recommendation based on {movie_title}:")

            recommend_column = st.columns(5)

            for i, (movieId, movie_info) in enumerate(list(st.session_state.recommend_movies.items())[5 + (5 * i):10 + (5 * i)]):
                display_movie(movieId, movie_info, recommend_column[i % 5])

        






