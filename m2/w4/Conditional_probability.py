import pandas as pd

# Load the generated dataset
df = pd.read_csv("Synthetic_movie_dataset_200_users.csv")

# Function to calculate conditional probability for each user
def calculate_user_conditional_probabilities(movie_column, target_column):
    # Calculate the overall P(Movie) for the dataset
    total_users = len(df)
    p_movie = df[movie_column].sum()    # P(Movie)

    # Calculate P(Movie and SquidGame-2)
    joint_prob = ((df[movie_column] == 1) & (df[target_column] == 1)).sum()   # P(Movie and SquidGame-2)

    # For each user, calculate the conditional probability P(SquidGame-2 | Movie)
    prob_column = []
    for _, row in df.iterrows():
        if row[movie_column] == 1:  # Only calculate for users who watched the given movie
            prob = joint_prob / p_movie
        else:
            prob = 0  # If the user didn't watch the movie, the conditional probability is 0
        prob_column.append(prob)
    
    return prob_column

# List of columns for the movies (excluding SquidGame-2)
movies = ['SquidGame-1', 'The8Show', 'Hellbound', 'AliceInBorderland']

# Calculate conditional probabilities for each user and each movie
for movie in movies:
    df[f'P(SquidGame-2 | {movie})'] = calculate_user_conditional_probabilities(movie, 'SquidGame-2')

# Display the dataset with the calculated conditional probabilities for each user
print(df[['User_ID'] + [f'P(SquidGame-2 | {movie})' for movie in movies]].head())

# Optionally, save to CSV for further use
df.to_csv("user_conditional_probabilities_squidgame.csv", index=False)

