import pandas as pd
import re
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


# Load and preprocess the dataset
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    # Handle missing or invalid weight values (empty strings or non-numeric)
    def parse_weight(weight) -> float:
        # Ensure weight is a string, then clean and convert it
        if isinstance(weight, str):
            weight = weight.strip()
            if weight:  # Check if not empty
                try:
                    # Extract numeric value from the weight string
                    return float(re.sub(r'[^\d.]+', '', weight))
                except ValueError:
                    return 0.0  # Return 0 if conversion fails
            else:
                return 0.0  # Return 0 if the string is empty
        elif isinstance(weight, (int, float)):  # If it's already a numeric type
            return float(weight)
        return 0.0  # Return 0.0 if it's of an unexpected type
    
    df['weight'] = df['weight'].apply(parse_weight)  # Apply the parsing function to the 'weight' column
    
    # Remove the 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])  # Drop 'id' column from the DataFrame
    
    df = df[['name', 'features.value', 'weight']]  # Extract the name, ingredients, and weight
    df = df.dropna(subset=['features.value', 'weight'])  # Remove rows with missing values
    df['ingredients'] = df['features.value'].apply(lambda x: x.lower().split(','))  # Process ingredients as a list of strings
    
    # Filter out rows containing the number '8676' in any column (like manufacturerNumber, upc, etc.)
    df = df[~df.apply(lambda row: row.astype(str).str.contains('8676').any(), axis=1)]
    
    # Remove numeric values from the 'name' column
    df['name'] = df['name'].apply(lambda x: re.sub(r'\d+', '', x))  # Remove numbers (digits) from the name
    
    return df


# Generate meal recommendations based on budget and ingredients preference
def recommend_meals(budget: float, ingredients: List[str], df: pd.DataFrame) -> pd.DataFrame:
    # Filter the meals by the budget
    affordable_items = df[df['weight'] <= budget]

    if affordable_items.empty:
        return "No meals found within the given budget."
    
    # Preprocess ingredients for similarity matching
    affordable_items['ingredients_str'] = affordable_items['ingredients'].apply(lambda x: ' '.join(x))

    # Vectorize the ingredients using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(affordable_items['ingredients_str'])

    # Vectorize the user's ingredient input
    user_input = ' '.join(ingredients)
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and meal ingredients
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add similarity scores to the dataframe
    affordable_items['similarity'] = similarity_scores

    # Filter the items with similarity >= 0.5
    recommendations = affordable_items[affordable_items['similarity'] >= 0.5]

    # Sort by similarity score in descending order
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    return recommendations[['name', 'ingredients']].head(10)


# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the budget and ingredients from the form
        budget = float(request.form['budget'])
        ingredients_input = request.form['ingredients']

        if budget > 0.0 and ingredients_input:
            # Load data
            df = load_data('ingredients_v1.csv')  # Update with the correct file path

            # Parse the ingredients input
            ingredients = [ingredient.strip().lower() for ingredient in ingredients_input.split(',')]

            # Get meal recommendations
            recommendations = recommend_meals(budget, ingredients, df)

            if isinstance(recommendations, str):
                return render_template('index.html', error_message=recommendations)
            else:
                # Render recommendations in the HTML table (without 'id' column)
                return render_template('index.html', recommendations=recommendations.to_html(classes='table table-striped', index=False))

    return render_template('index.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
