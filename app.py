from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pickled objects
with open("recipe_dataframe.pkl", "rb") as f:
    df = pickle.load(f)

with open(r"C:\Users\Tanmay\Desktop\Tanmay\Projects\Mini_project_recipe\tfidf_vectorizer (1).pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Search function
def search_recipes(recipe_name='', ingredients='', total_time='', allergens='', diet='', top_n=5):
    query = f"{recipe_name} {ingredients} {allergens} {diet} {total_time}"
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    return df.loc[top_indices, [
        'TranslatedRecipeName',
        'TranslatedIngredients',
        'TranslatedInstructions',
        'TotalTimeInMins',
        'CookTimeInMins',
        'PrepTimeInMins',
        'Allergens',
        'Diet',
        'Nutrition (per 100g)'
    ]]

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    recipes = None
    if request.method == 'POST':
        recipe_name = request.form.get('recipe_name', '')
        ingredients = request.form.get('ingredients', '')
        total_time = request.form.get('total_time', '')
        allergens = request.form.get('allergens', '')
        diet = request.form.get('diet', '')

        results = search_recipes(recipe_name, ingredients, total_time, allergens, diet)
        recipes = results.to_dict(orient='records')

    return render_template('index.html', recipes=recipes)

if __name__ == '__main__':
    app.run(debug=True)
