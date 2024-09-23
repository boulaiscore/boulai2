import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Caricamento del dataset e addestramento del modello
training_data_path = 'C:/Users/danil/Dropbox/Boulai/training_dataset.xlsx'
training_data = pd.read_excel(training_data_path, engine='openpyxl')

# Pre-elaborazione del dataset
training_data['full_description'] = training_data['full_description'].fillna('')
scaler = MinMaxScaler(feature_range=(0, 100))
y_train_scaled = scaler.fit_transform(training_data[['score']])
X_train, X_val, y_train, y_val = train_test_split(training_data['full_description'], y_train_scaled, test_size=0.2, random_state=42)

# Creazione del modello
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('regressor', SGDRegressor(loss='epsilon_insensitive', penalty=None, alpha=0.0001, max_iter=20000))
])
pipeline.fit(X_train, y_train.ravel())

# Funzione per predire il punteggio di innovazione
def predict_innovation_score(description):
    X_test = [description]
    score = pipeline.predict(X_test)[0]
    return scaler.inverse_transform([[score]])[0][0]

# Endpoint per la previsione
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data['description']
    score = predict_innovation_score(description)
    score = max(0, min(100, score))  # Assicura che il punteggio sia tra 0 e 100
    return jsonify({'innovation_score': score})

if __name__ == '__main__':
    app.run(debug=True)
