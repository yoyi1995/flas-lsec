import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import os

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo entrenado (usa una ruta relativa)
model = tf.keras.models.load_model('modelo_landmarks.keras')

# Función para detectar la letra a partir de los landmarks
def detect_letter(landmarks):
    # Convertir landmarks a un array de NumPy y asegurarse de que tenga la forma correcta
    input_data = np.array(landmarks).reshape(1, -1)
    if input_data.shape[1] != 63:
        raise ValueError(f"El modelo espera una entrada con 63 valores, pero se recibió {input_data.shape[1]}.")

    # Realizar la predicción
    predictions = model.predict(input_data)
    predicted_letter_index = np.argmax(predictions)

    # Convertir el índice a letra
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Ajusta si tu modelo tiene otro mapeo
    predicted_letter = letters[predicted_letter_index]

    return predicted_letter

# Ruta para realizar predicciones
@app.route('/detect', methods=['POST'])
def detect():
    try:
        landmarks = request.json['landmarks']
        print("Landmarks recibidos:", landmarks)  # Debug

        # Realizar la predicción
        predicted_letter = detect_letter(landmarks)

        return jsonify({'predicted_letter': predicted_letter})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Error procesando los landmarks: {str(e)}'}), 500

# Ruta para verificar si el servidor está funcionando
@app.route('/')
def home():
    return jsonify({'message': 'API de Lenguaje de Señas está activa'})

# Correr la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

