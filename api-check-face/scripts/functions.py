import tensorflow as tf

def load_model_and_predict_image(image_path, model_path):
    # Carregar o modelo
    model = tf.keras.models.load_model(model_path)

    # Carregar a imagem de teste
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Normalização dos valores dos pixels (se necessário)
    image_array = tf.expand_dims(image_array, 0)  # Adicionando uma dimensão extra para representar o batch

    # Fazendo a previsão com o modelo carregado
    predictions = model.predict(image_array)

    # Obtendo o resultado da previsão (0 ou 1 no caso de classificação binária)
    predicted_class = int(predictions[0][0] + 0.5)

    # Retornando o resultado da previsão
    if predicted_class == 0:
        return "A imagem é classificada como 'real'."
    else:
        return "A imagem é classificada como 'fake'."

