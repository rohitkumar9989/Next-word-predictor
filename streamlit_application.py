import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import os
import ast
import numpy as np
from final_answer_question import TrainModel as tm

st.title("Next Word Predictor...")

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

if os.path.isfile("next_word.h5"):
    with open("./tokenized_data.txt", "r") as file:
        tokenizer_data = file.read()
        tokenizer = Tokenizer()
        tokenizer.word_index = ast.literal_eval(tokenizer_data)
    
    model = load_model("next_word.h5")
    st.write("Enter the text below and you'll be provided with suggestions")
    input_text = st.text_input("Enter the text here")
    st.write(f"Input text: {input_text}")

    if input_text:
        max_sequence_len = model.input_shape[1] + 1
        for i in range (10):
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            input_text+= f" {next_word}"
        st.write(f"Next Word Prediction: {input_text}")
else:
    main_tm = tm("./Data.txt")
    status = main_tm.preprocess_data()
    
    if status:
        st.write("Select the number of LSTM layers needed:")
        LSTM = st.select_slider("Enter the number of LSTM layers", options=[1, 2, 3, 4, 5])
        st.write("Select the number of Dense layers needed:")
        Dense = st.select_slider("Enter the number of Dense layers", options=[1, 2, 3, 4, 5])
        st.write("Select the output dimensions needed:")
        output_dim = st.select_slider("Enter the number of Output Dimensions", options=[100, 200, 300, 400, 500])

        main_tm.train_model(lstm=LSTM, dense=Dense, output_dim=output_dim)
        
        # Save the tokenizer for later use
        with open("tokenized_data.txt", "w") as file:
            file.write(str(main_tm.tokenizer.word_index))
        
        model = load_model("next_word.h5")
        st.write("Enter the text below and you'll be provided with suggestions")
        input_text = st.text_input("Enter the text here")
        st.write(f"Input text: {input_text}")

        if input_text:
            max_sequence_len = model.input_shape[1] + 1
            for i in range (10):
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
                input_text+= f" {next_word}"
            st.write(f"Next Word Prediction: {input_text}")
    else:
        st.write("An error occurred")
