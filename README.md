# Next-word-predictor
Predicting the next word using a sequence if characters

## Overview
This mission demonstrates the implementation of a subsequent word Prediction version the usage of TensorFlow and Streamlit. The version is designed to predict the subsequent phrase in a given textual content sequence, making use of LSTM (long short-term memory) layers for processing sequences and dense layers for classification. The version is deployed in an interactive web interface the use of Streamlit, permitting customers to input text and acquire actual-time tips for the following phrase.

## Features

* Sequential Data Preprocessing: Tokenizes and processes text data into sequences for model training.
* Customizable Model Architecture: Allows users to choose the number of LSTM and Dense layers, as well as the output dimensions.
* Real-time Word Prediction: Provides next-word predictions based on user input through a Streamlit interface.
* Model Persistence: Saves the trained model and tokenizer for future use.
## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/rohitkumar9989/Next-word-prediction.git
    cd next-word-prediction
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_application.py
    ```

## Usage

### Training the Model

1. Place your text data in a file named `Data.txt` in the project directory.
2. Run the Streamlit app, and if a pre-trained model doesn't exist, you will be prompted to train the model.
3. Customize the model by selecting the number of LSTM and Dense layers and the output dimensions.
4. The trained model and tokenizer will be saved as `next_word.h5` and `tokenized_data.txt`, respectively.

### Making Predictions

1. Enter your text in the input box provided in the Streamlit app.
2. The model will suggest the next word, and you can generate up to 10 words in sequence.

## Example

![Demo](demo.mp4)

## Project Structure

- **`streamlit_application.py`**: Contains the Streamlit application code for user interaction and model prediction.
- **`final_answer_question.py`**:Includes the `TrainModel` class responsible for reading data, preprocessing, training the model, and saving the tokenizer and model.
- **`Data.txt`**: Text file used for training the model.
- **`requirements.txt`**: Lists all necessary Python packages to run the project.
