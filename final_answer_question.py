import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

class TrainModel:
    def __init__(self, txt_path):
        self.path = txt_path
        self.number_of_texts = 0
        self.text = ""
        self.tokenizer = Tokenizer()
        self.model = tf.keras.models.Sequential()
        self.final_data = []
        self.x = None
        self.y = None
        self.to_be_used = None
        self.embedding_length = 0

    def read_data(self):
        if os.path.isfile(self.path):
            with open(self.path, "r") as file:
                self.text += file.read()
            return True
        else:
            return False

    def preprocess_data(self):
        flag = self.read_data()
        if flag:
            self.tokenizer.fit_on_texts([self.text])
            self.number_of_texts = len(self.tokenizer.word_index)

            # Sequential encoding
            for lines in self.text.split("\n"):
                sequential_data = self.tokenizer.texts_to_sequences([lines])[0]
                for i in range(1, len(sequential_data)):
                    n_gram = sequential_data[:i + 1]
                    self.final_data.append(n_gram)

            self.embedding_length = max([len(x) for x in self.final_data])
            self.to_be_used = sequence.pad_sequences(
                self.final_data,
                maxlen=self.embedding_length,
                padding="pre"
            )

            self.x = self.to_be_used[:, :-1]
            self.y = to_categorical(
                self.to_be_used[:, -1],
                num_classes=self.number_of_texts + 1
            )

            # Save the tokenizer
            with open("tokenized_data.txt", "w") as file:
                file.write(str(self.tokenizer.word_index))

            return True
        else:
            return False

    def train_model(self, **kwargs):
        lstm_layers = kwargs.get("lstm", 1)
        dense_layers = kwargs.get("dense", 1)
        output_dim = kwargs.get("output_dim", 128)

        self.model.add(
            tf.keras.layers.Embedding(
                input_dim=self.number_of_texts + 1,
                output_dim=output_dim,
                input_length=self.embedding_length - 1
            )
        )

        for i in range(lstm_layers):
            self.model.add(
                tf.keras.layers.LSTM(
                    150,
                    return_sequences=True if i < lstm_layers - 1 else False
                )
            )

        for i in range(dense_layers):
            self.model.add(
                tf.keras.layers.Dense(
                    100,
                    activation="relu"
                )
            )

        self.model.add(tf.keras.layers.Dense(self.number_of_texts + 1, activation="softmax"))
        self.compile_and_train_model()
        self.model.save("next_word.h5")

    def compile_and_train_model(self):
        self.model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        self.model.fit(
            self.x,
            self.y,
            epochs=50,
            verbose=1
        )
