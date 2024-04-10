from time import sleep

import redis
from tqdm import keras


class TextNN:

    def __init__(self):
        self.model = 1

    def predict(self, text):
        return {
            'text': text,
        }


def main():
    text_neural_network = TextNN()
    while True:
        sleep(0.01)


if __name__ == "__main__":
    main()