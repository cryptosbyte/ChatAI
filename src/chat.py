from model import NeuralNetwork
from utils import Tokenize, Bag_O_Words_Func
import random
import json
import torch


class Chat:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        with open("json/intents.json", "r") as file:
            self.intents = json.load(file)

        self.data = torch.load("bin/data.pth")
        input_size = self.data["input_size"]
        output_size = self.data["output_size"]
        self.all_words = self.data["all_words"]
        self.tags = self.data["tags"]
        model_state = self.data["model_state"]

        self.model = NeuralNetwork(input_size, 8, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def start(self):

        print("Let's chat!")
        while True:
            sentence = input("You: ")
            if sentence == "quit":
                break

            X = Bag_O_Words_Func(Tokenize(sentence), self.all_words)
            X = torch.from_numpy(X.reshape(1, X.shape[0])).to(self.device)

            output = self.model(X)
            _, predicted = torch.max(output, dim=1)

            tag = self.tags[predicted.item()]

            probabilities = torch.softmax(output, dim=1)
            probability = probabilities[0][predicted.item()]

            if probability.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        print(
                            f"Kyle [Res. Prob {probability.item()}]: {random.choice(intent['responses'])}")
            else:
                print(
                    f"Kyle [Res. Prob {probability.item()}]: I do not understand...")


Chat().start()
