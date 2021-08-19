from utils import Tokenize, Stem, Bag_O_Words_Func
from model import NeuralNetwork
from torch.utils.data import Dataset, DataLoader
import json
import numpy
import torch
import torch.nn as nn


class Trainer:
    def __init__(self):

        with open("json/intents.json", "r") as file:
            self.intents = json.load(file)

        self.all_words = []
        self.tags = []
        self.xy = []
        self.ignore_punctuations = [".", ",", "?", "!"]

        self.IntentsManager()

        self.X_train = []
        self.Y_train = []

        self.Bag_O_Words()

        self.X_train = numpy.array(self.X_train)
        self.Y_train = numpy.array(self.Y_train)

    def IntentsManager(self) -> None:

        for intent in self.intents["intents"]:
            tag = intent["tag"]

            self.tags.append(tag)

            for pattern in intent["patterns"]:
                word = Tokenize(pattern)

                self.all_words.extend(word)
                self.xy.append((word, tag))

        self.all_words = [
            Stem(word) for word in self.all_words
            if word not in self.ignore_punctuations
        ]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

    def Bag_O_Words(self) -> None:

        for (pattern_sentence, tag) in self.xy:
            bag = Bag_O_Words_Func(pattern_sentence, self.all_words)
            label = self.tags.index(tag)

            self.X_train.append(bag)
            self.Y_train.append(label)

    def Return_Data(self) -> object:
        return {
            "X": self.X_train,
            "Y": self.Y_train,
            "tags": self.tags,
            "all_words": self.all_words,
        }


TrainerData = Trainer().Return_Data()


class ChatDataset(Dataset):

    def __init__(self):

        self.n_samples = len(TrainerData["X"])
        self.x_data = TrainerData["X"]
        self.y_data = TrainerData["Y"]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.n_samples


# Hyperparameters (Controls ML's process)
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
input_size = len(TrainerData["X"][0])
output_size = len(TrainerData["tags"])

dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_size, 8, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)  # Forward
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # Backward & Optimizer
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")

torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    # "hidden_size": 8,
    "all_words": TrainerData["all_words"],
    "tags": TrainerData["tags"]
}, "bin/data.pth")

print("data saved to data.pth")
