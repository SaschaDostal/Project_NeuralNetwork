import NeuralNetwork
import Plotter

import csv

def main():
    max_epochs = 1
    learning_rate = 0.05
    hidden_layers = 2
    hidden_layer_size = 3   #includes bias
    input_layer_size = 3    #includes bias
    output_layer_size = 1

    training_data = readCSV('data/testdata_train.csv')

    nn = NeuralNetwork.NeuralNetwork(hidden_layers, hidden_layer_size, input_layer_size, output_layer_size)
    nn.train(training_data, max_epochs, learning_rate)


def readCSV(path):
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data.append(list(reader))
    return data[0][1:]

if __name__ == "__main__":
    main()