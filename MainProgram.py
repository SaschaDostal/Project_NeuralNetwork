import NeuralNetwork
import Plotter

import csv


def main():
    training_data = readCSV('data/testdata_train_3.csv')
    
    max_epochs = 1000
    learning_rate = 0.0001
    hidden_layers = 4
    hidden_layer_size = 5  # includes bias
    input_layer_size = len(training_data[0])  # includes bias
    output_layer_size = 1

    answer = input("Load existing neural network? (y/n)\n")
    if(answer == 'y'):
        answer = input("Path/Filename: ")
        nn = NeuralNetwork.NeuralNetwork(path_to_nn=answer)
    else:
        nn = NeuralNetwork.NeuralNetwork(hidden_layers=hidden_layers, hidden_layer_size=hidden_layer_size,
                                         input_layer_size=input_layer_size, output_layer_size=output_layer_size)

    continue_training = True
    while continue_training:
        nn.train(training_data, max_epochs, learning_rate)

        Plotter.plot_learning_curve(nn.epoch_list, nn.loss_list, nn.predicted_list)
        if input_layer_size == 3:
            Plotter.plot_map(nn, training_data)

        answer = input("Continue training? (y/n)\n")
        if(answer == 'y'):
            answer = input(
                "Choose new learning rate (currently: " + str(learning_rate) + "): ")
            learning_rate = float(answer)
            answer = input(
                "Choose epochs (last time: " + str(max_epochs) + "): ")
            max_epochs = int(answer)
        else:
            answer = input("Save neural network? (y/n)\n")
            if(answer == 'y'):
                answer = input("Path/Filename: ")
                nn.save_nn(answer)
            continue_training = False


def readCSV(path):
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data.append(list(reader))
    return data[0][1:]


if __name__ == "__main__":
    main()
