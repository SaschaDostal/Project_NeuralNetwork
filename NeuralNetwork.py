import random
import csv
import math

class NeuralNetwork:
    def __init__(self, hidden_layers, hidden_layer_size, input_layer_size, output_layer_size):
        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.initialize_weights(self.hidden_layers, self.hidden_layer_size, self.input_layer_size, self.output_layer_size)

    # creates a list with weights for every transition between two knots
    # Format: [number of transition][number of left knot][number of right knot]
    def initialize_weights(self, hidden_layer, hidden_layer_size, input_layer_size, output_layer_size):
        weights = []
        for transitions in range (hidden_layer + 1):
            transition = []
            print("TRANSITION - " + str(transitions))

            if transitions == 0:
                num_left = input_layer_size
            else:
                num_left = hidden_layer_size

            for left_knot in range(num_left):
                left = []
                print("    LEFTKNOT - " + str(left_knot))

                if transitions == hidden_layer:
                    num_right = output_layer_size
                else:
                    num_right = hidden_layer_size - 1 # "-1" wegen Bias-Knoten

                for right_knot in range(num_right):
                    random_num = random.random()
                    left.append(random_num)
                    print("        RIGHTKNOT - " + str(right_knot) + " - " + str(random_num))

                transition.append(left)
            weights.append(transition)
        
        self.weights=weights

    def train(self, training_data, epochs, learning_rate):
        random.shuffle(training_data)

        for epoch in range(epochs):
            for sample in training_data:
                self.forward_pass(sample)
                self.backward_pass()

    def forward_pass(self, sample):
        output = sample[:-1]
        ##print("sample" + str(sample[:-1]))
        # for each transition
        for l in range(len(self.weights)):
            output.append(1.0) # adding bias to output
            out = []
            # for each right knot in transition l
            for k in range(len(self.weights[l][0])):
                input = self.sum(output, self.weights[l], k)
                out.append(self.sig(input))
                ##print("input " + str(input) + " output " + str(self.sig(input)))
            ##print("OUT" + str(out))
            output=out

    def sum(self, output, w, k):
        sum = 0
        for j in range(len(output)):
            sum += float(output[j]) * float(w[j][k])
            ##print("sum += " + str (output[j]) + " * " + str(w[j][k]))
        return sum

    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def backward_pass(self):
        pass

