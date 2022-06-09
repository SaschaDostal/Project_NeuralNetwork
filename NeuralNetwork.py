import random
import math
import json
import statistics


class NeuralNetwork:
    def __init__(self, path_to_nn=None, hidden_layers=2, hidden_layer_size=3, input_layer_size=3, output_layer_size=1):
        print(path_to_nn)
        if path_to_nn is None:
            self.hidden_layers = hidden_layers
            self.hidden_layer_size = hidden_layer_size
            self.input_layer_size = input_layer_size
            self.output_layer_size = output_layer_size
            self.initialize_weights(
                self.hidden_layers, self.hidden_layer_size, self.input_layer_size, self.output_layer_size)
            print("Neural network initialized:")
        else:
            with open(path_to_nn) as data_file:
                self.weights = json.load(data_file)
            self.hidden_layers = len(self.weights) - 1
            self.hidden_layer_size = len(self.weights[1])
            self.input_layer_size = len(self.weights[0])
            self.output_layer_size = len(self.weights[-1][0])
            print("Neural network loaded:")

        print("Hidden layers: " + str(self.hidden_layers))
        print("Hidden layer size: " + str(self.hidden_layer_size))
        print("Input layer size: " + str(self.input_layer_size))
        print("Output layer size: " + str(self.output_layer_size))

    # creates a list with weights for every transition between two knots
    # Format: [number of transition][number of left knot][number of right knot]

    def initialize_weights(self, hidden_layer, hidden_layer_size, input_layer_size, output_layer_size):
        weights = []
        for transitions in range(hidden_layer + 1):
            transition = []
            #print("TRANSITION - " + str(transitions))

            if transitions == 0:
                num_left = input_layer_size
            else:
                num_left = hidden_layer_size

            for left_knot in range(num_left):
                left = []
                #print("    LEFTKNOT - " + str(left_knot))

                if transitions == hidden_layer:
                    num_right = output_layer_size
                else:
                    num_right = hidden_layer_size - 1  # "-1" wegen Bias-Knoten

                for right_knot in range(num_right):
                    random_num = random.random()
                    left.append(random_num)
                    #print("        RIGHTKNOT - " + str(right_knot) + " - " + str(random_num))

                transition.append(left)
            weights.append(transition)

        self.weights = weights

    def train(self, training_data, epochs, learning_rate):
        self.predicted_list = []    # for graph
        self.loss_list = []         # for graph
        self.epoch_list = []        # for graph
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            total = 0
            true = 0
            self.loss_per_sample = []
            for sample in training_data:
                output = self.forward_pass(sample)
                total += 1
                if round(output[0]) == int(sample[-1]):
                    true += 1
                self.backward_pass(output, sample, learning_rate)
            
            self.predicted_list.append(float(true)/total)                   # for graph
            self.loss_list.append(statistics.mean(self.loss_per_sample))    # for graph
            self.epoch_list.append(epoch)                                   # for graph
            
            if epoch == 0 or epoch % 100 == 0:
                print("Correct Predicted: {:7.4f}".format(
                    float(true)/total) + "%")

    def forward_pass(self, sample):
        output = sample[:-1]
        self.currentInput = []
        self.currentOutput = []
        #print("sample" + str(sample[:-1]))
        # for each transition
        for l in range(len(self.weights)):
            output.append(1.0)  # adding bias to output
            self.currentOutput.append(output)
            out_ = []
            in_ = []

            # for each right knot in transition l
            for k in range(len(self.weights[l][0])):
                input = 0
                # calculate sum of outputs
                for j in range(len(output)):
                    input += float(output[j]) * self.weights[l][j][k]

                in_.append(input)
                out_.append(self.sig(input))
                #print("input " + str(input) + " output " + str(self.sig(input)))
            #print("OUT" + str(out_))

            self.currentInput.append(in_)

            output = out_
        return output

    def backward_pass(self, output, sample, learning_rate):
        deltas = []

        # for each knot k in the output layer do
        # delta[k] = d-sig(in[k])*(y[k]-out[k])
        output_deltas = []
        for k in range(self.output_layer_size):
            output_deltas.append((float(
                sample[k - self.output_layer_size]) - output[k]) * self.d_sig(self.currentInput[-1][k]))
            self.loss_per_sample.append(abs(float(sample[k - self.output_layer_size])-output[k]))
            #print("Class:" + sample[k - self.output_layer_size] + ", Prediction: " + str(output[k]) + ", Diff: " + str(float(sample[k - self.output_layer_size])-output[k]))
        deltas.append(output_deltas)

        # for hidden layer l = L - 1 to 2 do
        #   for each (not bias) knot j in hidden layer l do
        #       delta[j] = g'(in[j]) * sum(w[j][k] * delta[k])
        for t in range(self.hidden_layers):
            hidden_deltas = []
            # -1 because of bias knot
            for j in range(len(self.weights[-(1 + t)])-1):
                sum = 0.0
                for k in range(len(self.weights[-(1 + t)][j])):
                    sum += self.weights[-(1 + t)][j][k] * deltas[0][k]
                hidden_deltas.append(
                    sum * self.d_sig(self.currentInput[-(2 + t)][j]))
            deltas.insert(0, hidden_deltas)

        # for each w[j][k] do
        #   w[j][k] = w[j][k] + alpha * out[j] * delta[k]
        for t in range(len(self.weights)):
            for j in range(len(self.weights[t])-1):  # -1 because of bias knot
                for k in range(len(self.weights[t][j])):
                    self.weights[t][j][k] = self.weights[t][j][k] + \
                        learning_rate * \
                        float(self.currentOutput[t][j]) * deltas[t][k]

    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def d_sig(self, x):
        return self.sig(x)*(1.0 - self.sig(x))

    def delta(self, x):
        return self.d_sig(x)

    def predict(self, sample):
        return round(self.forward_pass(sample)[0])

    def save_nn(self, path):
        with open(path + '.json', 'w') as f:
            json.dump(self.weights, f)
