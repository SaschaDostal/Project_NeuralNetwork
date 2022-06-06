import random

class NeuralNetwork:
    def __init__(self, hidden_layers, hidden_layer_size, input_layer_size, output_layer_size):
        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size

        self.nn_weights = self.initialize_weights(self.hidden_layers, self.hidden_layer_size, self.input_layer_size, self.output_layer_size)

    # creates a list with weights for every transition between two knots
    # Format: [number of transition][number of left knot][number of right knot]
    def initialize_weights(self, hidden_layer, hidden_layer_size, input_layer_size, output_layer_size):
        nn = []
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

            nn.append(transition)

        return nn

nn = NeuralNetwork(2, 3, 1, 1)

