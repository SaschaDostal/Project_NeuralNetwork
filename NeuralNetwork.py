import random

def main():
    max_epochs = 100
    hidden_layer = 4
    hidden_layer_size = 3
    input_layer_size = 2
    output_layer_size = 1

    nn = create_nn(hidden_layer, hidden_layer_size, input_layer_size, output_layer_size)
    #nn = train_nn(nn, max_epochs, training_data)

def create_nn(hidden_layer, hidden_layer_size, input_layer_size, output_layer_size):
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

def train_nn(nn, epochs, training_data):
    # TODO: shuffle training_data

    for epoch in range(epochs):
        for sample in range(training_data):
            initialize_input_layer(sample)
            forward_pass(nn)
            backward_pass()
    return nn

def initialize_input_layer():
    pass

def forward_pass(nn):
    for transition in nn:
        

def backward_pass():
    pass

if __name__ == "__main__":
    main()