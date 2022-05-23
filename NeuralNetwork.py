import random
import csv

def main():
    max_epochs = 100
    hidden_layer = 4
    hidden_layer_size = 3
    input_layer_size = 2
    output_layer_size = 1

    #nn = create_nn(hidden_layer, hidden_layer_size, input_layer_size, output_layer_size)

    training_data = readCSV('diabetes_train.csv')
    #nn = train_nn(nn, max_epochs, training_data)
    
    test_data = readCSV('diabetes_test.csv')
    

    #plot()

# creates a list with a value for every transition between two knots
# Format: [number of transition][number of left knot][number of right knot]
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

def train_nn(nn, epochs, training_data, learning_rate):
    
    random.shuffle(training_data)

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
        pass
        

def backward_pass():
    pass

def plot():
    import matplotlib.pyplot as plt
    
    # x-axis values
    a = [1,2,3,4,5]
    c = [6,7,8,9,10]
    # y-axis values
    b = [2,4,5,7,6]
    d = [8,9,11,12,12]

    precision = 200

    max_x = 10
    max_y = 12
    
    # color area for class 1
    x_1 = []
    y_1 = []
    for x in range(precision):
        for y in range(precision):
            if test(x/float(precision) * max_x, y/float(precision) * max_y) == 0:
                x_1.append(x/float(precision) * max_x)
                y_1.append(y/float(precision) * max_y)
    plt.scatter(x_1, y_1, color= "#630700", marker='s', s=50)

    # color area for class 2
    x_2 = []
    y_2 = []
    for x in range(precision):
        for y in range(precision):
            if test(x/float(precision) * max_x, y/float(precision) * max_y) == 1:
                x_2.append(x/float(precision) * max_x)
                y_2.append(y/float(precision) * max_y)
    plt.scatter(x_2, y_2, color= "#000763", marker='s', s=50)

    # draw points for test samples
    plt.scatter(a, b, label= "Class 1", color= "red", s=10)
    plt.scatter(c, d, label= "Class 2", color= "blue", s=10)
    
    # x-axis label
    plt.xlabel('x - axis')
    # y-axis label
    plt.ylabel('y - axis')

    plt.title('Visualization of test samples')
    plt.legend()
    plt.show()

def test(x, y):
    if x < 7 and y < 7:
        return 0
    else:
        return 1

def readCSV(path):
    data = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data.append(list(reader))
    return data[0][1:]

if __name__ == "__main__":
    main()