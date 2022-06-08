import matplotlib.pyplot as plt

def plot():
    
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