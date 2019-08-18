from numpy import exp, array, random, dot, transpose, zeros, size, arange, full_like, append
from sklearn import datasets, preprocessing



iris = datasets.load_iris()

def normalize(x):

    y = zeros((size(x, 0), (size(x, 1))))

    for k in range(size(x, 0)):
        row_max = max(x[k, :])
        row_min = min(x[k, :])

        for n in range(size(x, 1)):
            if(row_max-row_min) == 0:
                y[k, n] = 0
            else:
                y[k, n] = (((x[k, n]) - row_min) / (row_max - row_min))

    return y

#class Layer():
#    def __init__(self, neuron_number, input_number):
#        self.weights = random.random((input_number, neuron_number)) - 1

class Neuron():
    def __init__(self, input_number):
        self.weights = random.random(input_number)

        self.desired_value = None
        self.input_connections = array(None)
        self.weighted_sum = None
        self.neuron_output = None
        self.error = None
        self.delta = None

    def set_desired_value(self, desired_value):
        self.desired_value = desired_value
        return self.desired_value

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1 - x)

    def forward(self, input):

        scaler = preprocessing.StandardScaler()

        input = input.reshape(-1, 1)

        self.input_connections = scaler.fit_transform(input)
        self.input_connections = self.input_connections.reshape(-1)
        self.weighted_sum = self.count_sum(self.input_connections)
        self.neuron_output = self.count_output(self.weighted_sum)
        return self.neuron_output

    def count_sum(self, input_connection):
        weighted_sum = dot(input_connection, self.weights)
        return weighted_sum

    def count_output(self, weighted_sum):
        output = self.sigmoid(weighted_sum)
        return output

    def compute_final_error(self):
        self.error = self.neuron_output - self.desired_value
        return self.error

    def compute_error(self, next_layer_error, next_layer_weights, next_layer_output):
        self.error = next_layer_error * next_layer_weights * self.sigmoid_deriv(next_layer_output)
        return self.error

    def calculate_delta(self, prev_layer_output):
        self.delta = self.error * self.sigmoid_deriv(self.neuron_output) * prev_layer_output
        return self.delta

    def update_weights(self, prev_layer_output):
        print("LAST OUT: ", prev_layer_output)
        print(self.weights)
        for propag in range(len(prev_layer_output)):
            update_weight = self.calculate_delta(prev_layer_output[propag])
            self.weights[propag] += update_weight

        print(self.weights)
        return self.weights

class Layer():
    def __init__(self, layer_size, input_size):
        self.outputs_backprop = arange(input_size)
        self.layer_size = layer_size
        self.Layer = arange(layer_size, dtype=Neuron)
        self.layer_output = arange(layer_size, dtype=float)

        for neuron in range(layer_size):
            current_neuron = Neuron(input_size)
            self.Layer[neuron] = current_neuron

    def set_layer_output(self, inp_data):
        for neuron in range(len(self.Layer)):
            self.outputs_backprop = input_data
            self.layer_output[neuron] = self.Layer[neuron].forward(inp_data)
        return self.layer_output

    def set_layer_desired_value(self, desired_values):
        for neuron in range(len(self.Layer)):
            self.Layer[neuron].set_desired_value(desired_values[neuron])

    def count_layer_error(self, next_layer_output, final="not last"):
        for neuron in range(len(self.Layer)):
            if final == "last":
                self.Layer[neuron].compute_final_error()
            else:
                self.Layer[neuron].compute_error()

    def update_layer_weights(self, prev_layer_outputs):
        for neuron in range(len(self.Layer)):
            self.Layer[neuron].update_weights(prev_layer_outputs)

class Network():
    def __init__(self, inp_data, *args):

        input_layer = Layer(args[0], inp_data)
        self.input_layer = input_layer

        output_layer = Layer(args[len(args) - 1], args[(len(args) - 2)])
        self.output_layer = output_layer

        if len(args) <= 2:
            print("Too little layers.")
        else:
            self.hidden_layers = arange((len(args)-2), dtype=Layer)
            for layer in range(1, len(args)-1):
                hidden_layer = Layer(args[layer], args[layer-1])
                self.hidden_layers[layer-1] = hidden_layer

    # def sigmoid_deriv(self, x):
    #     return x * (1 - x)
    #
    # def think(self, input):
    #     layer1_out = self.sigmoid(dot(input, self.layer1.weights))
    #     layer2_out = self.sigmoid(dot(layer1_out, self.layer2.weights))
    #     layer3_out = self.sigmoid(dot(layer2_out, self.layer3.weights))
    #     return layer1_out, layer2_out, layer3_out
    #
    # def train(self, input, output, train_number):
    #     for iteration in range(train_number):
    #         layer1_out, layer2_out, layer3_out = self.think(input)
    #
    #         layer3_error = layer3_out - output
    #         layer3_delta = layer3_error * self.sigmoid_deriv(layer3_out)
    #
    #         layer2_error = layer3_delta.dot(self.layer3.weights.T)
    #         layer2_delta = layer2_error * self.sigmoid_deriv(layer2_out)
    #
    #         layer1_error = layer2_delta.dot(self.layer2.weights.T)
    #         layer1_delta = layer1_error * self.sigmoid_deriv(layer1_out)
    #
    #         layer1_update = input.T.dot(layer1_delta)
    #         layer2_update = layer1_out.T.dot(layer2_delta)
    #         layer3_update = layer2_out.T.dot(layer3_delta)
    #
    #         self.layer1.weights += layer1_update
    #         self.layer2.weights += layer2_update
    #         self.layer3.weights += layer3_update


    def print_weights(self):
        print ("    Layer 1 Neuron 1: ")
        print(N.Network[0].Layer[0].weights)
        print ("    Layer 2 Neuron 1: ")
        print(N.Network[1].Layer[0].weights)
        print ("    Layer 3 Neuron 1: ")
        print(N.Network[2].Layer[0].weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    N = Network(4, 4, 3, 2)
    input_data = array([2, 2, 0, 0], dtype=float)
    out_data = array([1, 1])

    current_data = N.input_layer.set_layer_output(input_data)
    print(current_data)

    for layer in N.hidden_layers:
        current_data = layer.set_layer_output(current_data)
        print(current_data)

    current_data = N.output_layer.set_layer_output(current_data)
    print(current_data)

    N.output_layer.set_layer_desired_value(out_data)
    N.output_layer.count_layer_error(out_data, "last")
    N.output_layer.update_layer_weights(out_data)



