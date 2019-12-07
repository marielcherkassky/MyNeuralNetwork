import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection as ms


class MyNN:
    def __init__(self, learning_rate, layer_sizes):
        '''
        learning_rate - the learning to use in backward
        layer_sizes - a list of numbers, each number repreents the nuber of neurons
                      to have in every layer. Therfore, the length of the list
                      represents the number layers this network has.
        '''
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.model_params = {}
        self.memory = {}
        self.grads = {}

        # Initializing weights
        for layer_index in range(len(layer_sizes) - 1):
            W_input = layer_sizes[layer_index + 1]
            W_output = layer_sizes[layer_index]
            self.model_params['W_' + str(layer_index + 1)] = np.random.randn(W_input, W_output) * 0.1
            self.model_params['b_' + str(layer_index + 1)] = np.random.randn(W_input) * 0.1

    def forward_single_instance(self, x):
        a_l_input = x
        self.memory['a_0'] = x
        for layer_index in range(len(self.layer_sizes) - 1):
            W_l = self.model_params['W_' + str(layer_index + 1)]
            b_l = self.model_params['b_' + str(layer_index + 1)]
            z_l = np.dot(W_l,a_l_input) + b_l  # Matrix of weights on layer i * the input vector on layer i + bias * 1 on that layer
            a_l_output = 1 / (1 + np.exp(-z_l))  # sigmoid func , a_l is a vector of all the a in the layer l
            self.memory['a_' + str(layer_index + 1)] = a_l_output  # save results in memory
            a_l_input = a_l_output  # input of next layer is output of current layer
        return a_l_input

    def log_loss(self,y_hat, y):  # y is also an array type !
        '''
        Logistic loss, assuming a single value in y_hat and y.
        '''
        cost = -y[0] * np.log(y_hat[0]) - (1 - y[0]) * np.log(1 - y_hat[0])  # The loss function calculation
        return cost

    def backward_single_instance(self, y):
        a_output = self.memory['a_' + str(len(self.layer_sizes) - 1)]
        dz = a_output - y  # y label - y predicted (vector)

        for layer_index in range(len(self.layer_sizes) - 1, 0,-1):  # going backwards from last layer untill the first one
            a_l_1 = self.memory['a_' + str(layer_index - 1)]  # Matrix of a (neuron input) in the current layer
            dW = np.dot(dz.reshape(-1, 1), a_l_1.reshape(1,-1))  # Make the a_l_1 horizontal vector and dz is 1d scalar -> dW horizontal vector
            self.grads['dW_' + str(layer_index)] = dW
            W_l = self.model_params['W_' + str(layer_index)]
            self.grads['db_' + str(layer_index)] = dz.reshape(-1)
            dz = (a_l_1 * (1 - a_l_1)).reshape(-1, 1) * np.dot(W_l.T, dz.reshape(-1, 1))

    # Update the weights of  the model params
    def update(self):
        for layer_index in range(len(self.layer_sizes) - 1):
            #print("updating weights on layer : {}".format(layer_index+1))
            self.model_params['W_' + str(layer_index + 1)] -= (self.learning_rate*self.grads['dW_' + str(layer_index + 1)])
            self.model_params['b_' + str(layer_index + 1)] -= (self.learning_rate*self.grads['db_' + str(layer_index + 1)])

    # forward algorithem, input batch X, The shape should be : (network_input_size,number_of_instances)
    def forward_batch(self, X):
        self.memory["A_{}".format(0)] = X  # X we get is a matrix of columnar input instances
        A_l_input = X
        for layer_index in range(len(self.layer_sizes) - 1):
            W_l = self.model_params['W_' + str(layer_index + 1)]
            b_l = self.model_params['b_' + str(layer_index + 1)]
            Z_l = np.dot(W_l, A_l_input) + b_l.reshape(-1,1)  # One row contains : all the Z values that will be inserted into the specific neuron for all samples
            A_l_output = 1 / (1 + np.exp(-Z_l)) # Activation func on the matrix (apply the func on each cell)
            self.memory['A_{}'.format(layer_index + 1)] = A_l_output
            A_l_input = A_l_output  # # input of next layer is output of current layer

        return A_l_output

    # backwards algorithem for batchmod, y.shape = (1, number_of_instance)
    def backward_batch(self, y):
        A_l_output = self.memory['A_' + str(len(self.layer_sizes) - 1)]
        dZ = A_l_output - y  # result is vector

        for layer_index in range(len(self.layer_sizes) - 1, 0,-1):  # going backwards from last layer untill the first one
            #print("backwards working, layer : {}".format(layer_index))
            A_l_input = self.memory['A_' + str(layer_index - 1)]  # Matrix of a (neuron input) in the current layer
            dW = np.dot(dZ * (1 / self.memory["A_0"].shape[1]), A_l_input.T)  # 1/m , A_l_input is a matrix of all values of neurons for all samples in batch
            self.grads['dW_' + str(layer_index)] = dW
            W_l = self.model_params['W_' + str(layer_index)]
            self.grads['db_' + str(layer_index)] = dZ.T.mean(0) # Each row in dz contains info for many samples, therefore we calculate average for every row and replace all the row with the average
            dZ = (A_l_input * (1 - A_l_input)) * np.dot(W_l.T, dZ)

    # loss function for batch mode
    def log_loss_batch(self, Y_hat, Y):
        m = Y.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return cost

    # This function takes X (matrix of columnar vectors !) and y, shuffles the data and divide it to batches.
    @staticmethod
    def shuffle_and_batch(X, y, batch_size):
        # Shuffle
        data = np.vstack([X, y])  # add y as last row to data
        data = data.T
        np.random.shuffle(data)

        # Batches
        batches_with_y = np.array_split(data, batch_size) # split the data (with y attached to it ) into batches
        # batch_with_y[:, :-1] - get all rows and all columns except the last column(y value)
        # batch_with_y[:, -1] - get all rows and only the last column (y value)
        return np.array(
            [
                (np.array(batch_with_y[:, :-1]).T,
                 np.array(batch_with_y[:, -1]).reshape(1,-1)
                 ) for batch_with_y in batches_with_y
            ]
        )


    def train(self,X, y, epochs, batch_size):
        '''
        Train procedure, please note the TODOs inside
        '''
        loss_per_epoch=[]
        for e in range(1, epochs + 1):
            print("epoch {}".format(e))
            epoch_loss = 0
            batches = self.shuffle_and_batch(X, y, batch_size)
            for X_b, y_b in batches:
                y_hat = self.forward_batch(X_b)
                epoch_loss += self.log_loss_batch(y_hat, y_b)
                self.backward_batch(y_b)
                self.update()
            loss_per_epoch.append(epoch_loss / len(batches))
            print(f'Epoch {e}, loss={epoch_loss / len(batches)}')
        return loss_per_epoch



def runSingleExample():
    print("test")
    nn = MyNN(0.01, [3, 2, 1])
    x = np.random.randn(3)
    y = np.random.randn(1)
    log_losts = []
    for epoch in range(0, 200):
        y_hat = nn.forward_single_instance(x)
        loss = nn.log_loss(y_hat, y).tolist()
        log_losts.append(loss)
        nn.backward_single_instance(y)
        nn.update()
    plt.plot(log_losts)
    plt.show()


def runBatchExample():
    nn = MyNN(0.001, [6, 4, 3, 1])
    X = np.random.randn(6, 100)
    y = np.random.randn(1, 100)
    batch_size = 8
    epochs = 200
    nn.train(X, y, epochs, batch_size)

def WorkOnRealDataExample():
    df = pd.read_csv("day.csv")
    # Data preparation
    data = df[["temp", "atemp", "hum", "windspeed", "weekday", "cnt"]]
    data["success"] = data["cnt"] > data["cnt"].describe()["mean"]
    data = data.drop(columns="cnt")

    # Data preparation
    data = df[["temp", "atemp", "hum", "windspeed", "weekday", "cnt"]]
    data["success"] = data["cnt"] > data["cnt"].describe()["mean"]
    data = data.drop(columns="cnt")

    # NN architecture and initialization
    batch_size = 8
    network_architecture = [5, 40, 30, 10, 7, 5, 3, 1]
    learning_rate = 0.01
    epochs = 100
    batch_size = 8
    mynn = MyNN(learning_rate, network_architecture)
    X_train, X_test, y_train, y_test = ms.train_test_split(data[["temp", "atemp", "hum", "windspeed", "weekday"]],
                                                           data["success"])

    X_train = np.array(X_train.T)
    X_test = np.array(X_test.T)
    y_train = np.array(y_train).reshape(1, -1)
    y_test = np.array(y_test).reshape(1, -1)
    print(X_train.shape)
    print(y_train.shape)
    loss_per_epoch = np.array(mynn.train(X_train, y_train, epochs, batch_size))
    loss_per_epoch = loss_per_epoch.reshape(1, -1)
    plt.plot(loss_per_epoch[0])
    plt.show()
    print(loss_per_epoch)


def main():
    #runSingleExample()
    #runBatchExample()
    #WorkOnRealDataExample()

    pass

if __name__ == "__main__":
    main()
