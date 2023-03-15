import math

import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        prod = nn.as_scalar(self.run(x))
        if prod >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        upd = True
        while upd:
            upd = False
            for x, y in dataset.iterate_once(batch_size):
                pred = self.get_prediction(x)
                if pred == nn.as_scalar(y):
                    continue
                else:
                    self.w.update(nn.Constant(nn.as_scalar(y) * x.data), 1)
                    upd = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        # Batch size: between 1 and the size of the dataset.
        self.bs = 50
        # Learning rate: between 0.001 and 1.0.
        # Number of hidden layers: between 1 and 3.
        "*** YOUR CODE HERE ***"
        self.lr = .03
        self.w1 = nn.Parameter(1, 50)  # Hidden layer sizes: between 10 and 400.
        self.b1 = nn.Parameter(1, 50)
        self.w2 = nn.Parameter(50, 25)
        self.b2 = nn.Parameter(1, 25)
        self.w3 = nn.Parameter(25, 1)
        self.b3 = nn.Parameter(1, 1)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        l1_f = nn.Linear(x, self.w1)  # nn.Linear(features, weights)
        l1 = nn.ReLU(nn.AddBias(l1_f, self.b1))  # nn.AddBias(features, bias)

        l2_f = nn.Linear(l1, self.w2)
        l2 = nn.ReLU(nn.AddBias(l2_f, self.b2))

        l3_f = nn.Linear(l2, self.w3)
        l3 = nn.AddBias(l3_f, self.b3)
        return l3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_o = self.run(x)
        res = nn.SquareLoss(y_o, y)
        return res

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        training_loss = float('inf')
        while training_loss >= .019:  # full points if it gets a loss of 0.02 or better
            for x, y in dataset.iterate_once(self.bs):
                training_loss = self.get_loss(x, y)
                grads = nn.gradients(training_loss, self.parameters)
                training_loss = nn.as_scalar(training_loss)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(grads[i], -self.lr)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.bs = 50
        self.lr = .09
        self.w1 = nn.Parameter(784, 350)
        self.b1 = nn.Parameter(1, 350)
        self.w2 = nn.Parameter(350, 150)
        self.b2 = nn.Parameter(1, 150)
        self.w3 = nn.Parameter(150, 50)
        self.b3 = nn.Parameter(1, 50)
        self.w4 = nn.Parameter(50, 10)
        self.b4 = nn.Parameter(1, 10)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l1_f = nn.Linear(x, self.w1)  # nn.Linear(features, weights)
        l1 = nn.ReLU(nn.AddBias(l1_f, self.b1))  # nn.AddBias(features, bias)

        l2_f = nn.Linear(l1, self.w2)
        l2 = nn.ReLU(nn.AddBias(l2_f, self.b2))

        l3_f = nn.Linear(l2, self.w3)
        l3 = nn.AddBias(l3_f, self.b3)

        l_o = nn.AddBias(nn.Linear(l3, self.w4), self.b4)
        return l_o

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_o = self.run(x)
        res = nn.SoftmaxLoss(y_o, y)
        return res

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        training_loss = float('inf')
        va = 0
        while va < .98:
            for x, y in dataset.iterate_once(self.bs):
                training_loss = self.get_loss(x, y)
                grads = nn.gradients(training_loss, self.parameters)
                training_loss = nn.as_scalar(training_loss)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(grads[i], -self.lr)
            va = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = .08
        self.initial_w = nn.Parameter(self.num_chars, 200)
        self.initial_b = nn.Parameter(1, 200)
        self.x_w = nn.Parameter(self.num_chars, 200)
        self.h_w = nn.Parameter(200, 200)
        self.b = nn.Parameter(1, 200)
        self.output_w = nn.Parameter(200, len(self.languages))
        self.output_b = nn.Parameter(1, len(self.languages))

        self.params = [self.initial_w, self.initial_b, self.x_w, self.h_w,
                       self.b, self.output_w, self.output_b]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        f_i = nn.Linear(xs[0], self.initial_w)
        h_i = nn.ReLU(nn.AddBias(f_i, self.initial_b))
        for char in xs[1:]:
            h_i = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.x_w), nn.Linear(h_i, self.h_w)), self.b))
        output = nn.AddBias(nn.Linear(h_i, self.output_w), self.output_b)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_o = self.run(xs)
        res = nn.SoftmaxLoss(y_o, y)
        return res

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        loss = math.inf
        acc = 0
        while acc < .85:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(gradients[i], -self.lr)
            acc = dataset.get_validation_accuracy()
