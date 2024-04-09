import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x_point)
        
    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        score = nn.DotProduct(self.w, x_point)
        scalar_score = nn.as_scalar(score)
        if scalar_score >= 0:
            return 1
        else:
            return -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            misclassified = False
            for x_point, label in dataset.iterate_once(1):
                prediction = self.get_prediction(x_point)
                label = nn.as_scalar(label)
                if prediction != label:
                    misclassified = True
                    self.w.update(label, x_point)
            if not misclassified:
                break
                
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        
        self.W1 = nn.Parameter(1, 20)
        self.b1 = nn.Parameter(1, 20)
        
        self.W2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """\
        # first layer
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        
        # Second layer
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        
        return x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        loss = nn.SquareLoss(predictions, y)
        return loss

    def train_model(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.01
        count = 0
        
        while True:
            for x, y in dataset.iterate_once(10):  
                loss = self.get_loss(x, y)
                count += 1
                # Compute gradients
                gradients = nn.gradients([self.W1, self.b1, self.W2, self.b2], loss)
                
                # Update parameters
                self.W1.update(-learning_rate, gradients[0])
                self.b1.update(-learning_rate, gradients[1])
                self.W2.update(-learning_rate, gradients[2])
                self.b2.update(-learning_rate, gradients[3])
            
            # Check the loss on the entire dataset to decide when to stop training
            total_loss = 0
            for x, y in dataset.iterate_once(1):
                total_loss += nn.as_scalar(self.get_loss(x, y))
            print("Total loss:", total_loss)
            
            #when to stop
            if total_loss/count < learning_rate:
                break
       

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
        # Input layer to hidden layer
        self.W1 = nn.Parameter(784, 256)  
        self.b1 = nn.Parameter(1, 256)

        # Hidden layer to output layer
        self.W2 = nn.Parameter(256, 128)  
        self.b2 = nn.Parameter(1, 128)

        self.W3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)

        # Learning rate
        self.learning_rate = 0.02


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
        x = nn.Linear(x, self.W1)
        x = nn.ReLU(nn.AddBias(x, self.b1))
        x = nn.Linear(x, self.W2)
        x = nn.ReLU(nn.AddBias(x, self.b2))
        x = nn.Linear(x, self.W3)
        x = nn.AddBias(x, self.b3)
        return x

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
        prediction = self.run(x)
        return nn.SoftmaxLoss(prediction, y)


    def train_model(self, dataset):
        """
        Trains the model.
        """
        Accurate = False
        epoch_num = 0
        while not Accurate:
            epoch_num += 1
            for x, y in dataset.iterate_once(50):
                loss = self.get_loss(x, y)
                g_W1,g_b1,g_W2,g_b2,g_W3,g_b3 = nn.gradients([self.W1,self.b1,self.W2,self.b2,self.W3,self.b3], loss)
                self.W1.update(-self.learning_rate, g_W1)
                self.b1.update(-self.learning_rate, g_b1)
                self.W2.update(-self.learning_rate, g_W2)
                self.b2.update(-self.learning_rate, g_b2)
                self.W3.update(-self.learning_rate, g_W3)
                self.b3.update(-self.learning_rate, g_b3)
            if(dataset.get_validation_accuracy() > 0.975):
                Accurate = True
        print("Epoch: " + str(epoch_num))
       
