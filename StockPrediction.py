# Basic Imports
import time
import numpy as np
import warnings

# TensorFlow and Keras Imports
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

# scikit-learn and imblearn Imports
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

# Ignore warnings
warnings.filterwarnings("ignore")

class StockPrediction():
    def __init__(self, Xtr, ytr, Xts, yts, Xval=None, yval=None, sampling=None, description=""):
        """
        Initializes the StockPrediction class with training and testing data, and optional validation data.
        Optionally applies undersampling or oversampling to the training data.

        Parameters:
            Xtr (numpy array): Training data features
            ytr (pandas dataframe): Training data labels
            Xts (numpy array): Testing data features
            yts (pandas dataframe): Testing data labels
            Xval (numpy array, optional): Validation data features
            yval (pandas dataframe, optional): Validation data labels
            sampling (str, optional): Type of sampling to apply to the training data. Can be 'under' for undersampling or 'over' for oversampling.
            description (str, optional): Description of the current instance.
        """
        self.Xtr, self.ytr, self.Xts, self.yts, self.Xvl, self.yvl = Xtr, ytr, Xts, yts, Xval, yval
        self.sampling = sampling
        self.description = description

        if self.sampling == 'under':
            # Apply undersampling to the training data
            undersample = NearMiss(version=1, n_neighbors=3)
            self.Xtr, self.ytr = undersample.fit_resample(self.Xtr, self.ytr)
        elif sampling == 'over':
            # Apply oversampling to the training data
            ros = RandomOverSampler(random_state=0)
            self.Xtr, self.ytr = ros.fit_resample(self.Xtr, self.ytr)

        self._describe()

    def _describe(self):
        """
        Prints information about the current instance, including the image resolution, sampling type,
        number of classes, training set size, validation set size (if applicable), and testing set size.
        Also prints the number of instances of each class in the training and testing sets.

        Parameters:
            self: instance of the current object
        """
        print("="*30)
        if self.description:
            print(f"{self.description}\n")
        # Calculate the image resolution based on the number of features in the training data
        self.image_dimension = int(np.sqrt(self.Xtr.shape[1]))
        print(f"Image Resolution        : {self.image_dimension}x{self.image_dimension}")
        print(f"Sampling                : {self.sampling}\n")

        # Get a list of unique classes in the training labels
        self.classes = list(set(list(self.ytr['Label'])))
        print(f"{len(self.classes)} Classes               : {self.classes}")
        # Print the number of instances of each class in the training and testing sets
        for clas_ in self.classes:
            print(f"Class {clas_}; Training: {list(self.ytr['Label']).count(clas_)}, Testing: {list(self.yts['Label']).count(clas_)} ")
        # Get the size of the training set
        self.train_points = self.Xtr.shape[0]
        print(f"\nTraining Set Size       : {self.train_points}")
        # If a validation set is provided, get its size
        if not self.Xval.empty:
            self.valid_points = self.Xval.shape[0]
            print(f"Validation Set Size     : {self.valid_points}")
        # Get the size of the testing set
        self.test_points = self.Xts.shape[0]
        print(f"Testing Set Size : {self.test_points}\n")
        print("="*30)

    def buildNN(self, nin, nout, opt, lr, nh):
        """
        Builds a neural network model with a specified number of hidden layers.

        Parameters:
            self: instance of the current object
            nin (int): Number of input features
            nout (int): Number of output classes
            opt (keras.optimizers): Optimizer to use for the model
            lr (float): Learning rate for the optimizer
            nh (int): Number of hidden layers

        Returns:
            model (keras.Model): The built neural network model
        """
        model = Sequential()
        # If there is more than one hidden layer, add them to the model
        if nh > 1:
            for i in range(0, nh + 1):
                model.add(Dense(units=128, input_shape=(nin,), activation='sigmoid', name='hidden '+ str(i+1)))
        else:
            # Add a single hidden layer and the output layer to the model
            model.add(Dense(units=128, input_shape=(nin,), activation='sigmoid', name='hidden1'))
            model.add(Dense(units=nout, activation='sigmoid', name='output'))
            # Compile the model with the specified optimizer and loss function
            opt_ = opt(learning_rate=lr)
            model.compile(optimizer=opt_, loss='binary_crossentropy', metrics=['accuracy'])
            # Print a summary of the model
            model.summary()
        return model

    def nn_cl_bo2(self, neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
        """
        Builds and evaluates a neural network model with specified hyperparameters using stratified k-fold cross validation.

        Parameters:
            self: instance of the current object
            neurons (int): Number of neurons in the hidden layers
            activation (str): Activation function to use in the hidden layers
            optimizer (str): Optimizer to use for training the model
            learning_rate (float): Learning rate for the optimizer
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train the model
            layers1 (int): Number of hidden layers with the specified number of neurons
            layers2 (int): Number of hidden layers with the specified number of neurons and dropout
            normalization (bool): Flag to indicate whether to use batch normalization
            dropout (bool): Flag to indicate whether to use dropout
            dropout_rate (float): Dropout rate for the dropout layers

        Returns:
            score (float): Mean accuracy score from stratified k-fold cross validation
        """
        # Dictionary of optimizers
        optimizerD = {'Adam': Adam(lr=learning_rate), 'SGD': SGD(lr=learning_rate),
                    'RMSprop': RMSprop(lr=learning_rate), 'Adadelta': Adadelta(lr=learning_rate),
                    'Adagrad': Adagrad(lr=learning_rate), 'Adamax': Adamax(lr=learning_rate),
                    'Nadam': Nadam(lr=learning_rate), 'Ftrl': Ftrl(lr=learning_rate)}
        # Round and map the neurons, batch size, epochs, and number of layers to integers
        neurons = round(neurons)
        batch_size = round(batch_size)
        epochs = round(epochs)
        layers1 = round(layers1)
        layers2 = round(layers2)
        # Define the function to build the neural network model
        def nn_cl_fun():
            """
            This function builds a neural network model with the specified hyperparameters.
            
            Parameters:
                - neurons: number of neurons in the hidden layers
                - activation: activation function for the hidden layers
                - optimizer: optimizer for the model
                - learning_rate: learning rate for the optimizer
                - batch_size: batch size for training
                - epochs: number of epochs for training
                - layers1: number of hidden layers before the dropout layer
                - layers2: number of hidden layers after the dropout layer
                - normalization: whether or not to use batch normalization
                - dropout: whether or not to use dropout
                - dropout_rate: dropout rate
            
            Returns:
                - A compiled neural network model
            """
            # Create an empty Sequential model
            nn = Sequential()
            # Add the first hidden layer with the specified number of neurons and activation function
            nn.add(Dense(neurons, input_dim=self.Xtr.shape[1], activation=activation))
            # If normalization is set to True, add a batch normalization layer
            if normalization > 0.5:
                nn.add(BatchNormalization())
            # Add the specified number of hidden layers with the specified number of neurons and activation function
            for i in range(layers1):
                nn.add(Dense(neurons, activation=activation))
            # If dropout is set to True, add a dropout layer with the specified dropout rate
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate, seed=123))
            # Add the specified number of hidden layers with the specified number of neurons and activation function
            for i in range(layers2):
                nn.add(Dense(neurons, activation=activation))
            # Add the output layer with a sigmoid activation function
            nn.add(Dense(1, activation='sigmoid'))
            # Compile the model with the specified loss function, optimizer, and metrics
            nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return nn

        # Create an early stopping callback to stop training when the accuracy stops improving
        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        # Create a KerasClassifier object with the nn_cl_fun function as the build function
        nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
        # Create a stratified k-fold cross validator with 5 folds
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        # Use cross_val_score to evaluate the model using stratified k-fold cross validation
        score = cross_val_score(nn, self.Xtr, self.ytr, scoring=make_scorer(accuracy_score), cv=kfold, fit_params={'callbacks': [es]}).mean()
        return score

    def nn_cl_fun(self, params_nn_):
        """
        This function creates a neural network model with the specified parameters.

        Parameters
        ----------
        params_nn_ : dict
            A dictionary containing the model parameters.
            The keys of the dictionary should be:
            - 'neurons' (int) : number of neurons in each layer
            - 'activation' (string) : activation function for the layers
            - 'normalization' (float) : whether or not to use batch normalization (values should be 0 or 1)
            - 'dropout' (float) : whether or not to use dropout (values should be 0 or 1)
            - 'dropout_rate' (float) : dropout rate (float between 0 and 1)
            - 'layers1' (int) : number of hidden layers with dropout
            - 'layers2' (int) : number of hidden layers without dropout
            - 'optimizer' (keras optimizer object) : optimizer to be used during training

        Returns
        -------
        model : keras.Model
            A compiled neural network model.
        """
        nn = Sequential()
        nn.add(Dense(params_nn_['neurons'], input_dim=self.Xtr.shape[1], activation=params_nn_['activation']))
        if params_nn_['normalization'] > 0.5:
            nn.add(BatchNormalization())
        for i in range(params_nn_['layers1']):
            nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
            if params_nn_['dropout'] > 0.5:
                nn.add(Dropout(params_nn_['dropout_rate'], seed=123))
        for i in range(params_nn_['layers2']):
            nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=params_nn_['optimizer'], metrics=['accuracy'])
        return nn

    def ml_evaluate_model(self, model, describe = False):
        """Evaluate a machine learning model.

        Parameters
        ----------
        model: sklearn model
            The model to be evaluated.
        describe: bool, optional
            Whether to describe the datasets. The default is False.

        Returns
        -------
        model: sklearn model
            The trained model.
        cm: numpy array
            The confusion matrix.
        """
        if describe:
            self._describe()
        print("-"*30)
        print(f"\nModel                   : {model}")
        train_time_start = time.time()
        model.fit(self.Xtr, self.ytr)
        train_time_stop = time.time()
        print(f"Training Time           : {round(train_time_stop - train_time_start, 3)}s")

        test_time_start = time.time()
        ypred = model.predict(self.Xts)
        test_time_stop = time.time()
        print(f"Test Time               : {round(test_time_stop - test_time_start, 3)}s\n")

        accuracy_perc = round(accuracy_score(self.yts, ypred)*100, 5)
        print(f"Accuracy                : {accuracy_perc}%")
        cm=confusion_matrix(self.yts,ypred)

        for clas in self.classes:
            false_perc = round(np.sum(cm[clas][[i for i in range(len(self.classes)) if i != clas]])*100/np.sum(cm[clas]), 5)
            print(f"False class {clas} rate      : {false_perc}%")
        return(model, cm) 

    def nn_evaluate_model(self, model, batch_size, nit, nepoch_per_it, describe = False):
        """Evaluate a neural network model.

        Parameters
        ----------
        model : Keras model
            The model to be evaluated.
        batch_size : int
            The batch size to use for training.
        nit : int
            The number of iterations to perform.
        nepoch_per_it : int
            The number of epochs to perform per iteration.
        describe : bool, optional
            Whether to describe the datasets. The default is False.

        Returns
        -------
        model : Keras model
            The trained model.
        cm : numpy array
            The confusion matrix.

        """
        if describe:
            self._describe()
        print("-"*30)
        print(f"\nModel : {model}")
        # Loss, accuracy and epoch per iteration
        loss = np.zeros(nit)
        acc = np.zeros(nit)
        epoch_it = np.zeros(nit)
        train_time_start = time.time()
        # Main iteration loop
        for it in range(nit):
            # Continue the fit of the model
            init_epoch = it*nepoch_per_it
            model.fit(self.Xtr, self.ytr, epochs=nepoch_per_it, validation_data=(self.Xval, self.yval), batch_size=batch_size, verbose=0)
            # Measure the loss and accuracy on the training data
            lossi, acci = model.evaluate(self.Xts, self.yts, verbose=0)
            epochi = (it+1)*nepoch_per_it
            epoch_it[it] = epochi
            loss[it] = lossi
            acc[it] = acci
            print("epoch=%4d loss=%12.4e acc=%7.5f" % (epochi,lossi,acci))

        train_time_stop = time.time()
        print ("\n")
        print(f"Training Time           : {round(train_time_stop - train_time_start, 3)}s")

        test_time_start = time.time()
        # predict crisp classes for test set
        yhat_classes = (model.predict(self.Xts) > 0.5).astype("int32")
        yhat_classes = yhat_classes[:, 0]

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(self.yts, yhat_classes)
        print('Accuracy: %f' % accuracy) 
        test_time_stop = time.time()
        print(f"Test Time               : {round(test_time_stop - test_time_start, 3)}s\n")

        accuracy_perc = round(accuracy*100, 5)
        print(f"Accuracy                : {accuracy_perc}%")
        cm=confusion_matrix(self.yts,yhat_classes)

        for clas in self.classes:
            false_perc = round(np.sum(cm[clas][[i for i in range(len(self.classes)) if i != clas]])*100/np.sum(cm[clas]), 5)
            print(f"False class {clas} rate      : {false_perc}%")
        return(model, cm) 