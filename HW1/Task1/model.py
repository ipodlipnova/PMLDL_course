from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

iris_data = load_iris() # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


# Build the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1', use_bias=True, bias_initializer='ones'))
model.add(Dense(10, activation='relu', name='fc2', use_bias=True, bias_initializer='ones'))
model.add(Dense(3, activation='softmax', name='output', use_bias=True, bias_initializer='ones'))
# Adam optimizer with learning rate of 0.001
optimizer = Adam (lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print('Neural Network Model Summary: ')
print(model.summary())


def network_classification():
    """
    Classification using neural network
    """
    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    # Test on unseen data
    results = model.evaluate(test_x, test_y)
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))
