import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load the forex data
df = pd.read_csv('forex_data.csv')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_data[:, :-1], scaled_data[:, -1], test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the testing set
mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean squared error: {mse:.4f}")

# Make predictions on new data
new_data = np.array([[0.3, 0.2, 0.1, 0.4, 0.5]])
scaled_new_data = scaler.transform(new_data)
predicted_value = model.predict(scaled_new_data)[0][0]
print(f"Predicted value: {predicted_value:.4f}")