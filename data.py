import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

# Read the CSV file
df = pd.read_csv("Agmar.csv")

# Preprocess the data
df['Price Date'] = pd.to_datetime(df['Price Date'])
df['Year'] = df['Price Date'].dt.year
df['Month'] = df['Price Date'].dt.month
df = df[['Year', 'Month', 'Modal Price (Rs./Quintal)']]

# Split into features and target
X = df[['Year', 'Month']]
y = df['Modal Price (Rs./Quintal)']

# Scale the features and target
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(1, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)
print('Mean Squared Error:', mse)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the scaled predictions
predictions = scaler.inverse_transform(predictions)

# Print some sample predictions
for i in range(5):
    print('Actual:', scaler.inverse_transform(y_test[i].reshape(-1, 1))[0][0])
    print('Predicted:', predictions[i][0])
    print()

# Save the model
model.save('model.h5')
