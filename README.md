# Model Implementation Demonstration

# 1. Data Reprocessing

```ruby
import pandas as pd
import numpy as np
```

To get all the data
```ruby
data = pd.read_csv('archive.csv')
data
```

To view the first two rows of the dataset
```ruby
data.head()
```

Filtering the dataframe
```ruby
short = data[['Year', 'Punxsutawney Phil', 'February Average Temperature']]
```
```ruby
short = short[short['Punxsutawney Phil'] != "No Record"]

# Filter out rows with missing values in the 'FebAvgTemp' column
short = short[short['February Average Temperature'].notnull()]

# Remove the last row (assuming you want to remove the last row)
short = short[:-1]

# Rename the columns
short.columns = ['Year', 'Punxsutawney Phil', 'February Average Temperature']

# Display the resulting DataFrame
short
```

Graph Plotting
```ruby
short.plot.bar(x = 'Year')
```

PREDICTIONS CORRECT: Winter = 32 degrees and no shadow means an early spring, full shadow means 6 more weeks of winter. So here punx would be correct for all top coldest years => prediction lines up with temps that are deemed winter
```ruby
minVals = short.sort_values(by='February Average Temperature')
minVals.head()
```

PREDICTIONS: But here punx would be incorrect every year for the warmest years(should have had predicted no shadow.
```ruby
maxVals = short.sort_values(by='February Average Temperature', ascending = False)
maxVals.head()
```
```ruby
import pandas as pd

# Load your data from the "archive.csv" file
short = pd.read_csv('archive.csv')

# Perform the preprocessing steps on the loaded data
short = short[short['Punxsutawney Phil'] != "No Record"]
short = short[short['February Average Temperature'].notnull()]
short = short[:-1]
short = short.rename(columns={'year': 'Year', 'Punxsutawney Phil': 'Punxsutawney Phil', 'February Average Temperature': 'February Average Temperature'})

# Display the resulting DataFrame
print(short)
```

# 2. Data Splitting

```ruby
from sklearn.model_selection import train_test_split

# Define Features (X) and Target (y)
X = data[['February Average Temperature', 'February Average Temperature (Northeast)', 'February Average Temperature (Midwest)', 'February Average Temperature (Pennsylvania)']]
y = data['Punxsutawney Phil']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
```

Imute missing values - fill in the missing values with appropriate values, such as the mean, median, or mode of the respective column (Remove NaN values)
```ruby
from sklearn.impute import SimpleImputer

# Initialize the imputer with a strategy (e.g., 'mean')
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your training data
X_train_imputed = imputer.fit_transform(X_train)

# Now, X_train_imputed contains no missing values, and you can use it for training

print("Imputed Training Data (X_train_imputed):")
print(X_train_imputed)
```

```ruby
X_train.dropna(axis=0, inplace=True)
y_train = y_train[X_train.index]  # Align target variable

# Now, X_train and y_train have no missing values, but some rows are dropped

# print the resulting dataset after dropping samples
print("Training Data After Dropping Samples (X_train):")
print(X_train)
```

```ruby
from sklearn.model_selection import train_test_split

# Assuming 'data' is your preprocessed DataFrame
X = data.drop('Punxsutawney Phil', axis=1)  # Features
y = data['Punxsutawney Phil']  # Target variable

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set dimensions:", X_train.shape, y_train.shape)
print("Validation set dimensions:", X_val.shape, y_val.shape)
```

Check for missing values in the 'year' column (NaN) and then remove them
```ruby
missing_years = X_train['Year'].isnull()
X_train = X_train[~missing_years]
```

Fill NaN values in 'Year' with a default value and split the 'Year' column into 'Start_Year' and 'End_Year' and convert to integers. Lastley drop the original 'Year' column
```ruby
X_train['Year'] = X_train['Year'].fillna('0')

X_train['Start_Year'] = X_train['Year'].str.split('-').str[0].astype(int)

X_train = X_train.drop('Year', axis=1)
```

Define X_train (features) and y_train (target variable) 
```ruby
data = pd.read_csv('archive.csv')
X_train = data.drop(columns=['Punxsutawney Phil']) 
y_train = data['Punxsutawney Phil'] 
```

 X_train and y_train will not contain any rows with missing values
```ruby
y_train = y_train[~X_train.isnull().any(axis=1)]
X_train = X_train.dropna(axis=0)
```

 Split the data to handle missed values
```ruby
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

Range the values properly
```ruby
# Fill NaN values in 'Year' with a default value (e.g., 0)
X_train['Year'] = X_train['Year'].fillna('0')
# Replace the range '1901-2000' with the midpoint value '1950'
X_train['Year'] = X_train['Year'].replace('1901-2000', '1950').astype(int)
```

Drop corresponding X_train to keep the data aligned
```ruby
y_train = y_train[~X_train.isnull().any(axis=1)]
X_train = X_train.dropna(axis=0)
print(y_train.isnull().sum())
```
```ruby
y_train = y_train.dropna()
X_train = X_train.dropna()
```

Check if both X_train and Y_train values are '0'
```ruby
missing_values_X = X_train.isnull().sum().sum()
print(f"Number of missing values in X_train: {missing_values_X}")
missing_values_y = y_train.isnull().sum()
print(f"Number of missing values in y_train: {missing_values_y}")
```

# 3. Select Machine Learning Algorithm

```ruby
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
```

# 4. Model Training - RandomForestClassifier

```ruby
model.fit(X_train, y_train)
```

Check the feature importances (if available) to see the expected feature names - this is for predictions
```ruby
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    for feature_name, importance in zip(feature_names, feature_importances):
        print(f"Feature: {feature_name}, Importance: {importance}")
else:
    print("This model does not provide feature importances.")

# If feature importances are available, it will print the feature names and their importances.
```

# 4. Model Training - Neural Network Model

```ruby
import tensorflow as tf
from tensorflow import keras
import numpy as np
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

```ruby
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # input shape should match the number of features
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

```ruby
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

```ruby
y_train = (y_train == 'Full Shadow').astype(int)
y_val = (y_val == 'Full Shadow').astype(int)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
```

```ruby
# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)

# Make predictions
y_pred = model.predict(X_val)

# Convert the predicted probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
```

Successfully optained predictions from neural network model
```ruby
new_data = pd.DataFrame({
    'Year': [2023],
    'February Average Temperature': [40.0],
    'February Average Temperature (Northeast)': [38.0],
    'February Average Temperature (Midwest)': [42.0],
    'February Average Temperature (Pennsylvania)': [41.0],
    'March Average Temperature': [45.0],
    'March Average Temperature (Northeast)': [43.0],
    'March Average Temperature (Midwest)': [47.0],
    'March Average Temperature (Pennsylvania)': [46.0]
})

predicted_class = model.predict(new_data)

print("Predicted Shadow:", predicted_class[0])
```

```ruby
predictions = model.predict(new_data)
```

Demonstrating FULL Shadow
```ruby
threshold = 0.5

# Predictions from the model
predictions = model.predict(new_data)

# Apply the threshold to classify predictions
predicted_labels = (predictions >= threshold).astype(int)

# Interpret the result
if predicted_labels[0] == 1:
    print("Predicted Shadow: Full Shadow")
else:
    print("Predicted Shadow: No Shadow")
```

Demonstrating NO Shadow
```ruby
threshold = 3

# Predictions from the model
predictions = model.predict(new_data)

# Apply the threshold to classify predictions
predicted_labels = (predictions >= threshold).astype(int)

# Interpret the result
if predicted_labels[0]:
    print("Predicted Shadow: Full Shadow")
else:
    print("Predicted Shadow: No Shadow")
```

# 5. Model Evaluation

1. Make Predictions
```ruby
y_pred = model.predict(X_val)
```

2. Calculate the Metrics
```ruby
from sklearn.metrics import precision_score

precision = precision_score(y_val, y_pred, average=None, zero_division=1)
print("Precision for each class:", precision)
```

3. Recall (Sensitivity) - measures the percentage of actual positive cases correctly predicted by the model
```ruby
 from sklearn.metrics import recall_score

recall = recall_score(y_val, y_pred, average=None)
print("Recall for each class:", recall)
```

4. F1-Score - harmonic mean of precision and recall and is often used to balance the trade-off between precision and recall
```ruby
from sklearn.metrics import f1_score

f1 = f1_score(y_val, y_pred, average=None)
print("F1-Score for each class:", f1)
```
```ruby
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
```



# How to use this model for Deployment

This is a FastAPI application for predicting if there is going to be a Groundhog Shadow (Punxsutawney Phil) based on temperature data. The purpose is to notice trends within the temperatures and the shadow of the Punxsutawney Phil to see if it can predict the temperature in February. If it's going to be winter or summer.

## Getting Started

These instructions will help you set up and run the application on your local machine. You can then deploy it to a server for production use.

### Prerequisites

You'll need the following installed on your machine:

- Python (3.7 or higher)
- pip (Python package manager)
- git (for version control)

### Installing

1. Clone the repository:

   ```bash
   git clone https://github.com/anchenayres/finalGroundHog.git

2. Navigate to the project directory:

   ```bash
   cd finalGroundHog

3. Create a Python virtual environment:

   ```bash
   python -m venv venv

4. Activate the virtual environment
   
  * On Windows:
    ```bash
    venv\Scripts\activate
    
  * On MacOs:
    ```bash
    source venv/bin/activate

5. Install the required Python packages
     ```bash
    pip install -r requirements.txt

  ### Running the Application

1. Start the FastAPI developer server
   ```bash
   uvicorn main:app --reload
The app will be accessible at http://127.0.0.1:8000

2. Access the FastAPI documentation at:

http://127.0.0.1:8000/docs

3. Access the main endpoint for predictions at:

http://127.0.0.1:8000/predict/

  ### Usage

The main prediction endpoint /predict/ accepts the following query parameters:

* year (int): The year for the prediction
* feb_temp (float): February average temperature
* mar_temp (float): March average temperature

Optional query parameters:

* northeast_feb (float): February average temperature for the Northeast region.
* midwest_feb (float): February average temperature for the Midwest region.
* penns_feb (float): February average temperature for Pennsylvania.
* northeast_mar (float): March average temperature for the Northeast region.
* midwest_mar (float): March average temperature for the Midwest region.
* penns_mar (float): March average temperature for Pennsylvania.

    ### Deployment

To deploy this application to a server, follow these steps:

1. Set up a production server with Python and the necessary environment
2. Install production-grade ASGI server software (e.g., Gunicorn)
3. Deploy your code to the production server
4. Run the ASGI server with your FastAPI application
5. Make sure to configure your production server to handle incoming requests and set up any necessary domain or IP address routing

Make sure to configure your production server to handle incoming requests and set up any necessary domain or IP address routing

  ### Built With

1. FastAPI - The web framework used
2. joblib - For loading the prediction model
3. pandas - For data manipulation
