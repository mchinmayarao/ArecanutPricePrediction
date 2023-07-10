# Agmar LSTM Model

This project implements a Long Short-Term Memory (LSTM) model for predicting the modal price of a commodity based on the historical data provided in the "Agmar.csv" dataset.

## Project Overview

The main steps of the project are as follows:

1. Data Preprocessing: The dataset is preprocessed by converting the "Price Date" column to datetime format and extracting the year and month. The relevant columns are selected, and the data is scaled using the MinMaxScaler.

2. Model Building: A sequential LSTM model is constructed using Keras. It consists of an LSTM layer with 128 units and a dense layer with 1 unit.

3. Model Training: The model is compiled with the Adam optimizer and trained on the preprocessed data for 100 epochs with a batch size of 32.

4. Model Evaluation: The trained model is evaluated using the testing data, and the Mean Squared Error (MSE) is calculated as the evaluation metric.

5. Prediction and Visualization: The model is utilized to make predictions on the testing data, and the results are visualized by comparing the actual and predicted values.

6. Model Saving: The trained LSTM model is saved to a file named "model.h5" for future use.





## Prerequisites

To run the notebook and reproduce the results, the following dependencies are required:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorflow

## Usage

1. Clone this repository or download the "agmar_lstm_model.ipynb" notebook file.

2. Install the required dependencies using pip or conda:

   ```shell
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
   ```

3. Open the "agmar_lstm_model.ipynb" notebook in Jupyter Notebook or JupyterLab.

4. Execute each cell in the notebook sequentially to reproduce the results and visualize the data.

5. Customize the code as needed, such as modifying hyperparameters, adding additional visualizations, or applying different preprocessing techniques.

## References

- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [matplotlib Documentation](https://matplotlib.org/contents.html)
- [seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)



Feel free to explore and utilize this LSTM model for predicting modal prices based on historical agricultural data. If you have any questions or suggestions, please feel free to reach out.
