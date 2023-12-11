import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

app = dash.Dash()

app.layout = html.Div([
    dcc.Dropdown(
        id="my-dropdown",
        options=[
            {'label': 'Google', 'value': 'GOOGL'},
            {'label': 'Apple', 'value': 'AAPL'},
            {'label': 'Microsoft', 'value': 'MSFT'},
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Lenovo', 'value': 'LNVGY'},
        ],
        value='GOOGL'
    ),
    dcc.Graph(id="my-graph")
], style={'width': '500'})


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def graph(selected_dropdown_value):
    df = web.DataReader(selected_dropdown_value, 'yahoo', dt(2018, 1, 1), dt.now())

    
    sample_input = np.array([df['Close'].values[-1]])
    prediction = model.predict(sample_input)
    
   
    sns.histplot(df['Close'], kde=True)

 
    mean, p_value = stats.ttest_1samp(df['Close'], 0)

    return {
        'data': [{'x': df.index, 'y': df['Close']}],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}},
        'annotations': [
            {
                'x': 0,
                'y': 1,
                'xref': 'paper',
                'yref': 'paper',
                'text': f'Model Prediction: {prediction[0][0]}',
                'showarrow': False,
                'font': {'color': 'green'}
            },
            {
                'x': 0,
                'y': 0.9,
                'xref': 'paper',
                'yref': 'paper',
                'text': f'Mean: {mean}, p-value: {p_value}',
                'showarrow': False,
                'font': {'color': 'blue'}
            }
        ]
    }


if __name__ == "__main__":
    app.run_server(debug=True)
