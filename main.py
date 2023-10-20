import streamlit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error
streamlit.set_page_config(page_title="Nellai Agency-Time Series Forecasting App",page_icon="📊",layout="wide")
header=streamlit.container()
dataset=streamlit.container()
features=streamlit.container()
modelTraining=streamlit.container()
streamlit.markdown("""
<style>
.main{
background-color:#0E1117;
color:white;
}
body { 
color:white;
}
</style>
""",unsafe_allow_html=True
 )

with header:
    streamlit.header('📊 Mini Project:Time Series Forecasting using Univariate LSTM Model for')
    streamlit.title(' Nellai Agencies')
    streamlit.text('Designed by:Isaac Ritharson,Karunya Institute of Techology and Sciences')
    streamlit.text('Finding the Total Net Income amount of sales based on previous 31 days income')

with dataset:
    streamlit.header('Sample monthly sales data (May-Jun) ')
    streamlit.text('This data is an example of an original data acquired from the company itslef')


    sales_data= pd.read_csv('data.csv')
    streamlit.write(sales_data.head(5))

    streamlit.subheader('Calculated sum of Net sales May 17,2022 to Jun 16,2022')
    streamlit.text('only has sales Net amount including taxes')

    sales_data = pd.read_csv('Book1.csv')
    streamlit.write(sales_data.head(31))

    gross_dis= pd.DataFrame(sales_data['Net sales'])
    streamlit.line_chart(gross_dis)

with features:
    streamlit.header('Features')
    streamlit.markdown('* **Net Sales:** i created this because to sum the net amount to forecast the future income')
    streamlit.markdown('* **Date:** It is important to know the date as it is the independent variable')

with modelTraining:
    streamlit.title('Time to train LSTM model📈')

    sel_col,dis_col=streamlit.columns(2)

    uploaded_file = streamlit.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        streamlit.header('Best Performing Product of this month🥇')
        sel_col, dis_col = streamlit.columns(2)
        df3 = df[['Brand Desc', 'Net Amount']]
        df4 = df3.groupby('Brand Desc').sum()
        streamlit.bar_chart(df4)

        streamlit.line_chart(df4)
        df4 = df4.sort_values('Net Amount')
        streamlit.write("therefore,the best performing product of this month is Milk Bikis ")
        df4.loc[df4['Net Amount'].idxmax()]

        streamlit.header('Most prominent customer of the previous month👨‍👦')
        n_companies = sel_col.slider('Slide to view more prominent customers of this month', min_value=1, max_value=100,
                                     value=26,
                                     step=1)

        df4 = df[['Sold To Party Name', 'Net Amount']]
        df4 = df4.groupby('Sold To Party Name').sum()
        df4 = df4.sort_values('Net Amount')
        df4 = df4.tail(n_companies)
        streamlit.bar_chart(df4)
        streamlit.write("therefore,the most prominent customer of this month is M M A stores ")
        df4.loc[df4['Net Amount'].idxmax()]

        sel_col, dis_col = streamlit.columns(2)
        input_feature = sel_col.text_input('which Product should be used as the input feature?', 'Milk Bikis')

        streamlit.subheader('Raw Company Sales Data Display')
        streamlit.write(df)
        streamlit.write(len(df))
        df7= df.loc[df['Brand Desc'] == input_feature ]
        streamlit.write(df7)
        df12 = df7[['Invoice Date', 'Net Amount']]
        df13 = df12.groupby('Invoice Date').sum()

        streamlit.subheader('Sum of '+ input_feature +' sales for each day in month')
        streamlit.write(df13)

        streamlit.text('Plotting the Graph')
        streamlit.line_chart(df13)


        df1 = df[['Invoice Date', 'Net Amount']]

        df1 = df1.groupby('Invoice Date').sum()

        streamlit.subheader('Sum of Overall total sales for each day in month')
        streamlit.write(df1)

        streamlit.text('Plotting the Graph')
        streamlit.line_chart(df1)

        streamlit.text('here you get to choose the no of days for forecast and see how the performance changes!')
        n_steps = sel_col.selectbox('select the no of previous day sales to refer(timestamps)',options=[9]
                                 )
        future = sel_col.selectbox('No of days you want to predict', options=[27])

        streamlit.write(n_steps)

        timeseries_data = df13['Net Amount'].to_numpy()
        streamlit.text('timeseries_data(Best Product sales)')
        streamlit.write(timeseries_data)


        # Preparing independent and dependent features
        def prepare_data(timeseries_data, n_steps):
            X, y = [], []
            for i in range(len(timeseries_data)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(timeseries_data) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)


        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)

        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')
        # fit model
        model.fit(X, y, epochs=1900,verbose=0)

        i = 0
        x_input = np.array(df13['Net Amount'].head(n_steps+i))
        temp_input = x_input.tolist()
        lst_output = []

        while (i < future):
            if (len(temp_input) > n_steps):
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input))
                # print(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i, yhat))
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                # print(temp_input)\
                lst_output.append(yhat[0][0])
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i + 1
        day_new = np.arange(0, 27)
        day_pred = np.arange(27, 27 + future)

        plt.plot(day_new, timeseries_data)
        plt.plot(day_pred, lst_output)
        streamlit.pyplot(fig=plt)

        X_test = np.array(df13['Net Amount'].head(n_steps)).reshape((-1, 9, n_features))
        y_test = np.array([df13['Net Amount'][10]])
        y_pred = model.predict(X_test)
        import math
        streamlit.write("mean_absolute_error", math.sqrt(mean_absolute_error(y_test, y_pred)))

        timeseries_data = df1['Net Amount'].to_numpy()
        streamlit.text('timeseries_data(overall sales)')
        streamlit.write(timeseries_data)


        # Preparing independent and dependent features
        def prepare_data(timeseries_data, n_steps):
            X, y = [], []
            for i in range(len(timeseries_data)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(timeseries_data) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)

        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        model = Sequential()
        model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(200, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')
        # fit model
        model.fit(X, y, epochs=1900,verbose=0)

        #prediction

        x_input = np.array(df1['Net Amount'].tail(n_steps))
        temp_input = x_input.tolist()
        lst_output = []
        i = 0
        while (i < future):
            if (len(temp_input) > n_steps):
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input))
                # print(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i, yhat))
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.append(yhat[0][0])
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i + 1

        day_new = np.arange(0, 27)
        day_pred = np.arange(27, 27+future)

        plt.plot(day_new, timeseries_data)
        plt.plot(day_pred, lst_output)
        streamlit.pyplot(fig=plt)

        X_test = np.array([221687.16, 193129.18 ,125863.47 ,161339.53 ,154265.03, 296935.18, 200765.98,
   71127.09  ,13342.52],).reshape((-1, 9, n_features))
        y_test = np.array([171749.92])
        y_pred = model.predict(X_test)
        import math

        streamlit.write("mean_absolute_error", math.sqrt(mean_absolute_error(y_test, y_pred)))


        streamlit.header('Profit Calculation')
        sel_col, dis_col = streamlit.columns(2)
        input_features = sel_col.text_input('Enter the amount spent on buying supplies and goods',4500000)
        amount_spent = input_features
        total_sales = df1['Net Amount'].sum()
        streamlit.text('The Net total amount of sales this month is:')
        streamlit.write(total_sales)
        profit = total_sales - float(amount_spent)
        streamlit.text('Therefore Profit for this month is:')
        streamlit.write(profit)















