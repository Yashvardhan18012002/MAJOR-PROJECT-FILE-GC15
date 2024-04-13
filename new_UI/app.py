# pip install rasa==2.0.2
# rasa run actions
# rasa run --cors "*" --enable-api
# pip install rasa-nlu==0.11.5
# greenlet==0.4.16
# pip install slackclient==1.3.1

# rasa train



from flask import *
import pymysql
import pandas as pd
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# from urllib import request
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
from tensorflow.keras.models import save_model, load_model
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
from sklearn import preprocessing
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
from sklearn import preprocessing
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from yahoo_fin import stock_info as si
import requests
from bs4 import BeautifulSoup
import requests_html 
import json
import requests
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

global usrname
usrname = ""

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="stockmarket",charset='utf8')
        return connection
    except:
        print("Something went wrong in database Connection")

def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

# con=dbConnection()
# cursor=con.cursor()

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'
output=[]#("message stark","hi")]

##########################################################################################################
#                                           Register
##########################################################################################################
@app.route("/register", methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        #Parse form data    
        # print("hii register")
        USERNAME = request.form['USERNAME']
        EMAIL = request.form['EMAIL']
        PASSWORD = request.form['PASSWORD']
        MOBILE = request.form['MOBILE']

        print(USERNAME,EMAIL,PASSWORD,MOBILE)

        try: 
            con = dbConnection()
            cursor = con.cursor()
            sql1 = "INSERT INTO tblregister (uname, email, password,mobile) VALUES (%s, %s, %s, %s)"
            val1 = (USERNAME,EMAIL,PASSWORD,MOBILE)
            cursor.execute(sql1, val1)
            print("query 1 submitted")
            con.commit()
            dbClose()

            FinalMsg = "Congrats! Your account registerd successfully!"
            return FinalMsg
        except:
            con.rollback()
            msg = "Database Error occured"
            return msg
         
        finally:
            dbClose()
        
    return render_template("register.html")
##########################################################################################################
#                                               Login
##########################################################################################################
@app.route("/", methods = ['POST', 'GET'])
def login():
    
    return render_template('login.html')


@app.route("/login1", methods = ['POST', 'GET'])
def login1():
    if request.method == 'POST':
        email = request.form['Email']
        password = request.form['password'] 

        print(email,password)

        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM tblregister WHERE email = %s AND password = %s', (email, password))
        result = cursor.fetchone()
        dbClose()
        print("result")
        print(result)
        if result_count>0:
            print("len of result")
            session['uname'] = result[1]
            session['userid'] = result[0]

            global usrname
            usrname += session.get("uname")
            return "success"
        else:
            return "fail"
   
#########################################################################################################
#                                       Home page
##########################################################################################################
@app.route("/index")
def index():
    data = yf.download(tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],group_by = 'ticker',threads=True,period='1mo', interval='1d')

    data.reset_index(level=0, inplace=True)

    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Adj Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'UBER', period='1d', interval='1d')
    df5 = yf.download(tickers = 'TSLA', period='1d', interval='1d')
    df6 = yf.download(tickers = 'TWTR', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "GOOGL")
    df4.insert(0, "Ticker", "UBER")
    df5.insert(0, "Ticker", "TSLA")
    df6.insert(0, "Ticker", "TWTR")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)
    
    return render_template('index.html',plot_div_left=plot_div_left,recent_stocks=recent_stocks)

##########################################################################################################
#                                               Search Page
##########################################################################################################
dff = pd.read_csv("Fortune_1000.csv")

@app.route("/search", methods = ['POST', 'GET'])
def search():

    temperatures = list(dff['company'])

    return render_template('search.html', values=temperatures,languages=temperatures)


##########################################################################################################
#                                       Prediction
##########################################################################################################
# function to calculate percentage difference considering baseValue as 100%
def percentageChange(baseValue, currentValue):
    return((float(currentValue)-baseValue) / abs(baseValue)) *100.00

# function to get the actual value using baseValue and percentage
def reversePercentageChange(baseValue, percentage):
    return float(baseValue) + float(baseValue * percentage / 100.00)

# function to transform a list of values into the list of percentages. For calculating percentages for each element in the list
# the base is always the previous element in the list.
def transformToPercentageChange(x):
    baseValue = x[0]
    x[0] = 0
    for i in range(1,len(x)):
        pChange = percentageChange(baseValue,x[i])
        baseValue = x[i]
        x[i] = pChange

# function to transform a list of percentages to the list of actual values. For calculating actual values for each element in the list
# the base is always the previous calculated element in the list.

dictionaryofdateandprice={}

# Training and evaluation with 10-fold cross-validation
def cross_validate_model(X, y,datatoprdict, n_splits=2):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mape_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshaping data for LSTM model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Building LSTM model
        model = build_model(X_train.shape[1], num_features=1)

        # Training the model
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        # Loading the saved model
        # loaded_model = load_model('lstm_model.h5')
        # Making predictions
        predictions = model.predict(X_test)
        predictions1 = model.predict(datatoprdict)
        # print('----------------actual------------------')
        # print(X_test.shape)
        # print(X_test[:1])
        # print('-----------------prdicted-----------------')
        # print(datatoprdict.shape)
        # print(datatoprdict[:1])
        # Calculating MAPE
        mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
        mape_scores.append(mape)

    return np.mean(mape_scores),predictions

def reverseTransformToPercentageChange(baseValue, x):
    x_transform = []
    for i in range(0,len(x)):
        value = reversePercentageChange(baseValue,x[i])
        baseValue = value
        x_transform.append(value)
    return x_transform

def build_model(time_steps, num_features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, num_features)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetching data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocessing data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Creating time series data
def create_time_series_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def datapred(ticker_name, number_of_days):    

    # Fetching data
    ticker = ticker_name
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    df = fetch_data(ticker, start_date, end_date)
    maindf = df
    new_data1 = maindf
    
    # Preprocessing data
    scaled_data, scaler = preprocess_data(df)

    # Creating time series data
    time_steps = 60
    X_main, y_main = create_time_series_data(scaled_data, time_steps)
    print(X_main.shape)
    print()
    print(y_main.shape)

    ##################################################################################################

    df_ml = df[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'],1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]

    X_forecast = np.array(X_forecast).astype(np.float32)
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0],X_forecast.shape[1],1))

    print(X_main.shape)
    print(X_forecast.shape)
    mape,forecast_prediction = cross_validate_model(X_main, y_main, X_forecast)
    accuracy= 100- mape
    
    forecast = forecast_prediction.tolist()
    print("----------------------")
    print(accuracy)
    print("----------------------")

    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i][0]*100)
        
    pred_df = pd.DataFrame(pred_dict)
        
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')
    
    return plot_div_pred, new_data1, forecast

@app.route("/prediction", methods = ['POST', 'GET'])
def prediction():
    if request.method=='POST':
        company_name = request.form['ticker']
        number_of_days = request.form['days']

        cmp_ticker = dff[dff["company"]==company_name]
        ticker_name = cmp_ticker["Ticker"].values[0]


        plot_div_pred, new_data1, forecast = datapred(ticker_name, number_of_days)
        print('---------------------------------')
        new_data1['Date'] = new_data1.index
        
        # print(new_data1.columns)
        # print(new_data1)

        normal_fig = go.Figure([go.Scatter(x=new_data1['Date'], y=new_data1['Close'])])
        normal_fig.update_xaxes(rangeslider_visible=True)
        normal_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
        plot_div = plot(normal_fig, auto_open=False, output_type='div')


        ticker = pd.read_csv('Tickers.csv')
        to_search = ticker_name
        ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                        'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
        
        for i in range(0,ticker.shape[0]):
            if ticker.Symbol[i] == to_search:
                print()
                print("ticker_name")
                print(ticker.Symbol[i])
                print()
                Symbol = ticker.Symbol[i]
                Name = ticker.Name[i]
                Last_Sale = ticker.Last_Sale[i]
                Net_Change = ticker.Net_Change[i]
                Percent_Change = ticker.Percent_Change[i]
                Market_Cap = ticker.Market_Cap[i]
                Country = ticker.Country[i]
                IPO_Year = ticker.IPO_Year[i]
                Volume = ticker.Volume[i]
                Sector = ticker.Sector[i]
                Industry = ticker.Industry[i]
                break

        company_name = company_name
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")

        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng = []
        lnk = []
        for i in table1:
            a = i["href"]
            lnk.append(a)
            b = i.text
            headng.append(b)

        headng = headng[:8]
        lnk = lnk[:8]
        
        link_lst = []
        for i in range(len(lnk)):
            a = "https://news.google.com/"+lnk[i][2:]
            link_lst.append(a)
        lnk = link_lst

        news_flst = zip(headng,lnk)

        # dataoftweets=get_tweets(company_name)

        print()
        print("Symbol")
        print(Symbol)
        print()


        return render_template('result.html',Symbol=Symbol, Name=Name,news_flst=news_flst,plot_div_pred=plot_div_pred, plot_div=plot_div,forecast=forecast,ticker_value=ticker_name,
            number_of_days=number_of_days,Last_Sale=Last_Sale,Net_Change=Net_Change,
            Percent_Change=Percent_Change,Market_Cap=Market_Cap,Country=Country,IPO_Year=IPO_Year,Volume=Volume,Sector=Sector,Industry=Industry)
    return render_template('result.html')
##########################################################################################################
#                                               contact
##########################################################################################################
@app.route("/contact", methods = ['POST', 'GET'])
def contact():
    username=session.get('uname')
    return render_template('contact.html',firstName=username)
##########################################################################################################
#                                               contact
##########################################################################################################

@app.route("/ticker", methods = ['POST', 'GET'])
def ticker():
    username=session.get('uname')
    return render_template('ticker.html',firstName=username)

@app.route("/about", methods = ['POST', 'GET'])
def about():
   
    return render_template('about.html')

# @app.route("/register", methods = ['POST', 'GET'])
# def register():
    
#     return render_template('register.html')



@app.route("/logout", methods = ['POST', 'GET'])
def logout():
    session.pop('uname',None)
    session.pop('userid',None)
    return redirect(url_for('login'))

####################################################################CHATBOT############################################################
@app.route('/getRequest1', methods=['POST'])
def getRequest1():
    print("GET")
    if request.method =='POST':
        print("Post")
     
        data = request.get_json()
        userText = data.get('userMessage')
   
        print(userText)
        data = json.dumps({"sender": "Rasa","message": userText})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        res = requests.post('http://localhost:5005/webhooks/rest/webhook', data= data, headers = headers)
        res = res.json()
        print()
        print("Output")
        print(res)
        print()
        val = res[0]['text']
        output.append(val)
        print()
        print("val")
        print(val)
        print()
        responses = val
        print("=====================")
        print(responses)
        print("=====================")
        
        return responses
  


if __name__=='__main__':
    # app.run(debug=True)
    app.run('0.0.0.0')