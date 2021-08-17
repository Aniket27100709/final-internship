import datetime as DT
from datetime import timedelta
import pandas as pd
from flask import Flask, render_template,request,session,redirect,url_for,g

from sklearn import metrics
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import os

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kaleido
import seaborn as sns

import math
import random
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

#color pallette
cnf = '#393e46'
dth = '#ff2e63'
rec = '#21bf73'
act = '#fe9801'

#from fbprophet import Prophet
import numpy as np
plt.rcParams['figure.figsize']=10,12
#from fbprophet.plot import plot_plotly
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go








app = Flask(__name__)
app.secret_key="coviddataanalysis"


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template("home.html")




@app.route('/Spread-of-covid-cases')
def spread_india():

    flag=os. path. exists("static/fig1.png")
    if ~flag:
        df = pd.read_csv('Cases_in_India.csv', parse_dates=['Date'])
        confirmed = df.groupby('Date').sum()['Total Confirmed'].reset_index()
        recovered = df.groupby('Date').sum()['Total Recovered'].reset_index()
        deaths=df.groupby("Date").sum()["Total Deceased"].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = confirmed['Date'], y = confirmed['Total Confirmed'], mode = 'lines+markers', name = 'Confirmed', line = dict(color = "Orange", width = 2)))
        fig.add_trace(go.Scatter(x = recovered['Date'], y = recovered['Total Recovered'], mode = 'lines+markers', name = 'Recovered', line = dict(color = "Green", width = 2)))
        fig.add_trace(go.Scatter(x = deaths['Date'], y = deaths['Total Deceased'], mode = 'lines+markers', name = 'Deaths', line = dict(color = "Red", width = 2)))
        fig.update_layout(title = 'Spread of Coronavirus in India', xaxis_tickfont_size = 14, yaxis = dict(title = 'Number of Cases'))
        fig.write_image("static/fig1.png")

        #img1 end    
        
        df['Date'] = df['Date'].astype(str)
        temp = df.groupby('Date')['Total Confirmed', 'Total Deceased', 'Total Recovered'].sum().reset_index()
        temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop = True)

        tm = temp.melt(id_vars = 'Date', value_vars = ['Total Confirmed', 'Total Deceased', 'Total Recovered'])
        fig = px.treemap(tm, path = ['variable'], values = 'value', height = 250, width = 800, color_discrete_sequence=[act, rec, dth])

        fig.data[0].textinfo = 'label+text+value'
        fig.write_image("static/fig2.png")

        #img 2 end

        data = df
        fig3=sns.relplot(x='Total Deceased', y='Total Confirmed', hue='Total Recovered', data=df)
        plt.savefig("static/fig3.png")

        #img 3 end

        fig4=sns.catplot(x='Total Confirmed', kind='box', data=df)
        plt.savefig("static/fig4.png")


        #img 4 end

        sns.catplot(x='Total Deceased', kind='box', data=df)
        plt.savefig("static/fig5.png")

        #img 5 end

        sns.displot(data['Total Confirmed'], bins=10)
        plt.savefig("static/fig6.png")

        #img 6 end

        sns.displot(data['Total Deceased'], bins=10)
        plt.savefig("static/fig7.png")

        #img 7 end

        corelation = data.corr()
        sns.pairplot(data)
        plt.savefig("static/fig8.png")

        #img 8 end

    return render_template("spread.html")


















@app.route('/spread-india-vs-other-countries')
def spread_india_vs_other_countries():
    df = pd.read_csv('covid_19_data_cleaned.csv', parse_dates=['Date'])
    country_daywise = pd.read_csv('country_daywise.csv', parse_dates=['Date'])
    countywise = pd.read_csv('countrywise.csv')
    daywise = pd.read_csv('daywise.csv', parse_dates=['Date'])
    df['Province/State'] = df['Province/State'].fillna("")
    confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
    recovered = df.groupby('Date').sum()['Recovered'].reset_index()
    deaths = df.groupby('Date').sum()['Deaths'].reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = confirmed['Date'], y = confirmed['Confirmed'], mode = 'lines+markers', name = 'Confirmed', line = dict(color = "Orange", width = 2)))
    fig.add_trace(go.Scatter(x = recovered['Date'], y = recovered['Recovered'], mode = 'lines+markers', name = 'Recovered', line = dict(color = "Green", width = 2)))
    fig.add_trace(go.Scatter(x = deaths['Date'], y = deaths['Deaths'], mode = 'lines+markers', name = 'Deaths', line = dict(color = "Red", width = 2)))
    fig.update_layout(title = 'Worldwide Covid-19 Cases', xaxis_tickfont_size = 14, yaxis = dict(title = 'Number of Cases'))

    fig.write_image("static/fig9.png")

    #img 9 end
    
    df['Date'] = df['Date'].astype(str)
    fig = px.density_mapbox(df, lat = 'Lat', lon = 'Long', hover_name = 'Country', hover_data = ['Confirmed', 'Recovered', 'Deaths'], animation_frame='Date', color_continuous_scale='Portland', radius = 7, zoom = 0, height = 700)
    fig.update_layout(title = 'Worldwide Covid-19 Cases with Time Laps')
    fig.update_layout(mapbox_style = 'open-street-map', mapbox_center_lon = 0)
    fig.write_image("static/fig10.png")

    #img 10 end

    temp = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
    temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop = True)

    tm = temp.melt(id_vars = 'Date', value_vars = ['Active', 'Deaths', 'Recovered'])
    fig = px.treemap(tm, path = ['variable'], values = 'value', height = 250, width = 800, color_discrete_sequence=[act, rec, dth])

    fig.data[0].textinfo = 'label+text+value'
    fig.write_image("static/fig11.png")

    #img 11 end

    fig = px.choropleth(country_daywise, locations= 'Country', locationmode='country names', color = np.log(country_daywise['Confirmed']),
    hover_name = 'Country', animation_frame = country_daywise['Date'].dt.strftime('%y-%m-%d'),
    title='Cases over time', color_continuous_scale=px.colors.sequential.Inferno)

    fig.update(layout_coloraxis_showscale = True)
    fig.write_image("static/fig12.png")

    #img 12 end

    top = 15

    fig_c = px.bar(countywise.sort_values('Confirmed').tail(top), x = 'Confirmed', y = 'Country', 
                  text = 'Confirmed', orientation = 'h', color_discrete_sequence = [act] )
    fig_d = px.bar(countywise.sort_values('Deaths').tail(top), x = 'Deaths', y = 'Country', 
                  text = 'Deaths', orientation = 'h', color_discrete_sequence = [dth] )
                   
    fig_a = px.bar(countywise.sort_values('Active').tail(top), x = 'Active', y = 'Country', 
                  text = 'Active', orientation = 'h', color_discrete_sequence = ['#434343'] )
    fig_r = px.bar(countywise.sort_values('Recovered').tail(top), x = 'Recovered', y = 'Country', 
                  text = 'Recovered', orientation = 'h', color_discrete_sequence = [rec] )               



    fig = make_subplots(rows = 5, cols = 2, shared_xaxes=False, horizontal_spacing=0.14,
                       vertical_spacing=0.1, 
                       subplot_titles=('Confirmed Cases', 'Deaths Reported'))
                        
    fig.add_trace(fig_c['data'][0], row = 1, col = 1)
    fig.add_trace(fig_d['data'][0], row = 1, col = 2)

    fig.add_trace(fig_r['data'][0], row = 2, col = 1)
    fig.add_trace(fig_a['data'][0], row = 2, col = 2)
                        


    fig.update_layout(height = 3000)
    fig.write_image("static/fig13.png")

    # img 13 end

    fig = px.bar(country_daywise, x = 'Date', y = 'Confirmed', color = 'Country', height = 600,
            title='Confirmed', color_discrete_sequence=px.colors.cyclical.mygbm)

    fig.write_image("static/fig14.png")

    #img 14 end
    return render_template("india_vs_countries.html")






@app.route('/age-wise')
def age_wise_corona():
    data = pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv")
    data_1= pd.read_csv("Book3.csv")
    m=data.groupby('State')['Active'].sum().sort_values(ascending=False)
    data.groupby('State')['Active'].sum().sort_values(ascending=False)
    data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)
    d=data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)
    d.plot.bar(figsize=(15,5))
    plt.savefig("static/fig15.png")
    #image 15 end

    
    p=data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)/403312*100
    p.plot.bar(figsize=(15,5))
    plt.savefig("static/fig16.png")
    #image 16 end
    
    sns.scatterplot(x="State" , y="Active" ,data=data)
    sns.scatterplot(data=data_1, x="Age Group", y="No. Cases", hue="No. Deaths",palette= "deep")
    l=data_1.drop([11])
    sns.scatterplot(data=l, x="Age Group", y="No. Cases", hue="No. Deaths",size = "No. Deaths",sizes=(100,150))
    age_wise=data_1.groupby('Age Group')['No. Cases'].sum().drop('Total').sort_values(ascending=False)
    age_wise.plot.bar(figsize=(15,5))
    plt.savefig("static/fig17.png")
    #image 17 end
    return render_template("age_wise.html")





@app.route('/state-wise')
def state_wise():
    data = pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv")
    data_1= pd.read_csv("Book3.csv")
    m=data.groupby('State')['Active'].sum().sort_values(ascending=False)
    data.groupby('State')['Active'].sum().sort_values(ascending=False)
    m.plot.bar(figsize=(25,25))
    plt.savefig("static/fig18.png")
    #imge 18 end

    data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)
    d=data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)

    p=data.groupby('State')['Active'].sum().drop('Total').drop('State Unassigned').sort_values(ascending=False)/403312*100
    
    p.plot.bar(figsize=(200,50))
    plt.savefig("static/fig19.png")
    #imge 19 end

    plt.figure(figsize=(100,15))
    sns.scatterplot(x="State" , y="Active" ,data=data)
    plt.savefig("static/fig20.png")
    #imge 20 end

    plt.figure(figsize=(12,5))
    sns.scatterplot(data=data_1, x="Age Group", y="No. Cases", hue="No. Deaths",palette= "deep")
    plt.savefig("static/fig21.png")
    #imge 21 end

    l=data_1.drop([11])
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=l, x="Age Group", y="No. Cases", hue="No. Deaths",size = "No. Deaths",sizes=(100,150))
    plt.savefig("static/fig22.png")
    #imge 22 end

    age_wise=data_1.groupby('Age Group')['No. Cases'].sum().drop('Total').sort_values(ascending=False)
    age_wise.plot.bar(figsize=(15,5))
    plt.savefig("static/fig23.png")
    #imge 23 end

    sns.pairplot(data=data)
    plt.savefig("static/fig24.png")
    #imge 24 end


    return render_template("state_wise.html")




@app.route('/future-prediction')
def future_prediction():
    from fbprophet import Prophet
    data=pd.read_csv("states8.csv")
    data['Confirmed']=data['Confirmed'].astype(float)
    data.columns=['ds','y']
    data['ds']=pd.to_datetime(data['ds'])
    model=Prophet()
    data.columns
    model.fit(data)
    future_dates=model.make_future_dataframe(periods=365)
    prediction=model.predict(future_dates)
    prediction['trend']=prediction['trend'].astype(int)
    # predicted projection
    model.plot(prediction)

    plt.savefig("static/fig25.png")
    #imge 25 end


    model.plot_components(prediction)
    plt.savefig("static/fig26.png")
    #imge 26 end
    
    return render_template("future_prediction.html")


@app.route('/symptoms')
def symptoms_for_covid():
    symptom=pd.read_csv("Symptoms.csv")
    data = {'Fever':80, 'Dry Cough':70, 'Fatigue':40,
        'Sputum Production':30,'Shortness of Breath':20,'Muscle pain':15,'Sore throat':15,'Headache':15,'Chills':15,'Nausea or Vomiting':10,'Nasal congestion':10,'Diarrhoea':5,'Hemoptysis':5,'Conjunctival congestion':5}
    symp=list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (30, 45))
    plt.bar(symp, values, color ='navy',width = 0.4)
    plt.xlabel("Symptoms")
    plt.ylabel("Percentage")
    plt.title("Covid patients suffering from symptoms in (%)")

    plt.savefig("static/fig27.png")
    #imge 27 end
    
    return render_template("symptoms.html")
    



if __name__ == "__main__":
    app.run(debug=True)



