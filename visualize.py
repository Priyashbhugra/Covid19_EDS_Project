import pandas as pd
import numpy as np
import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from src.data.get_data import get_johns_hopkins
from src.data.process_JH_data import store_relational_JH_data
from src.features.build_features import Gen_feature
import plotly.graph_objects as go
from Visualize_SIR_modeling import SIR_modelling


import os
# Get Data from  Get Johns Hopkins
get_johns_hopkins()
# Store relational Data
store_relational_JH_data()
# Generate features
Gen_feature()

print(os.getcwd())


df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')
df_analyse = pd.read_csv('data/processed/COVID_final_set.csv', sep = ';')
df_population = pd.read_csv('data/processed/population.csv', sep = ';')

fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Project of Enterprise Data Science(EDS) on COVID-19 Dataset of John Hopkins University

    This Dashboard helps to visualize the data of COVID-19 pandemic for multiple countries with respect to approximated timeline.

    '''),
    dcc.Tabs(id='main_tab', value='main_tab', children=[
        dcc.Tab(id='tab1', label='Visualization of COVID19 data', value='tab1', children=[

            dcc.Markdown('''
    ## Select or Enter multiple country for visualization
    '''),


    dcc.Dropdown(
        id='country_list',
        options=[ {'label': country,'value':country} for country in df_input_large['country'].unique()],
        value=['Germany','India','France'], # pre-selected Countries
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
]),
    dcc.Tab(id='tab2', label='SIR Model', value='tab2', children=[
dcc.Markdown('''

    # Implmentation of SIR model for Multiple countries

    '''),
    dcc.Dropdown(
        id = 'country_list_dropdown',
        options=[ {'label': each,'value':each} for each in df_analyse['country'].unique()],
        value= 'Germany', # which are pre-selected
        multi=False),
    dcc.Graph(figure = fig, id = 'SIR_graph')
    ])
])
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_list', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):


    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    for country in country_list:

        df_plot=df_input_large[df_input_large['country']==country]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=country
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }
@app.callback(
    Output('SIR_graph', 'figure'),
    [Input('country_list_dropdown', 'value')])

def SIR_figure(country_list_dropdown):
    traces = []
    population = df_population[df_population['COUNTRY'] == country_list_dropdown]['Value'].values[0]
    df_plot = df_analyse[df_analyse['country'] == country_list_dropdown]
    df_plot = df_plot[['state', 'country', 'confirmed', 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
    df_plot.sort_values('date', ascending = True).head()
    df_plot = df_plot.confirmed[35:]
    t, fitted,popt = SIR_modelling(df_plot,population)

    traces.append(dict (x = t,
                        y = fitted,
                        mode = 'markers',
                        opacity = 0.9,
                        name = 'SIR-fit curve')
                  )

    traces.append(dict (x = t,
                        y = df_plot,
                        mode = 'lines',
                        opacity = 0.9,
                        name = 'Actual Data')
                  )


    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,
                title = 'SIR model fitting for '+country_list_dropdown+', Population = '+str(population)+", Optimal Beta = "+str("{:.2f}".format(popt[0]))+" and Gamma = "+str("{:.2f}".format(popt[1])),

                xaxis= {'title':'Number of days since Pandemic starts',
                       'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#080807"),
                      },

                yaxis={'title': "Number of Infected People"}

        )
    }

if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False,port = 9050)
