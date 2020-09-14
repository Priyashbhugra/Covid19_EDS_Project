import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from Visualize_SIR_modeling import SIR_modelling

import plotly.graph_objects as go
from scipy import optimize
from scipy import integrate

import os
print(os.getcwd())
df_analyse = pd.read_csv('data/processed/COVID_final_set.csv', sep = ';')
df_population = pd.read_csv('data/processed/population.csv', sep = ';')
fig = go.Figure()
app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''

    # Implmentation of SIR model for Multiple countries

    '''),
    dcc.Dropdown(
        id = 'country_list',
        options=[ {'label': each,'value':each} for each in df_analyse['country'].unique()],
        value= 'Germany', # which are pre-selected
        multi=False),
    dcc.Graph(figure = fig, id = 'SIR_graph')
    ])

def SIR(countries):
    SIR_modelling()

@app.callback(
    Output('SIR_graph', 'figure'),
    [Input('country_list', 'value')])

def SIR_figure(country_list):
    traces = []
    population = df_population[df_population['COUNTRY'] == country_list]['Value'].values[0]
    df_plot = df_analyse[df_analyse['country'] == country_list]
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
                title = 'SIR model fitting for '+country_list+', Population = '+str(population)+", Optimal Beta = "+str("{:.2f}".format(popt[0]))+" and Gamma = "+str("{:.2f}".format(popt[1])),

                xaxis= {'title':'Number of days since Pandemic starts',
                       'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#080807"),
                      },

                yaxis={'title': "Number of Infected People"}

        )
    }


if __name__ == '__main__':
    app.run_server(debug = True, use_reloader = False,port = 9010)