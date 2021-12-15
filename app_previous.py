# ML Library
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import graphviz

# Machine Learning Library
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Decision Tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# XGBoost
import xgboost as xgb

from warnings import filterwarnings
filterwarnings('ignore')

# Dash Deploy
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import flask

# Machine Learning

app_train = pd.read_csv("./resources/application_train.csv", encoding='utf-8', sep=',')
app_test = pd.read_csv("./resources/application_test.csv", encoding='utf-8', sep=',')
app_train.drop_duplicates()
app_test.drop_duplicates()

# Aligning datasets
train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join='inner', axis=1)
app_train['TARGET'] = train_labels
train_row_num = 'Number of rows: ' + str(app_train.shape[0])
train_col_num = 'Number of columns: ' + str(app_train.shape[1])
test_row_num = 'Number of rows: ' + str(app_test.shape[0])
test_col_num = 'Number of columns: ' + str(app_test.shape[1])

# Unbalanced Data
unbalanced_train_fig = sns.countplot(x="TARGET", data=app_train).set(title='Balance of Target')
unbalanced_count = 'Target distribution before:\n0: ' + str(app_train['TARGET'].value_counts()[0])+'\n' \
                       + '1: ' + str(app_train['TARGET'].value_counts()[1])
# Oversampling
msk = app_train['TARGET'] == 1
num_to_oversample = len(app_train) - 2*msk.sum()
df_positive_oversample = app_train[msk].sample(n=num_to_oversample, replace=True)
df_train_oversample = pd.concat([app_train, df_positive_oversample])
app_train = df_train_oversample

# Balanced Data
balanced_train_fig = sns.countplot(x="TARGET", data=app_train).set(title='Balance of Target')
balanced_count = 'Target distribution after:\n0: ' + str(app_train['TARGET'].value_counts()[0])+'\n' \
                       + '1: ' + str(app_train['TARGET'].value_counts()[1])



















# DASH

app = dash.Dash(__name__)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": "#202020",
    "color": '#B8B8B8'
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                html.Br(),
                dbc.NavLink("Home", href="/", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Data Study & Cleanup", href="/page-0", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Logistic Regression", href="/page-1", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Decision Tree", href="/page-2", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Random Forest", href="/page-3", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("XGBoost", href="/page-4", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Model testing on application test", href="/page-5", active="exact"),
                html.Br(),
                html.Br(),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Br(),
        html.Img(src='./assets/bank.png')
    ],
    style=SIDEBAR_STYLE,
)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    sidebar,
])

# Home

layout_index = html.Div([
    # Title & Description
    html.H1('Home Credit Default Risk'),
    html.Hr(),
    html.Div('We have a dataframe in which are registered a lot of information about a bank\'s clients.'),
    html.Br(),
    html.Div('The bank wants to know if they should give them a loan or not. The machine learning '
             'model needs to predict whether or not the client will repay the loan.'),
    html.Br(),
    html.Div('If the target is equal to 0: the loan was repaid, if it\'s equal to 1: the loan was not repaid.'),
    html.Br(),
    html.Div('Thanks to the given features, the models will learn how to predict the target.'''),
    html.Br(),

    # Summary
    html.H2('Summary'),
    dcc.Link('Data Study & cleanup', href='/page-0'),
    html.Br(),
    dcc.Link('Regression Logistic', href='/page-1'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('Random Forest', href='/page-3'),
    html.Br(),
    dcc.Link('XGBoost', href='/page-4'),
    html.Br(),
    dcc.Link('Model testing on Application Test', href='/page-5'),
    html.Br(),

    # Conclusion
    html.H3('Conclusion'),
    html.Div('The data given was very unbalanced, we had to use oversampling to balance it to get accurate models.'),
    html.Br(),
    html.Div('We can see that the least efficient model was the logistic regression.'),
    html.Br(),
    html.Div('As the bank, if the model predicts too many true positives, this is not an issue.'),
    html.Br(),
    html.Div('The bank would have not given the loan to someone who could have repaid it. '
             'This isn\'t so great for the clients.'),
    html.Br(),
    html.Img(src='./assets/bank1.png'),
])

# Data Study & Cleanup

layout_page_0 = html.Div([
    html.H1('Data Study & Cleanup'),
    html.Hr(),
    html.Br(),
    # Data before cleanup
    html.H2(children='''Dataframe before cleanup''',
            ),
    html.Img(src='./assets/train_dtable.jpg'),
    html.Br(),
    html.Div('We align the train and test dataframes so that they have the same columns. '
             'This way, our models will learn and predict targets with the same features.'),
    html.Div('Application Train:'),
    html.Br(),
    html.Div(train_row_num),
    html.Div(train_col_num),
    html.Br(),
    html.Div('Application Test:'),
    html.Br(),
    html.Div(test_row_num),
    html.Div(test_col_num),
    html.Br(),

    # Unbalanced data

    dcc.Graph(
        id='unbalanced_train_fig',
        figure=unbalanced_train_fig
    ),

    html.Div('* We oversample the dataframe to get an even amount of each target value.'),
    html.Br(),
    html.Div(unbalanced_count),
    html.Br(),
    html.Div(balanced_count),
    html.Br(),

])

# Logistic Regression

layout_page_1 = html.Div([
    html.H1('Logistic Regression'),
    html.Hr(),
    html.Br(),

])

# Decision Tree

layout_page_2 = html.Div([
    html.H1('Decision Tree'),
    html.Hr(),
    html.Br(),

])

# Random Forest

layout_page_3 = html.Div([
    html.H1('Random Forest'),
    html.Hr(),
    html.Br(),

])

# XGBoost

layout_page_4 = html.Div([
    html.H1('XGBoost'),
    html.Hr(),
    html.Br(),

])


layout_page_5 = html.Div([
    html.H1('Model testing on application test'),
    html.Hr(),
    html.Br(),

])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_0,
    layout_page_1,
    layout_page_2,
    layout_page_3,
    layout_page_4,
    layout_page_5,
])


# Index callbacks
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/page-0":
        return layout_page_0
    elif pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    elif pathname == "/page-4":
        return layout_page_4
    elif pathname == "/page-5":
        return layout_page_5
    else:
        return layout_index


if __name__ == '__main__':
    app.run_server(debug=True)
