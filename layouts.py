# ML Library
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

# Machine Learning

app_train = pd.read_csv("./resources/application_train.csv", encoding='utf-8', sep=',')
app_test = pd.read_csv("./resources/application_test.csv", encoding='utf-8', sep=',')
app_train.drop_duplicates()
app_test.drop_duplicates()

# Aligning datasets
train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join='inner', axis=1)
app_train['TARGET'] = train_labels
alignment = 'Application train:\n\n' + 'Number of rows: ' + str(app_train.shape[0]) + '\n' + 'Number of columns: ' \
            + str(app_train.shape[1]) + '\n\n' + 'Application test:\n' + 'Number of rows: ' + str(app_test.shape[0]) \
            + '\n' + 'Number of columns: ' + str(app_test.shape[1])

# Unbalanced Data
unbalanced_train_fig = px.histogram(app_train,
                                    x="TARGET",
                                    color="TARGET",
                                    title='Unbalanced Data',
                                    height=700,
                                    width=1000
                                    )
unbalanced_count = 'Target distribution before:\n0: ' + str(app_train['TARGET'].value_counts()[0]) + '\n' \
                   + '1: ' + str(app_train['TARGET'].value_counts()[1])

# Oversampling
msk = app_train['TARGET'] == 1
num_to_oversample = len(app_train) - 2 * msk.sum()
df_positive_oversample = app_train[msk].sample(n=num_to_oversample, replace=True)
df_train_oversample = pd.concat([app_train, df_positive_oversample])
app_train = df_train_oversample

# Balanced Data
balanced_train_fig = px.histogram(app_train,
                                  x="TARGET",
                                  color="TARGET",
                                  title='Balanced Data',
                                  height=700,
                                  width=1000
                                  )
balanced_count = 'Target distribution after:\n0: ' + str(app_train['TARGET'].value_counts()[0]) + '\n' \
                 + '1: ' + str(app_train['TARGET'].value_counts()[1])
compare_balance = unbalanced_count + '\n' + balanced_count

# Gender Distribution
gender_group = app_train.groupby(['CODE_GENDER']).size().reset_index(name='count')
gender_fig = px.pie(gender_group,
                    values='count',
                    names='CODE_GENDER',
                    title='Gender Distribution',
                    height=700,
                    width=1000
                    )

# Contract Type Distribution
contract_group = app_train.groupby(['NAME_CONTRACT_TYPE']).size().reset_index(name='count')
contract_fig = px.pie(contract_group, values='count', names='NAME_CONTRACT_TYPE', title='Contract Type Distribution')

# DAYS BIRTH feature
# TODO
#  (app_train['DAYS_BIRTH'] / -365).describe()

# Boxplot
mini = abs(app_train['DAYS_BIRTH'].max())
if mini > 365:
    birth_mini = "Days birth min: " + str(mini / 365) + " years\n"
else:
    "Days birth min: " + str(mini) + " days\n"

maxi = abs(app_train['DAYS_BIRTH'].min())
birth_maxi = "Days birth max: " + str(maxi / 365) + " years\n"
age_minmax = birth_mini + '\n' + birth_maxi


daysbirth_boxplot = px.box(app_train,
                           y='DAYS_BIRTH',
                           title='Boxplot of Days Birth',
                           height=700,
                           width=1000
                           )

# Label encoding
le = LabelEncoder()
le_count = 0

for col in app_train:
    if app_train[col].dtype == 'object':
        le.fit(app_train[col])
        app_train[col] = le.transform(app_train[col])
        app_test[col] = le.transform(app_test[col])
        le_count += 1
app_train.reset_index()
app_test.reset_index()

label_encoding_str = '%d columns were label encoded.' % le_count

# Replacing Infinite values with NaN values
app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
app_test.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_train)
imputer = imputer.fit_transform(app_train)
app_train = pd.DataFrame(imputer, columns=app_train.columns.values.tolist())
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_test)
imputer = imputer.fit_transform(app_test)
app_test = pd.DataFrame(imputer, columns=app_test.columns.values.tolist())

# Days Employed feature

# TODO
#  (app_train['DAYS_EMPLOYED'] / -365).describe()

# Min - Max
mini = abs(app_train['DAYS_EMPLOYED'].max())
if mini > 365:
    employ_min = "Days employed min :" + str(mini / 365) + "years"
else:
    employ_min = "Days employed min :" + str(mini) + "days"

maxi = abs(app_train['DAYS_EMPLOYED'].min())
employ_max = "Days employed max :" + str(maxi / 365) + "years"
employ_minmax = employ_min + '\n' + employ_max

# Boxplot before cleanup
employ_boxplot_bf = px.box(app_train,
                           y='DAYS_EMPLOYED',
                           title='Boxplot of Days Employed before cleaning',
                           height=700,
                           width=1000
                           )
# Removing anomalies
app_train.drop(app_train.index[(app_train["DAYS_EMPLOYED"] > 12000)], axis=0, inplace=True)
app_test.drop(app_test.index[(app_test["DAYS_EMPLOYED"] > 12000)], axis=0, inplace=True)

# Boxplot after cleanup
employ_boxplot_af = px.box(app_train,
                           y='DAYS_EMPLOYED',
                           title='Boxplot of Days Employed after cleaning',
                           height=700,
                           width=1000
                           )

# Menu SideBar

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
    html.H2('Dataframe before cleanup'),
    html.Img(src='./assets/train_dtable.jpg'),
    html.Br(),
    html.Div('We align the train and test dataframes so that they have the same columns. '
             'This way, our models will learn and predict targets with the same features.'),
    html.P(alignment),
    html.Br(),

    html.Div(className='container',
             children=[
                 html.P('Unbalanced data figure :'),
                 dcc.Graph(figure=unbalanced_train_fig),
             ], style={'textAlign': 'center'}),

    html.Div('* We oversample the dataframe to get an even amount of each target value.'),
    html.P(compare_balance),
    html.Br(),

    html.Div(className='container',
             children=[
                 html.P('Balanced data figure :'),
                 dcc.Graph(figure=balanced_train_fig),
             ]),

    # Gender Distribution PieChart
    html.Div(className='container',
             children=[
                 html.P('Gender Distribution piechart :'),
                 dcc.Graph(figure=gender_fig),
             ]),

    # Contract Type Distribution PieChart
    html.Div(className='container',
             children=[
                 html.P('Contract Type Distribution piechart :'),
                 dcc.Graph(figure=contract_fig),
             ]),

    # Days Birth Feature
    html.Div(className='container',
             children=[
                 html.P('Days Birth Boxplot:'),
                 dcc.Graph(figure=daysbirth_boxplot),
             ]),

    html.P(age_minmax),


    # DAYS EMPLOYED Feature
    html.P(employ_minmax),

    html.Div(className='container',
             children=[
                 html.P('Days Employ Boxplot before cleanup:'),
                 dcc.Graph(figure=employ_boxplot_bf),
             ]),

    html.P('We remove anomalies from the days Employed feature.'),

    html.Div(className='container',
             children=[
                 html.P('Days Employ Boxplot after cleanup:'),
                 dcc.Graph(figure=employ_boxplot_af),
             ]),

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
