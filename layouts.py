# ML Library
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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

from dash import dcc, dash_table
from dash import html
import dash_bootstrap_components as dbc

# Colors dictionary for html
colors = {
    'background': '#111111',
    'title': '#7FDBFF',
    'text': '#a6a6a6',
    'blue': '#000067'
}
# Machine Learning

app_train = pd.read_csv("./resources/application_train.csv", encoding='utf-8', sep=',')
app_test = pd.read_csv("./resources/application_test.csv", encoding='utf-8', sep=',')
app_train.drop_duplicates()
app_test.drop_duplicates()
columns = app_train.iloc[:, :10]
print("read csv\n")

# Aligning datasets
train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join='inner', axis=1)
app_train['TARGET'] = train_labels
alignment = 'Application train:\n\n' + 'Number of rows: ' + str(app_train.shape[0]) + '\n' + 'Number of columns: ' \
            + str(app_train.shape[1]) + '\n\n' + 'Application test:\n' + 'Number of rows: ' + str(app_test.shape[0]) \
            + '\n' + 'Number of columns: ' + str(app_test.shape[1])
print("aligning dataset\n")

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
print("unbalanced fig\n")

# Oversampling
msk = app_train['TARGET'] == 1
num_to_oversample = len(app_train) - 2 * msk.sum()
df_positive_oversample = app_train[msk].sample(n=num_to_oversample, replace=True)
df_train_oversample = pd.concat([app_train, df_positive_oversample])
app_train = df_train_oversample
print("oversampling\n")

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
print("balanced fig\n")

# Gender Distribution
gender_group = app_train.groupby(['CODE_GENDER']).size().reset_index(name='count')
gender_fig = px.pie(gender_group,
                    values='count',
                    names='CODE_GENDER',
                    title='Gender Distribution',
                    height=700,
                    width=1000
                    )
print("gender distribution\n")

# Contract Type Distribution
contract_group = app_train.groupby(['NAME_CONTRACT_TYPE']).size().reset_index(name='count')
contract_fig = px.pie(contract_group, values='count', names='NAME_CONTRACT_TYPE', title='Contract Type Distribution')
print("contract distribution\n")

# DAYS BIRTH feature
# TODO
#  (app_train['DAYS_BIRTH'] / -365).describe()

# Min - Max
mini = abs(app_train['DAYS_BIRTH'].max())
if mini > 365:
    birth_mini = "Days birth min: " + str(mini / 365) + " years\n"
else:
    "Days birth min: " + str(mini) + " days\n"

maxi = abs(app_train['DAYS_BIRTH'].min())
birth_maxi = "Days birth max: " + str(maxi / 365) + " years\n"
age_minmax = birth_mini + '\n' + birth_maxi
print("birth minmax\n")

# Boxplot
daysbirth_boxplot = px.box(app_train,
                           y='DAYS_BIRTH',
                           title='Boxplot of Days Birth',
                           height=700,
                           width=1000
                           )
print("birth boxplot\n")

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
print("label encoding\n")

# Replacing Infinite values with NaN values
app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
app_test.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_train)
imputer = imputer.fit_transform(app_train)
app_train = pd.DataFrame(imputer, columns=app_train.columns.values.tolist())
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_test)
imputer = imputer.fit_transform(app_test)
app_test = pd.DataFrame(imputer, columns=app_test.columns.values.tolist())
print("infinite and nan values\n")

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
print("employed minmax\n")

# Removing anomalies
app_train.drop(app_train.index[(app_train["DAYS_EMPLOYED"] > 12000)], axis=0, inplace=True)
app_test.drop(app_test.index[(app_test["DAYS_EMPLOYED"] > 12000)], axis=0, inplace=True)
print("anomalies removed\n")

# Boxplot after cleanup
employ_boxplot_af = px.box(app_train,
                           y='DAYS_EMPLOYED',
                           title='Boxplot of Days Employed after cleaning',
                           height=700,
                           width=1000
                           )
print("employ boxplot\n")

# Splitting data into train / test
Xdf = app_train
Xdf.drop("TARGET", axis=1)
X = np.array(Xdf)
y = np.array(app_train["TARGET"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
# real = "Real values:\n\n" + y_test
print("split data into test/train\n")

# Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
# TODO
#  LR_predictions = "Predictions:\n\n" + y_pred + '\n'
#  LR_results = LR_predictions + '\n' + real
print("LR predict\n")

# LR Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
x = ['0', '1']
y = ['1', '0']
conf_value = [[str(y) for y in x] for x in conf_matrix]
LR_conf_fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')

LR_conf_str = 'The confusion matrix shows us the number of :\n' + \
              '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
              '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
              '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
              '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
print("LR confusion matrix\n")

# Model score
LR_accu = "Accuracy score:" + str(round((accuracy_score(y_test, y_pred) * 100), 2)) + '%\n' + \
          "\nAccuracy score using cross validation:" + \
          str(round((cross_val_score(LR, X_train, y_train, cv=3, scoring='accuracy').mean()) * 100, 2)) + '%\n'
print("LR accuracy score\n")

LR_precis = "Precision score:" + str(round((precision_score(y_test, y_pred, average='macro') * 100), 2)) + '%\n'
print("LR precision score\n")

LR_recall = "Recall score:" + str(round((metrics.recall_score(y_test, y_pred) * 100), 2)) + '%\n'
print("LR recall score\n")

LR_F1 = "F1 Score:", str(round((metrics.f1_score(y_test, y_pred) * 100), 2)) + '%\n'
print("LR F1 score\n")

# ROC CURVE
# TODO
#  prediction_prob = LR.predict_proba(X_test)[::, 1]
#  fpr, tpr, _ = metrics.roc_curve(y_test, prediction_prob)
#  auc = metrics.roc_auc_score(y_test, prediction_prob)
#  plt.title("Receiver Operating Characteristic curve")
#  plt.plot(fpr, tpr, label="AUC=" + str(auc))
#  plt.legend(loc=4)
#  plt.show()

# Testing on app-test
app_test_LR = app_test.copy()
app_test_LR['TARGET'] = 0
y_pred_test = LR.predict(app_test_LR)
app_test_LR['TARGET'] = y_pred_test.astype(int)
# TODO print
print("LR app-test\n")

# Decision Tree
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
# TODO
#  DT_predictions = "Predictions:\n\n" + str(y_pred) + '\n'
#  DT_results = DT_predictions + '\n' + real

# DT Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
x = ['0', '1']
y = ['1', '0']
conf_value = [[str(y) for y in x] for x in conf_matrix]
DT_conf_fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')

DT_conf_str = 'The confusion matrix shows us the number of :\n' + \
              '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
              '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
              '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
              '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
print("DT confusion matrix\n")

# Model score
DT_accu = "Accuracy score:" + str(round((accuracy_score(y_test, y_pred) * 100), 2)) + '%\n' + \
          "\nAccuracy score using cross validation:" + \
          str(round((cross_val_score(DT, X_train, y_train, cv=3, scoring='accuracy').mean()) * 100, 2)) + '%\n'
print("DT accuracy score\n")

# Testing on app-test
app_test_DT = app_test.copy()
app_test_DT['TARGET'] = 0
y_pred_test = DT.predict(app_test_DT)
app_test_DT['TARGET'] = y_pred_test.astype(int)
# TODO print
print('DT app-test\n')

# Random Forest

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
# TODO
#  RF_predictions = "Predictions:\n\n" + y_pred + '\n'
#  RF_results = RF_predictions + '\n' + real
print("RF predict\n")

# RF Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
x = ['0', '1']
y = ['1', '0']
conf_value = [[str(y) for y in x] for x in conf_matrix]
RF_conf_fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')
RF_conf_str = 'The confusion matrix shows us the number of :\n' + \
              '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
              '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
              '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
              '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
print("RF confusion matrix\n")

# Model score
RF_accu = "Accuracy score:" + str(round((accuracy_score(y_test, y_pred) * 100), 2)) + '%\n' + \
          "\nAccuracy score using cross validation:" + \
          str(round((cross_val_score(RF, X_train, y_train, cv=3, scoring='accuracy').mean()) * 100, 2)) + '%\n'
print("RF accuracy score\n")

# Testing on app-test
app_test_RF = app_test.copy()
app_test_RF['TARGET'] = 0
y_pred_test = RF.predict(app_test_RF)
app_test_RF['TARGET'] = y_pred_test.astype(int)
print("RF tapp-test\n")
# TODO print

# XGBoost
# TODO

# Comparing models on app-test
LR_target = app_test_LR['TARGET']
DT_target = app_test_DT['TARGET']
RF_target = app_test_RF['TARGET']

if LR_target.equals(DT_target):
    if LR_target.equals(RF_target):
        comparison = "All three models found the same target values on application test.\n"
else:
    comparison = "All three models did not find the same target values on application test.\n"

######################################################

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
                dbc.NavLink("Data Study & Cleanup", href="/data", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Logistic Regression", href="/logistic-regression", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Decision Tree", href="/decision-tree", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Random Forest", href="/random-forest", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("XGBoost", href="/xgboost", active="exact"),
                html.Br(),
                html.Br(),
                dbc.NavLink("Model testing on application test", href="/app-test", active="exact"),
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
    dcc.Link('Data Study & cleanup', href='/data'),
    html.Br(),
    dcc.Link('Regression Logistic', href='/logistic-regression'),
    html.Br(),
    dcc.Link('Decision Tree', href='/decision-tree'),
    html.Br(),
    dcc.Link('Random Forest', href='/random-forest'),
    html.Br(),
    dcc.Link('XGBoost', href='/xgboost'),
    html.Br(),
    dcc.Link('Model testing on Application Test', href='/app-test'),
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
    html.P("Our problem is a very binary one : will someone repay their credit or won't they ? \n"
           "This is why we use logistic regression as our machine learning model.\n"),

    # Confusion Matrix
    html.Div(className='container',
             children=[
                 html.P('Logistic Regression Confusion Matrix:'),
                 dcc.Graph(figure=LR_conf_fig),
             ], style={'textAlign': 'center'}),
    html.P(LR_conf_str),

    # Accuracy
    html.P(LR_accu),
    html.P('Model accuracy is a machine learning model performance metric that is defined as the ratio of true '
           'positives and true negatives to all positive and negative observations.\n'
           'The accuracy rate is great but it doesn’t tell us anything about the errors our machine learning models '
           'make on new data we haven’t seen before.\n'
           'Mathematically, it represents the ratio of the sum of true positive and true negatives out of all the '
           'predictions.\n'),

    # Precision
    html.P(LR_precis),
    html.P("The precision score is a useful measure of the success of prediction when the classes are very "
           "imbalanced.\n"
           "Mathematically, it represents the ratio of true positive to the sum of true positive and false "
           "positive.\n"),

    # Recall
    html.P(LR_recall),
    html.P("Model recall score represents the model’s ability to correctly predict the positives out of actual "
           "positives. This is unlike precision which measures how many predictions made by models are actually "
           "positive out of all positive predictions made.\n"
           "Recall score is a useful measure of success of prediction when the classes are very imbalanced.\n"
           "Mathematically, it represents the ratio of true positive to the sum of true positive and false "
           "negative.\n"),

    # F1
    html.P(LR_F1),
    html.P("F1-score is harmonic mean of precision and recall score and is used as a metrics in the scenarios "
           "where choosing either of precision or recall score can result in compromise in terms of model giving "
           "high false positives and false negatives respectively.\n"),

])

# Decision Tree

layout_page_2 = html.Div([
    html.H1('Decision Tree'),
    html.Hr(),
    html.Br(),

    # Confusion Matrix
    html.Div(className='container',
             children=[
                 html.P('Decision Tree Confusion Matrix:'),
                 dcc.Graph(figure=DT_conf_fig),
             ], style={'textAlign': 'center'}),
    html.P(DT_conf_str),

    # Accuracy
    html.P(DT_accu)

])

# Random Forest

layout_page_3 = html.Div([
    html.H1('Random Forest'),
    html.Hr(),
    html.Br(),

    # Confusion Matrix
    html.Div(className='container',
             children=[
                 html.P('Random Forest Confusion Matrix:'),
                 dcc.Graph(figure=RF_conf_fig),
             ], style={'textAlign': 'center'}),
    html.P(RF_conf_str),

    # Accuracy
    html.P(RF_accu)

])

# XGBoost

layout_page_4 = html.Div([
    html.H1('XGBoost'),
    html.Hr(),
    html.Br(),

])

# Comparing Models on app-test
layout_page_5 = html.Div([
    html.H1('Model testing on application test'),
    html.Hr(),
    html.Br(),
    html.P(comparison)
])

'''

    html.Div(className='container',
             children=[
                 html.P('Application train dataframe :'),
                 dash_table.DataTable(
                     id='data_first',
                     columns=[{"name": i, "id": i} for i in columns],
                     data=app_train.to_dict('records'),
                     page_size=10,
                     style_header={
                         'backgroundColor': colors['background'],
                         'color': colors['title']
                     },
                     style_data={
                         'backgroundColor': colors['background'],
                         'color': colors['text']
                     },
                 ),
             ], style={'textAlign': 'center'}),

'''
