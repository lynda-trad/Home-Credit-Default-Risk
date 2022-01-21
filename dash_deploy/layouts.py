# ML Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# Undersampling
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Grid Search
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Boruta
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from sklearn.datasets import load_boston

# Machine Learning Library
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

#################################
# DATAVIZ

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

# Education Type
education_type_fig = px.histogram(app_train,
                                  y="NAME_EDUCATION_TYPE",
                                  color="TARGET",
                                  title='Repayment distribution depending on the education type')

# Occupation Type
occupation_type_fig = px.histogram(app_train,
                                   y="NAME_HOUSING_TYPE",
                                   color="TARGET",
                                   title='Repayment distribution depending on the occupation type')

# Housing Type
housing_type_fig = px.histogram(app_train,
                                y="OCCUPATION_TYPE",
                                color="TARGET",
                                title='Repayment distribution depending on the housing type')

# Number of Children
children_num_fig = px.histogram(app_train,
                                x="CNT_CHILDREN",
                                color="TARGET",
                                title='Repayment distribution depending on the number of children')

# Family Status
family_status_fig = px.histogram(app_train,
                                 y="NAME_FAMILY_STATUS",
                                 color="TARGET",
                                 title='Repayment distribution depending on the family status')


# Gender Distribution

app_train = app_train[app_train.CODE_GENDER != 'XNA']
def gender_distribution():
    gender_group = app_train.groupby(['CODE_GENDER']).size().reset_index(name='count')
    return px.pie(gender_group,
                  values='count',
                  names='CODE_GENDER',
                  title='Gender Distribution',
                  height=700,
                  width=1000
                  )


gender_pie_fig = gender_distribution()
print("gender distribution\n")

gender_hist_fig = px.histogram(app_train,
                               x="CODE_GENDER",
                               color="TARGET",
                               title='Repayment distribution among genders')


# Contract Type Distribution
def contract_distribution():
    contract_group = app_train.groupby(['NAME_CONTRACT_TYPE']).size().reset_index(name='count')
    return px.pie(contract_group,
                  values='count',
                  names='NAME_CONTRACT_TYPE',
                  title='Contract Type Distribution')


contract_fig = contract_distribution()
print("contract distribution\n")


# DAYS BIRTH feature

# Min - Max
def birth_min_max():
    mini = abs(app_train['DAYS_BIRTH'].max())
    if mini > 365:
        birth_mini = "Days birth min: " + str(mini / 365) + " years\n"
    else:
        birth_mini = "Days birth min: " + str(mini) + " days\n"
    maxi = abs(app_train['DAYS_BIRTH'].min())
    birth_maxi = "Days birth max: " + str(maxi / 365) + " years\n"
    return birth_mini + '\n' + birth_maxi


age_minmax = birth_min_max()
print("birth minmax\n")

# Boxplot
daysbirth_boxplot = px.box(app_train,
                           y='DAYS_BIRTH',
                           title='Boxplot of Days Birth',
                           height=700,
                           width=1000
                           )
print("birth boxplot\n")


# Days Employed feature

# Min - Max
def employ_min_max():
    mini = abs(app_train['DAYS_EMPLOYED'].max())
    if mini > 365:
        employ_min = "Days employed min :" + str(mini / 365) + "years"
    else:
        employ_min = "Days employed min :" + str(mini) + "days"

    maxi = abs(app_train['DAYS_EMPLOYED'].min())
    employ_max = "Days employed max :" + str(maxi / 365) + "years"
    return employ_min + '\n' + employ_max


employ_minmax = employ_min_max()
print("employed minmax\n")

#################################
# DATA CLEANUP

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


# Label encoding

def labelEncodingAppTrain():
    le = LabelEncoder()
    count = 0
    for col in app_train:
        if app_train[col].dtype == 'object' or app_train[col].dtype == 'string':
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            count += 1
    app_train.reset_index()
    app_test.reset_index()
    return app_train, app_test, count


app_train, app_test, le_count = labelEncodingAppTrain()
label_encoding_str = '%d columns were label encoded.' % le_count
print("label encoding\n")

# Infinite Values
app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
app_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Missing Values
# app_train
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_train)
imputer = imputer.fit_transform(app_train)
app_train = pd.DataFrame(imputer, columns=app_train.columns.values.tolist())
# app_test
imputer = SimpleImputer(missing_values=np.nan, strategy="median").fit(app_test)
imputer = imputer.fit_transform(app_test)
app_test = pd.DataFrame(imputer, columns=app_test.columns.values.tolist())
print("infinite and nan values\n")

# Undersampling
X = app_train
Y = np.array(app_train['TARGET'])
X.drop('TARGET', axis=1, inplace=True)

rus = RandomUnderSampler(random_state=0)
app_train, y_resampled = rus.fit_resample(X, Y)
app_train['TARGET'] = y_resampled
print(sorted(Counter(y_resampled).items()), y_resampled.shape)

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


######################################
# MACHINE LEARNING

# Random Forest

# RF Confusion Matrix
def RF_confus_matrix():
    conf_matrix = [[3772, 1687], [1867, 3592]]
    x = ['0', '1']
    y = ['1', '0']
    conf_value = [[str(y) for y in x] for x in conf_matrix]
    fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')
    RF_str = 'The confusion matrix shows us the number of :\n' + \
             '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
             '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
             '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
             '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
    return fig, RF_str


RF_conf_fig, RF_conf_str = RF_confus_matrix()
print("RF confusion matrix\n")


# RF Model Score
def RF_model_score():
    RF_accu = "Accuracy score:" + '67.45' + '%\n' + \
              "\nAccuracy score using cross validation:" + '67.13%' + '%\n'
    print("RF accuracy score\n")

    RF_precis = "Precision score:" + '67.47' + '%\n'
    print("RF precision score\n")

    RF_recall = "Recall score:" + '65.8' + '%\n'
    print("RF recall score\n")

    RF_F1 = "F1 Score:", '66.9' + '%\n'
    print("RF F1 score\n")
    return RF_accu, RF_precis, RF_recall, RF_F1


RF_accu, RF_precis, RF_recall, RF_F1 = RF_model_score()

print("RF model score\n")

# KNeighbors
# KNeighbors Best Model Score
KN_bestmodelscore = str(55.99)
print("KN best model score\n")


#  Kneighbors Confusion Matrix
def KN_confus_matrix():
    conf_matrix = [[2977, 2482], [2323, 3136]]
    x = ['0', '1']
    y = ['1', '0']
    conf_value = [[str(y) for y in x] for x in conf_matrix]
    fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')
    KN_str = 'The confusion matrix shows us the number of :\n' + \
             '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
             '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
             '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
             '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
    return fig, KN_str


KN_conf_fig, KN_conf_str = KN_confus_matrix()
print("KN confusion matrix\n")

# Kneighbors Cross Validation Accuracy

KN_accu = "Accuracy score using cross validation:" + '56.33' + '%\n'
print("KN accuracy score\n")

"""
# Previous Machine Learning

# Boruta
# Splitting data into train / test
print("Starting Boruta\n")


# App train ou app test dans data split ????
def data_split(app_train, app_test):
    Xdf = app_train.copy()
    Xdf.drop('TARGET', axis=1, inplace=True)
    X_boruta = Xdf
    y = app_train["TARGET"]

    forest = RandomForestRegressor(
        n_jobs=-1,
        max_depth=5
    )
    boruta = bp(
        estimator=forest,
        n_estimators=20,
        max_iter=100  # numbers of trials
    )
    boruta.fit(np.array(X_boruta), np.array(y))

    # Features to keep
    green_area = X_boruta.columns[boruta.support_].to_list()
    blue_area = X_boruta.columns[boruta.support_weak_].to_list()
    features = green_area + blue_area

    X = X_boruta[features]
    app_train = app_train[features]
    app_test = app_test[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    return X, y, X_train, X_test, y_train, y_test, app_train, app_test


X, y, X_train, X_test, y_train, y_test, app_train, app_test = data_split(app_train, app_test)
# real = "Real values:\n\n" + y_test
print("split data into test/train\n")

# Random Forest

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
# TODO
#  RF_predictions = "Predictions:\n\n" + y_pred + '\n'
#  RF_results = RF_predictions + '\n' + real
print("RF predict\n")


# RF Confusion Matrix
def RF_confus_matrix():
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    x = ['0', '1']
    y = ['1', '0']
    conf_value = [[str(y) for y in x] for x in conf_matrix]
    fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')
    RF_str = 'The confusion matrix shows us the number of :\n' + \
             '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
             '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
             '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
             '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
    return fig, RF_str


RF_conf_fig, RF_conf_str = RF_confus_matrix()
print("RF confusion matrix\n")


# RF Model score
def RF_model_score():
    RF_accu = "Accuracy score:" + str(round((accuracy_score(y_test, y_pred) * 100), 2)) + '%\n' + \
              "\nAccuracy score using cross validation:" + \
              str(round((cross_val_score(RF, X_train, y_train, cv=3, scoring='accuracy').mean()) * 100, 2)) + '%\n'
    print("LR accuracy score\n")

    RF_precis = "Precision score:" + str(round((precision_score(y_test, y_pred, average='macro') * 100), 2)) + '%\n'
    print("LR precision score\n")

    RF_recall = "Recall score:" + str(round((metrics.recall_score(y_test, y_pred) * 100), 2)) + '%\n'
    print("LR recall score\n")

    RF_F1 = "F1 Score:", str(round((metrics.f1_score(y_test, y_pred) * 100), 2)) + '%\n'
    print("LR F1 score\n")
    return RF_accu, RF_precis, RF_recall, RF_F1


RF_accu, RF_precis, RF_recall, RF_F1 = RF_model_score()
print("RF Model score\n")


# Testing RF on app-test
def RF_app_test():
    app_test_RF = app_test.copy()
    # app_test_RF['TARGET'] = 0
    y_pred_test = RF.predict(app_test_RF)
    app_test_RF['TARGET'] = y_pred_test.astype(int)
    return app_test_RF


# TODO print
app_test_RF = RF_app_test()
print("RF tapp-test\n")

# KNeighbors
# Hyper parameters
param_grid = {'n_neighbors': np.arange(1, 5),
              'metric': ['euclidean', 'manhattan']
              }
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

#  Kneighbors Model training
grid.fit(X_train, y_train)

# KNeighbors Best Model Score

KScore = round(grid.best_score_, 2) * 100

# Kneighbors Best Parameters

Kbest_param = grid.best_params_

# Saving best Kneighbors model

KN = grid.best_estimator_
K_bestmodelscore = round(KN.score(X_test, y_test) * 100, 2)


#  Kneighbors Confusion Matrix
def KN_confus_matrix():
    conf_matrix = metrics.confusion_matrix(y_test, KN.predict(X_test))
    x = ['0', '1']
    y = ['1', '0']
    conf_value = [[str(y) for y in x] for x in conf_matrix]
    fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=conf_value, colorscale='Viridis')
    KN_str = 'The confusion matrix shows us the number of :\n' + \
             '\n* True positives :' + str(conf_matrix[0][0]) + '\n' + \
             '\n* True negatives :' + str(conf_matrix[0][1]) + '\n' + \
             '\n* False positives:' + str(conf_matrix[1][0]) + '\n' + \
             '\n* False negatives:' + str(conf_matrix[1][1]) + '\n'
    return fig, KN_str


KN_conf_fig, KN_conf_str = KN_confus_matrix()
print("KN Confusion Matrix\n")

# Kneighbors Cross Validation Accuracy

KN_accu = "Accuracy score using cross validation:" + \
          str(round((cross_val_score(KN, X_train, y_train, cv=3,
                                     scoring='accuracy').mean()) * 100, 2)) + '%\n'
print("KN accuracy score\n")

#  Kneighbors Learning Curve

N, train_score, val_score = learning_curve(KN,
                                           X_train,
                                           y_train,
                                           train_sizes=np.linspace(0.1, 1.0, 10),
                                           cv=5)

plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()
# plt.show()
"""

######################################################

# Menu SideBar
navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Data Study & Cleanup", href="/data"),
                dbc.DropdownMenuItem("Random Forest", href="/random-forest"),
                dbc.DropdownMenuItem("KNeighbors", href="/kneighbors"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Menu",
    brand_href="/",
    color="navbar-bran",
    dark=True,
)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
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
    dcc.Link('Random Forest', href='/random-forest'),
    html.Br(),
    dcc.Link('KNeighbors', href='/kneighbors'),
    html.Br(),
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

    # Data Viz

    # Unbalanced Data
    html.Div(className='container',
             children=[
                 html.P('Unbalanced data figure :'),
                 dcc.Graph(figure=unbalanced_train_fig),
             ], style={'textAlign': 'center'}),

    # Education Type
    html.Div(className='container',
             children=[
                 html.P('Education Type Repayment Distribution :'),
                 dcc.Graph(figure=education_type_fig),
             ], style={'textAlign': 'center'}),

    # Occupation Type
    html.Div(className='container',
             children=[
                 html.P('Occupation Type Repayment Distribution :'),
                 dcc.Graph(figure=occupation_type_fig),
             ], style={'textAlign': 'center'}),

    # Housing Type
    html.Div(className='container',
             children=[
                 html.P('Housing Type Repayment Distribution :'),
                 dcc.Graph(figure=housing_type_fig),
             ], style={'textAlign': 'center'}),

    # Housing Type
    html.Div(className='container',
             children=[
                 html.P('Repayment Distribution depending on the Number of Children :'),
                 dcc.Graph(figure=children_num_fig),
             ], style={'textAlign': 'center'}),

    # Family Status
    html.Div(className='container',
             children=[
                 html.P('Family Status Repayment Distribution :'),
                 dcc.Graph(figure=family_status_fig),
             ], style={'textAlign': 'center'}),

    # Gender Distribution PieChart
    html.Div(className='container',
             children=[
                 html.P('Gender Distribution piechart :'),
                 dcc.Graph(figure=gender_pie_fig),
             ]),

    # Gender Distribution Hist
    html.Div(className='container',
             children=[
                 html.P('Repayment distribution between genders :'),
                 dcc.Graph(figure=gender_hist_fig),
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

# Random Forest

layout_page_3 = html.Div([
    html.H1('Random Forest'),
    html.Hr(),
    html.Br(),

    html.Div('* We undersample the dataframe to get an even amount of each target value.'),
    html.P(compare_balance),
    html.Br(),

    # Balanced Data
    html.Div(className='container',
             children=[
                 html.P('Balanced data figure :'),
                 dcc.Graph(figure=balanced_train_fig),
             ]),

    # Confusion Matrix
    html.Div(className='container',
             children=[
                 html.P('Random Forest Confusion Matrix:'),
                 dcc.Graph(figure=RF_conf_fig),
             ], style={'textAlign': 'center'}),
    html.P(RF_conf_str),

    # Accuracy
    html.P(RF_accu),
    html.P('Model accuracy is a machine learning model performance metric that is defined as the ratio of true '
           'positives and true negatives to all positive and negative observations.\n'
           'The accuracy rate is great but it doesn’t tell us anything about the errors our machine learning models '
           'make on new data we haven’t seen before.\n'
           'Mathematically, it represents the ratio of the sum of true positive and true negatives out of all the '
           'predictions.\n'),

    # Precision
    html.P(RF_precis),
    html.P("The precision score is a useful measure of the success of prediction when the classes are very "
           "imbalanced.\n"
           "Mathematically, it represents the ratio of true positive to the sum of true positive and false "
           "positive.\n"),

    # Recall
    html.P(RF_recall),
    html.P("Model recall score represents the model’s ability to correctly predict the positives out of actual "
           "positives. This is unlike precision which measures how many predictions made by models are actually "
           "positive out of all positive predictions made.\n"
           "Recall score is a useful measure of success of prediction when the classes are very imbalanced.\n"
           "Mathematically, it represents the ratio of true positive to the sum of true positive and false "
           "negative.\n"),

    # F1
    html.P(RF_F1),
    html.P("F1-score is harmonic mean of precision and recall score and is used as a metrics in the scenarios "
           "where choosing either of precision or recall score can result in compromise in terms of model giving "
           "high false positives and false negatives respectively.\n"),

])

# KNeighbors

layout_page_4 = html.Div([
    html.H1('KNeighbors'),
    html.Hr(),
    html.Br(),
    html.P('We use GridSearch on KNeighbors and the Random Forest model.'),

    # Best Model Score
    html.P("Best Model Score: ", KN_bestmodelscore),

    # Confusion Matrix
    html.Div(className='container',
             children=[
                 html.P('KNeighbors Classifier Confusion Matrix:'),
                 dcc.Graph(figure=KN_conf_fig),
             ], style={'textAlign': 'center'}),
    html.P(KN_conf_str),

    # Accuracy
    html.P(KN_accu),

])
