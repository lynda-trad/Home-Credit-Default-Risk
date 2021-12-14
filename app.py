import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import flask

app = dash.Dash(__name__)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Home

layout_index = html.Div([
    html.H1('Home Credit Default Risk'),
    html.Div('We have a dataframe in which are registered a lot of information about a bank\'s clients.'),
    html.Br(),
    html.Div('The bank wants to know if they should give them a loan or not. The machine learning '
             'model needs to predict whether or not the client will repay the loan.'),
    html.Br(),
    html.Div('If the target is equal to 0: the loan was repaid, if it\'s equal to 1: the loan was not repaid.'),
    html.Br(),
    html.Div('Thanks to the given features, the models will learn how to predict the target.'''),
    html.Br(),

    html.H2('Summary'),
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

    html.H2('Conclusion'),
    html.Div('The data given was very unbalanced, we had to use oversampling to balance it to get accurate models.'),
    html.Br(),
    html.Div('We can see that the least efficient model was the logistic regression.'),
    html.Br(),
    html.Div('As the bank, if the model predicts too many true positives, this is not an issue.'),
    html.Br(),
    html.Div('The bank would have not given the loan to someone who could have repaid it. '
             'This isn\'t so great for the clients.'),
    html.Br(),
    html.Img(src='./assets/bank.png'),
])

# Logistic Regression

layout_page_1 = html.Div([
    html.H1('Logistic Regression'),
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('Random Forest', href='/page-3'),
    html.Br(),
    dcc.Link('XGBoost', href='/page-4'),
    html.Br(),
    dcc.Link('Model testing on application test', href='/page-5'),
    html.Br(),
    html.Img(src='./assets/bank.png')
])

# Decision Tree

layout_page_2 = html.Div([
    html.H1('Decision Tree'),
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('Random Forest', href='/page-3'),
    html.Br(),
    dcc.Link('XGBoost', href='/page-4'),
    html.Br(),
    dcc.Link('Model testing on application test', href='/page-5'),
    html.Br(),
    html.Img(src='./assets/bank.png')
])

# Random Forest

layout_page_3 = html.Div([
    html.H1('Random Forest'),
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Logistic Regression', href='/page-1'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('XGBoost', href='/page-4'),
    html.Br(),
    dcc.Link('Model testing on application test', href='/page-5'),
    html.Br(),
    html.Img(src='./assets/bank.png')
])

# XGBoost

layout_page_4 = html.Div([
    html.H1('XGBoost'),
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Logistic Regression', href='/page-1'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('Random Forest', href='/page-3'),
    html.Br(),
    dcc.Link('Model testing on application test', href='/page-5'),
    html.Br(),
    html.Img(src='./assets/bank.png')
])


layout_page_5 = html.Div([
    html.H1('Model testing on application test'),
    html.Br(),
    dcc.Link('Home', href='/'),
    html.Br(),
    dcc.Link('Logistic Regression', href='/page-1'),
    html.Br(),
    dcc.Link('Decision Tree', href='/page-2'),
    html.Br(),
    dcc.Link('Random Forest', href='/page-3'),
    html.Br(),
    dcc.Link('XGBoost', href='/page-4'),
    html.Br(),
    html.Img(src='./assets/bank.png')
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
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
    if pathname == "/page-1":
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
