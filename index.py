from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from layouts import sidebar, layout_index, layout_page_0, layout_page_1, \
    layout_page_2, layout_page_3, layout_page_4, layout_page_5
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    sidebar
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/data":
        return layout_page_0
    elif pathname == "/logistic-regression":
        return layout_page_1
    elif pathname == "/decision-tree":
        return layout_page_2
    elif pathname == "/random-forest":
        return layout_page_3
    elif pathname == "/xgboost":
        return layout_page_4
    elif pathname == "/app-test":
        return layout_page_5
    else:
        return layout_index


if __name__ == '__main__':
    app.run_server(debug=True)
