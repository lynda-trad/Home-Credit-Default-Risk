from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from layouts import sidebar, layout_index, \
    layout_page_0, layout_page_3, layout_page_4

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
    elif pathname == "/random-forest":
        return layout_page_3
    elif pathname == "/kneighbors":
        return layout_page_4
    else:
        return layout_index


if __name__ == '__main__':
    app.run_server(debug=True)
