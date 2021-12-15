from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from layouts import sidebar, layout_index, layout_page_0, layout_page_1, layout_page_2, layout_page_3, layout_page_4, \
    layout_page_5
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    sidebar
])


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
