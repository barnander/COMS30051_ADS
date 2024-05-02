import dash
from dash import Dash, html, dcc
import dash_leaflet as dl
import webbrowser
from threading import Timer
import pandas as pd
import numpy as np
import filter_data as filter

#Load data, get unique values and set defaults
data = pd.read_csv("GTD.csv")
years = data["iyear"].unique()
countries = np.sort(data["country_txt"].unique())

year_default = 2001
country_default = "United States"

def generate_markers(country, year):
    filtered_data = filter.filter_basic(data, country, year)
    markers = []
    for i in range(len(filtered_data)):
        date_string = f"{filtered_data.iloc[i]['iday']}/{filtered_data.iloc[i]['imonth']}/{filtered_data.iloc[i]['iyear']}"
        tool_tip = date_string + f": \n{filtered_data.iloc[i]['gname']} attack: \n{int(filtered_data.iloc[i]['nkill'])} dead, {int(filtered_data.iloc[i]['nwound'])} wounded"
        #scale marker to number of casualties
        scale = 10 ** (3) * np.sqrt(2 * filtered_data.iloc[i]['nkill'] + filtered_data.iloc[i]['nwound']) + 100
        marker = dl.Circle(center=[filtered_data.iloc[i]["latitude"], filtered_data.iloc[i]["longitude"]], 
            radius= scale,
            #TODO change color based on type of attack?
            #TODO issue with multiple markers at same location
            color="black",
            fillColor="red",
            fillOpacity=0.5,
            children=[
                dl.Tooltip(tool_tip),
                dl.Popup(filtered_data.iloc[i]["summary"]),
            ]
        )
        markers.append(marker)
    return markers

#build app
app = Dash(__name__)
app.layout = html.Div([
    # Add a dropdown for countries
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in countries
            ],
        value= country_default
    ),
    # Add a slider for years
    dcc.Slider(
        id='year-slider',
        min=1970,
        max=2020,
        step=1,
        value=year_default,
        marks={str(year): str(year) for year in np.arange(1970,2020,5)}
        ),
    # Make map with marker layer
    dl.Map(id="map-layer", center=[56,10], zoom=6, style={'height': '100vh'}, children=[
        dl.TileLayer(),
        dl.LayerGroup(id='marker-layer', children = generate_markers("United States", 2001))]
    
    ),   
])

# Update markers on map
@app.callback(
    dash.Output("marker-layer", "children"),
    [dash.Input("country-dropdown", "value"),
     dash.Input("year-slider", "value")]
)
def update_map(country, year):
    return generate_markers(country, year)



def open_browser():
      webbrowser.open_new("http://127.0.0.1:8050/")
# Run the app
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, use_reloader=False)