from flask import Flask, request, render_template
from flask_cors import CORS
import pydeck as pdk
import pandas as pd

app = Flask(__name__, template_folder='.')
CORS(app)

def load_countries(filename):
    with open(filename, 'r', encoding='UTF-8') as file:
        return [line.strip() for line in file if line.strip()]

def safe_sum(values):
    total = 0.0
    unknown_count = 0
    for val in values:
        if isinstance(val, float) and not pd.isna(val):
            total += val
        elif val != 'Not Known':
            total += float(val)
        else:
            unknown_count += 1

    if unknown_count == len(values):
        return 'Unknown'
    return total
    
gtd = pd.read_csv('gtd_data.csv')
lat = pd.read_csv('lat_data.csv', encoding='ISO-8859-1')
sat = pd.read_csv('sat_data.csv')
countries = load_countries('countries.txt')

@app.route('/')
def index():
    return render_template('index.html', countries=countries)

@app.route('/generate_map', methods=['GET'])
def generate_map():
    country = request.args.get('country')
    start_year = request.args.get('start_year')
    end_year = request.args.get('end_year')
    visualization_type = request.args.get('vis_type', 'heatmap')  
    show_large_arms = request.args.get('large_arms', 'true') == 'true'  
    show_small_arms = request.args.get('small_arms', 'true') == 'true'

    filtered_gtd = gtd.copy()
    if country:
        filtered_gtd = filtered_gtd[filtered_gtd['country'] == country]
    if start_year:
        filtered_gtd = filtered_gtd[filtered_gtd['iyear'] >= int(start_year)]
    if end_year:
        filtered_gtd = filtered_gtd[filtered_gtd['iyear'] <= int(end_year)]

    filtered_lat = lat.copy()
    if country:
        filtered_lat = filtered_lat[filtered_lat['importer'] == country]
    if start_year:
        filtered_lat = filtered_lat[filtered_lat['Year of order'] >= int(start_year)]
    if end_year:
        filtered_lat = filtered_lat[filtered_lat['Year of order'] <= int(end_year)]
    
    lat_combined = filtered_lat.groupby('exporter').agg({
        'importer': 'first',
        'latitude_importer': 'first',
        'longitude_importer': 'first',
        'latitude_exporter': 'first',
        'longitude_exporter': 'first',
        'Year of order': lambda x: list(x),
        'SIPRI TIV of delivered weapons': lambda x: safe_sum(x),
        'tooltip': lambda x: ',<br>'.join(x)
    }).reset_index()
    lat_combined['tooltip'] = 'Total SIPRI TIV: ' + lat_combined['SIPRI TIV of delivered weapons'].fillna('Unknown').astype(str) + '<br>' + lat_combined['tooltip']

    filtered_sat = sat.copy()
    if country:
        filtered_sat = filtered_sat[filtered_sat['importer'] == country]
    if start_year:
        filtered_sat = filtered_sat[filtered_sat['Year'] >= int(start_year)]
    if end_year:
        filtered_sat = filtered_sat[filtered_sat['Year'] <= int(end_year)]

    sat_combined = filtered_sat.groupby('exporter').agg({
        'importer': 'first',
        'latitude_importer': 'first',
        'longitude_importer': 'first',
        'latitude_exporter': 'first',
        'longitude_exporter': 'first',
        'Year': lambda x: list(x),
        'Value': lambda x: safe_sum(x),
        'tooltip': lambda x: ',<br>'.join(x)
    }).reset_index()
    sat_combined['tooltip'] = 'Total Value: $' + sat_combined['Value'].fillna('Unknown').astype(str) + '<br>' + sat_combined['tooltip']

    lat_layer = pdk.Layer(
        "ArcLayer",
        data=lat_combined,
        get_width=5,
        get_source_position=["longitude_exporter", "latitude_exporter"],
        get_target_position=["longitude_importer", "latitude_importer"],
        get_tilt=15,
        get_source_color=[240, 100, 0, 40],
        get_target_color=[0, 255, 0, 40],
        pickable=True,
        auto_highlight=True,
    )

    sat_layer = pdk.Layer(
    "ArcLayer",
    data=sat_combined,
    get_width=5,
    get_source_position=["longitude_importer", "latitude_importer"],
    get_target_position=["longitude_exporter", "latitude_exporter"],
    get_tilt=10,
    get_source_color=[0, 0, 255, 40],
    get_target_color=[0, 255, 0, 40],
    pickable=True,
    auto_highlight=True,
    )

    if visualization_type == 'heatmap':
        gtd_layer = pdk.Layer(
            "HeatmapLayer",
            data=filtered_gtd,
            get_position=['longitude', 'latitude'],
            get_weight=1,
            radius_pixels=40,
            opacity=0.9
        )
    else:  
        gtd_layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_gtd,
            get_position=['longitude', 'latitude'],
            get_color=[255, 0, 0, 160],
            get_radius=5000,
            pickable=True
        )

    layers = [gtd_layer]

    if show_large_arms:
        layers.append(lat_layer)

    if show_small_arms:
        layers.append(sat_layer)

    view_state = pdk.ViewState(
        latitude=filtered_gtd['latitude'].mean() if not filtered_gtd.empty else 0,
        longitude=filtered_gtd['longitude'].mean() if not filtered_gtd.empty else 0,
        zoom=4,
        bearing=0,
        pitch=45
    )
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            'html': '{tooltip}',
            }
    )
    return r.to_html(as_string=True)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
