from flask import Flask, request, render_template
from flask_cors import CORS
import pydeck as pdk
import pandas as pd

app = Flask(__name__, template_folder='.')
CORS(app)

df = pd.read_csv('GTD.csv')
df2 = pd.read_csv("filtered_transfers.csv")

def load_countries(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

@app.route('/')
def index():
    attack_countries = load_countries('countries1.txt')
    transfer_countries = load_countries('countries.txt')
    return render_template('index.html', attack_countries=attack_countries, transfer_countries=transfer_countries)

@app.route('/generate_map', methods=['GET'])
def generate_map():
    attack_country = request.args.get('attack_country')
    transfer_country = request.args.get('transfer_country')

    filtered_df = df[df['country_txt'] == attack_country] if attack_country else df
    filtered_df = filtered_df[["iyear", 'latitude', 'longitude', 'nkill']].dropna()
    filtered_df2 = df2[df2['Country_x'] == transfer_country] if transfer_country else df2
    filtered_df2 = filtered_df2[["Country_x", "Latitude_x", "Longitude_x", "Latitude_y", "Longitude_y", "SIPRI_TIV_of_delivered_weapons"]].dropna()

    layer1 = pdk.Layer(
        "ArcLayer",
        data=filtered_df2,
        get_width="2",
        get_source_position=["Longitude_y", "Latitude_y"],
        get_target_position=["Longitude_x", "Latitude_x"],
        get_tilt=15,
        get_source_color=[240, 100, 0, 40],
        get_target_color=[0, 255, 0, 40],
        pickable=True,
        auto_highlight=True,
    )
    layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_df,
        get_position=['longitude', 'latitude'],
        get_weight=1,
        radius_pixels=40,
        opacity=0.9
    )

    view_state = pdk.ViewState(
        latitude=filtered_df['latitude'].mean() if not filtered_df.empty else 0,
        longitude=filtered_df['longitude'].mean() if not filtered_df.empty else 0,
        zoom=4,
        bearing=0,
        pitch=45
    )
    
    r = pdk.Deck(
        layers=[layer, layer1],
        initial_view_state=view_state,
    )
    return r.to_html(as_string=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
