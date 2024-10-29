import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tkinter as tk
from tkinter import ttk, messagebox
import math
import heapq
import random
from config.config import load_config

config = load_config()
# Constants
SPEED_LIMIT = config['SPEED_LIMIT']
DELAY = config['DELAY']
KM_TO_METERS = config['KM_TO_METERS']
SECONDS_PER_HOUR = config['SECONDS_PER_HOUR']
EARTH_RADIUS = config['EARTH_RADIUS']

# Load the dataset
train_df = pd.read_csv('data/Boroondara.csv', encoding='utf-8').fillna(0)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df['Vehicle Flow'].values.reshape(-1, 1))
lstm = load_model('model/lstm.h5')

df = train_df.copy()

# Function to calculate distance between two points using lat/long (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS * c  # Distance in kilometers

# Build SCATS site data from the dataset
scats_data = {}
for _, row in df.iterrows():
    scats_number = row['SCATS Number']
    latitude = row['NB_LATITUDE']
    longitude = row['NB_LONGITUDE']
    vehicle_flow = row['Vehicle Flow']

    if scats_number not in scats_data:
        scats_data[scats_number] = {'lat': latitude, 'lon': longitude, 'vehicle_flow': vehicle_flow, 'neighbors': {}}

# Define neighbors between SCATS sites
for site_a in scats_data:
    for site_b in scats_data:
        if site_a != site_b:
            # Calculate distance between two SCATS sites
            distance = haversine(scats_data[site_a]['lat'], scats_data[site_a]['lon'], 
                                 scats_data[site_b]['lat'], scats_data[site_b]['lon'])
            # Adding distance to neighbors
            scats_data[site_a]['neighbors'][site_b] = distance

# Function to calculate travel time based on traffic volume and distance
def calculate_travel_time(volume, distance):
    speed_in_m_per_sec = (SPEED_LIMIT * KM_TO_METERS) / SECONDS_PER_HOUR
    travel_time = (distance * KM_TO_METERS) / speed_in_m_per_sec
    delay_time = DELAY
    return travel_time + delay_time + (volume * 0.1)  # Adjust volume impact as needed

# Dijkstra's algorithm for finding multiple routes
def find_multiple_routes(scats_data, origin, destination, max_routes=5):
    queue = [(0, origin, [origin])]  
    visited = {}
    routes = []

    while queue and len(routes) < max_routes:
        current_time, current_node, path = heapq.heappop(queue)

        if current_node in visited:
            if visited[current_node] <= current_time:
                continue
        
        visited[current_node] = current_time

        if current_node == destination:
            routes.append((current_time, path))
            continue

        # Explore neighbors
        for neighbor, distance in scats_data[current_node]['neighbors'].items():
            if neighbor not in path:  # Avoid cycles
                vehicle_flow = scats_data[current_node]['vehicle_flow']
                travel_time = calculate_travel_time(vehicle_flow, distance) + random.uniform(0, 10)
                heapq.heappush(queue, (current_time + travel_time, neighbor, path + [neighbor]))

    return routes

# Prediction function
def predict_traffic_flow(model, scaler, input_data):
    """Predict traffic flow for a single row of input data."""
    input_data_scaled = scaler.transform(input_data.reshape(-1, 1)).reshape(1, -1)
    input_data_reshaped = input_data_scaled.reshape(1, input_data_scaled.shape[1], 1)
    predicted = model.predict(input_data_reshaped)
    predicted_flow = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()[0]
    return predicted_flow

# Fetching data from dataset
def fetch_data():
    """Fetch data from the dataset based on SCATS Number."""
    scats_number = scats_entry.get()
    date_value = date_entry.get()

    try:
        # Filter the dataset based on SCATS Number and Date
        filtered_data = train_df[(train_df['SCATS Number'] == int(scats_number)) & 
                                  (train_df['Date'] == date_value)]

        if not filtered_data.empty:
            data_row = filtered_data.iloc[0]
            
            # Auto-fill fields with data from the first row of the filtered dataset
            cd_melway_entry.config(state='normal')
            cd_melway_entry.delete(0, tk.END)
            cd_melway_entry.insert(0, data_row['CD_MELWAY'])
            cd_melway_entry.config(state='readonly')

            nb_latitude_entry.config(state='normal')
            nb_latitude_entry.delete(0, tk.END)
            nb_latitude_entry.insert(0, data_row['NB_LATITUDE'])
            nb_latitude_entry.config(state='readonly')

            nb_longitude_entry.config(state='normal')
            nb_longitude_entry.delete(0, tk.END)
            nb_longitude_entry.insert(0, data_row['NB_LONGITUDE'])
            nb_longitude_entry.config(state='readonly')

            vr_internal_entry.config(state='normal')
            vr_internal_entry.delete(0, tk.END)
            vr_internal_entry.insert(0, data_row['VR Internal Stat'])
            vr_internal_entry.config(state='readonly')

            nb_type_entry.config(state='normal')
            nb_type_entry.delete(0, tk.END)
            nb_type_entry.insert(0, data_row['NB_TYPE_SURVEY'])
            nb_type_entry.config(state='readonly')

        else:
            messagebox.showerror("Error", "No data found for the provided SCATS Number and Date!")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid SCATS Number and Date in YYYY-MM-DD format.")

def predict():
    """Predict traffic flow based on the user input."""
    try:
        # Prepare input data excluding auto-filled fields
        new_data_row = np.array([
            0,
            int(scats_entry.get()),  
            int(location_entry.get()),
            int(cd_melway_entry.get()),  
            float(nb_latitude_entry.get()),  
            float(nb_longitude_entry.get()),
            int(hf_vicroads_entry.get()),  
            int(vr_internal_entry.get()),  
            int(vr_internal_loc_entry.get()), 
            int(nb_type_entry.get()),  
            start_time_entry.get(),
            0 
        ])
        predicted_flow = predict_traffic_flow(lstm, scaler, new_data_row)
        result_label.config(text=f'Predicted Traffic Flow: {predicted_flow:.2f}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# SCATS Routing GUI
def find_routes():
    """Find multiple routes based on user input."""
    try:
        origin = int(origin_entry.get())
        destination = int(destination_entry.get())

        if origin not in scats_data or destination not in scats_data:
            messagebox.showerror("Error", "Invalid SCATS Number.")
            return

        routes = find_multiple_routes(scats_data, origin, destination)
        if routes:
            route_output = "\n".join([
            f"Route {i+1}: {' -> '.join(map(str, path))}, Time: {int(time // 60)} min {time % 60:.2f} sec" 
            for i, (time, path) in enumerate(routes)
        ])

            messagebox.showinfo("Routes", route_output)
        else:
            messagebox.showinfo("Routes", "No routes found.")
    except ValueError:
        messagebox.showerror("Error", "Invalid SCATS Numbers entered.")

# Create the main application window
root = tk.Tk()
root.title("Traffic Flow & SCATS Routing Application")

# Create two frames for the left and right sections
left_frame = ttk.Frame(root, padding="10")
right_frame = ttk.Frame(root, padding="10")

left_frame.grid(row=0, column=0, sticky="nsew")
right_frame.grid(row=0, column=1, sticky="nsew")

# Configure window to expand the columns properly
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# ================= Left Frame (Prediction Section) =================
ttk.Label(left_frame, text="Traffic Flow Prediction", font=("Helvetica", 14)).grid(row=0, column=0, columnspan=2, pady=10)

ttk.Label(left_frame, text="SCATS Number:").grid(row=1, column=0, sticky="w")
scats_entry = ttk.Entry(left_frame)
scats_entry.grid(row=1, column=1)

ttk.Label(left_frame, text="Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w")
date_entry = ttk.Entry(left_frame)
date_entry.grid(row=2, column=1)

ttk.Button(left_frame, text="Fetch Data", command=fetch_data).grid(row=3, column=1, pady=5)

ttk.Label(left_frame, text="Location:").grid(row=4, column=0, sticky="w")
location_entry = ttk.Entry(left_frame)
location_entry.grid(row=4, column=1)

ttk.Label(left_frame, text="CD Melway:").grid(row=5, column=0, sticky="w")
cd_melway_entry = ttk.Entry(left_frame, state="readonly")
cd_melway_entry.grid(row=5, column=1)

ttk.Label(left_frame, text="NB Latitude:").grid(row=6, column=0, sticky="w")
nb_latitude_entry = ttk.Entry(left_frame, state="readonly")
nb_latitude_entry.grid(row=6, column=1)

ttk.Label(left_frame, text="NB Longitude:").grid(row=7, column=0, sticky="w")
nb_longitude_entry = ttk.Entry(left_frame, state="readonly")
nb_longitude_entry.grid(row=7, column=1)

ttk.Label(left_frame, text="HF VicRoads:").grid(row=8, column=0, sticky="w")
hf_vicroads_entry = ttk.Entry(left_frame)
hf_vicroads_entry.grid(row=8, column=1)

ttk.Label(left_frame, text="VR Internal Stat:").grid(row=9, column=0, sticky="w")
vr_internal_entry = ttk.Entry(left_frame, state="readonly")
vr_internal_entry.grid(row=9, column=1)

ttk.Label(left_frame, text="VR Internal Loc:").grid(row=10, column=0, sticky="w")
vr_internal_loc_entry = ttk.Entry(left_frame)
vr_internal_loc_entry.grid(row=10, column=1)

ttk.Label(left_frame, text="NB Type Survey:").grid(row=11, column=0, sticky="w")
nb_type_entry = ttk.Entry(left_frame, state="readonly")
nb_type_entry.grid(row=11, column=1)

ttk.Label(left_frame, text="Start Time:").grid(row=12, column=0, sticky="w")
start_time_entry = ttk.Entry(left_frame)
start_time_entry.grid(row=12, column=1)

ttk.Button(left_frame, text="Predict Traffic Flow", command=predict).grid(row=13, column=1, pady=10)

result_label = ttk.Label(left_frame, text="Predicted Traffic Flow: ")
result_label.grid(row=14, column=1, pady=10)

# ================= Right Frame (Route Finding Section) =================
ttk.Label(right_frame, text="Find SCATS Routes", font=("Helvetica", 14)).grid(row=0, column=0, columnspan=2, pady=10)

ttk.Label(right_frame, text="Origin SCATS Number:").grid(row=1, column=0, sticky="w")
origin_entry = ttk.Entry(right_frame)
origin_entry.grid(row=1, column=1)

ttk.Label(right_frame, text="Destination SCATS Number:").grid(row=2, column=0, sticky="w")
destination_entry = ttk.Entry(right_frame)
destination_entry.grid(row=2, column=1)

ttk.Button(right_frame, text="Find Routes", command=find_routes).grid(row=3, column=1, pady=10)

# Start the application loop
root.mainloop()
