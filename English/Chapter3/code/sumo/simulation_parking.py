"""Run the baseline SUMO parking simulation and collect edge / trip outputs."""

# Configuration.
# Main OD-based scenario.
setting_folder = "od_koto"  # random / od_koto
demand_scenario = "93"
# Random-demand alternative.
# setting_folder = "random"  # random / od_koto
network_dir = 'network/data'
output_dir = f'output/1.0normal_1.0evac_{demand_scenario}'
period_average = 300  # Aggregation interval used for MFD-related edge outputs.
cui = True

import os
import sys
import subprocess
import pandas as pd
import pandas_read_xml as pdx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import lxml.etree as ET  # Parse XML files with lxml.
import traci
# If you decide to switch to libsumo:
#import libsumo as traci
import sumolib
import platform
import distro
import copy
import argparse
import shutil
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(prog="simulation_parking.py",
                                 description="Run SUMO simulation with parking")
parser.add_argument('-f', '--folder', type=str, required=True,
                    help='Setting folder in sumo/demand/random or sumo/demand/od_koto')
args = parser.parse_args()

setting = args.folder
if args.folder.startswith("zone"):
    network_dir = f'network/data/{args.folder.split("_")[0]}'
    demand_dir = f'demand/{setting_folder}/{setting}'
else:
    demand_dir = f'demand/{setting_folder}/{setting}/{demand_scenario}'

def detect_os():
    """Return a coarse OS label used for SUMO environment setup."""
    os_name = platform.system()
    if os_name == 'Linux':
        if 'Ubuntu' in distro.id():
            return "Ubuntu"
        else:
            return "Other Linux"
    elif os_name == 'Darwin':
        return "macOS"
    else:
        return "Other OS"

os_type = detect_os()

if os_type == "Ubuntu":
    # Configure SUMO_HOME.
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")
else:
    # Configure SUMO_HOME.
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

# Input paths.
net_file = os.path.join(network_dir, 'output.net.xml')
#trips_file = os.path.join(demand_dir, 'trips.trips.xml')
routes_file = os.path.join(demand_dir, 'routes.rou.xml')
routes_file = os.path.join(demand_dir, 'trips.trips.xml')
parking_csv = os.path.join(demand_dir, 'shelter_list.csv')
zone_edge_file = os.path.join('../network_partitioning/zoning.csv')

# Output paths.
additional_file = 'additional.add.xml'  # Vehicle types and edge-data output definitions.
edge_output_file = os.path.join(output_dir, 'edge_data.xml')
parking_output_file = os.path.join(output_dir, 'parking_output.csv')
stat_output_file = os.path.join(output_dir, 'stat_data.xml')
trip_info_output_file = os.path.join(output_dir, 'tripinfo_data.xml')
vehroute_output_file = os.path.join(output_dir, 'vehroute_data.xml')

def get_simulation_start_time(xml_file):
    """Read the earliest departure time from a SUMO trip XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    min_depart = 1e10
    for trip in root.findall('trip'):
        depart_time = float(trip.get('depart', 0))
        if depart_time < min_depart:
            min_depart = depart_time
    return int(min_depart)

# Read the latest departure time and add a simulation buffer.
def get_simulation_end_time(xml_file, buffer_time=1800):
    """Read the latest departure time and add a simulation buffer."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    max_depart = 0
    for trip in root.findall('trip'):
        depart_time = float(trip.get('depart', 0))
        if depart_time > max_depart:
            max_depart = depart_time
    # Add a buffer after the latest departure.
    return int(max_depart + buffer_time)

simulation_start_time = 6 * 3600 # get_simulation_start_time(trips_file)
simulation_end_time = 15 * 3600 # 18 # get_simulation_end_time(trips_file)
print(f"Simulation time: {simulation_start_time} - {simulation_end_time}")

########################################## Simulation setup #############################################
def getNeighboringEdges(edge_id, net, radius=0.1):
    """Find nearby non-internal edges around both ends of an edge."""
    neighboring_edges = net.getEdge(edge_id).getShape()
    x_0, y_0 = neighboring_edges[0]
    x_last, y_last = neighboring_edges[-1]
    edges = net.getNeighboringEdges(x_0, y_0, r=radius) + net.getNeighboringEdges(x_last, y_last, r=radius)
    edge_ids = []
    for e, _ in edges:
        eid = e.getID()
        if eid and (not str(eid).startswith(":")):
            edge_ids.append(eid)
    return edge_ids
# Read edge ids from the SUMO network file.
tree = ET.parse(net_file)
root = tree.getroot()
xml_edges = []
# Collect edge tags.
for edge in root.findall('.//edge'):
    xml_edges.append(edge.get('id'))
# Read the zone-edge correspondence table.
zone_edges_df = pd.read_csv(zone_edge_file, header=0)

# Build an edge list for each zone.
zone2edge = {}
for i, col in enumerate(zone_edges_df.columns):
    # Keep only edges that actually exist in the SUMO network.
    zone2edge[i] = zone_edges_df[col].apply(lambda x: x if x in xml_edges else np.nan).dropna().tolist()
# Build the reverse edge-to-zone mapping.
edge2zone = {}
for zone_id, edges in zone2edge.items():
    for edge_id in edges:
        edge2zone[edge_id] = zone_id

# Create the additional.add.xml file.
df_park = pd.read_csv(parking_csv, encoding='utf-8', usecols=['name', 'address', 'lat', 'lon', 'capacity', 'edge_id'])
net = sumolib.net.readNet(net_file)
with open(additional_file, 'w') as f:
    f.write(f"""
<additional>
    <vType id="container" vClass="passenger" length="10.0" maxSpeed="20.0" color="1,0,0"/>
    <vType id="cable_car" vClass="passenger" length="15.0" maxSpeed="15.0" color="0,1,0"/>
    <vType id="subway" vClass="passenger" length="25.0" maxSpeed="30.0" color="0,0,1"/>
    <vType id="aircraft" vClass="passenger" length="50.0" maxSpeed="100.0" color="1,1,0"/>
    <vType id="wheelchair" vClass="passenger" length="1.0" maxSpeed="5.0" color="0,1,1"/>
    <vType id="scooter" vClass="passenger" length="2.0" maxSpeed="10.0" color="1,0,1"/>
    <vType id="drone" vClass="passenger" length="5.0" maxSpeed="50.0" color="0.5,0.5,0.5"/>
    <edgeData id="edgeData" file="{edge_output_file}" begin="{simulation_start_time}" end="{simulation_end_time}" freq="{period_average}"/>
""")
    
    parking_edge_ids = list()
    for index, row in df_park.iterrows():
        edge_id = row['edge_id']
        lane_id = edge_id + '_0'
        if edge_id not in parking_edge_ids:
            parking_edge_ids.append(edge_id)
            f.write(f"""
                    <parkingArea id="{edge_id}" name="{row['name']}" lane="{lane_id}" roadsideCapacity="{row['capacity']}" roadSide="right" angle="270" length="8"/>
                    """)
    
    for edge_id in parking_edge_ids:
        next_edges = getNeighboringEdges(edge_id, net, radius=20)
        next_next_edges = list()
        for next_id in next_edges:
            next_next_edges.extend(getNeighboringEdges(next_id, net, radius=20))
        connected_edges = list(set([edge_id] + next_edges + next_next_edges))
        f.write(f"""
            <rerouter id="Rerouter_{edge_id}" edges="{" ".join(connected_edges)}">
                <interval begin="{simulation_start_time}" end="{simulation_end_time}">
        """)
        for alt_edge_id in parking_edge_ids:
            if alt_edge_id != edge_id:
                f.write(f"""
                        <parkingAreaReroute id="{alt_edge_id}"/>
                """)
            else:
                f.write(f"""
                        <parkingAreaReroute id="{alt_edge_id}" visible="True"/>
                """)
        f.write(f"""
                </interval>
            </rerouter>
        """)
    f.write("""
</additional>
""")
print("Additional XML file created.")

# Create the SUMO configuration file.
sumo_config_file = f"{setting}.sumocfg"
with open(sumo_config_file, 'w') as f:
    f.write(f"""
    <configuration>
        <input>
            <net-file value="{net_file}"/>
            <route-files value="{routes_file}"/>
            <additional-files value="{additional_file}"/>
        </input>
        <time>
            <begin value="{simulation_start_time}"/>
            <end value="{simulation_end_time}"/>
            <step-length value="1.0"/>  <!-- 1-second simulation step -->
        </time>
        <processing>
            <time-to-teleport value="300"/> <!-- Teleport after 300 seconds if stuck -->
        </processing>
    </configuration>
    """)

if cui:
    def safe_rmtree(path: str, must_contain: str):
        p = Path(path).resolve()
        # Guard against deleting an unexpected directory.
        if must_contain not in str(p):
            raise ValueError(f"Refuse to delete suspicious path: {p}")
        if p.exists():
            shutil.rmtree(p)
    # Prepare output directories.
    base_out = Path(output_dir)
    base_out_str = str(base_out.resolve())

    # Recreate the output directory from a clean state.
    safe_rmtree(str(base_out), must_contain=str(Path(output_dir).resolve()))
    (base_out / "edge").mkdir(parents=True, exist_ok=True)
    (base_out / "mfd").mkdir(parents=True, exist_ok=True)
    (base_out / "parking").mkdir(parents=True, exist_ok=True)
    (base_out / "simulation_summary").mkdir(parents=True, exist_ok=True)
    summary_folder = str(base_out / "simulation_summary")
    
    # Command used to launch the SUMO simulation.
    sumo_cmd = [
        os.path.join(sumo_home, 'bin', 'sumo'),
        '-c', sumo_config_file,
        '--start',  # Start the simulation immediately.
        '--statistic-output', stat_output_file,
        '--tripinfo-output', trip_info_output_file,
        '--tripinfo-output.write-unfinished',
        '--vehroute-output', vehroute_output_file,
        '--vehroute-output.write-unfinished',
        '--vehroute-output.last-route',
        '--routing-algorithm', 'astar',
        "--device.rerouting.probability", "0.3",
        "--device.rerouting.period", "1800",  # Match the rerouting interval to the control interval.
        "--ignore-route-errors", "true",
        "--error-log", os.path.join(output_dir, "sumo.err.log"),
        "--message-log", os.path.join(output_dir, "sumo.msg.log"),
    ]

    # Run the SUMO simulation.
    #subprocess.run(sumo_cmd, check=True)
        
    unit_time = 600  # Record parking status every 10 minutes.
    sample_time = 60  # One-minute step used to match the aggregate MFD dynamics scale.
    # Read parking status through TraCI.
    traci.start(sumo_cmd)
    initial_dest = {}
    parking_vehicles = {}
    ex_parking_vehicles = set()
    searching_vehicles = {}
    for zone_id, edges in zone2edge.items():
        df_park.loc[df_park['edge_id'].isin(edges), 'zone_id'] = zone_id
        
    dest_neighbors_cache = {}
    def get_dest_neighbors(dest_edge_id, r=100):
        if dest_edge_id in dest_neighbors_cache:
            return dest_neighbors_cache[dest_edge_id]
        x, y = net.getEdge(dest_edge_id).getShape()[0]
        neighbors = {e[0].getID() for e in net.getNeighboringEdges(x, y, r=r)}
        dest_neighbors_cache[dest_edge_id] = neighbors
        return neighbors
    
    for step in range(simulation_start_time, simulation_end_time):
        # Advance the simulation by one step.
        traci.simulationStep()
        for veh_id in set(set(traci.vehicle.getIDList()) - ex_parking_vehicles):  # Vehicles already parked should not be counted as searching.
            if veh_id not in initial_dest:
                initial_dest[veh_id] = traci.vehicle.getRoute(veh_id)[-1]
            vehicle_position = traci.vehicle.getRoadID(veh_id)
            if not vehicle_position in edge2zone:
                continue
            
            destination_edge_id = traci.vehicle.getRoute(veh_id)[-1]
            # Get edges within 100m of the current destination edge.
            neighboring_edges = get_dest_neighbors(destination_edge_id, r=100)
            if traci.vehicle.isStoppedParking(veh_id):
                parking_vehicles[veh_id] = edge2zone[vehicle_position]
            elif ((vehicle_position in neighboring_edges) or (initial_dest[veh_id] != destination_edge_id)):
                if veh_id not in searching_vehicles:
                    searching_vehicles[veh_id] = set()
                searching_vehicles[veh_id].add(edge2zone[vehicle_position])
        
        if step % (60*30) == 0:
            print(f"Step {step}")
        if (step+1) % unit_time == 0:
            parking_success_vehicles = set(list(parking_vehicles.keys())) - ex_parking_vehicles  # Vehicles that newly parked during this interval.
            # Parking-area state.
            parking_occupancies = {}
            parking_occupancies = {parking_id: traci.parkingarea.getVehicleCount(parking_id) for parking_id in traci.parkingarea.getIDList()}
            df_park['occupancy'] = df_park['edge_id'].map(parking_occupancies)
            df_park['vacancy'] = df_park['capacity'] - df_park['occupancy']
            
            # Aggregate zone-level summary statistics.
            zone_vacancy = {zone_id: df_park.loc[df_park['zone_id']==zone_id, 'vacancy'].sum() for zone_id in zone2edge.keys()}
            zone_occupancy = {zone_id: df_park.loc[df_park['zone_id']==zone_id, 'occupancy'].sum() for zone_id in zone2edge.keys()}
            zone_searching_rate = {zone_id: sum([zone_id in searching_zone for searching_zone in searching_vehicles.values()]) * (sample_time / unit_time) for zone_id in zone2edge.keys()}
            zone_parking_success_rate = {zone_id: sum([parking_vehicles[v] == zone_id for v in parking_success_vehicles]) * (sample_time / unit_time) for zone_id in zone2edge.keys()}
            zone_sum_speed = {zone_id: [traci.edge.getLastStepMeanSpeed(edge) * 3.6 for edge in edges if traci.edge.getLastStepMeanSpeed(edge) is not None] for zone_id, edges in zone2edge.items()} # m/s -> km/h
            zone_avg_speed = {zone_id: sum(zone_sum_speed[zone_id]) / len(zone_sum_speed[zone_id]) if len(zone_sum_speed[zone_id]) > 0 else 0 for zone_id in zone2edge.keys()}
            
            # Save zone-level summary statistics.
            zone_statistics = []
            for zone_id in zone2edge.keys():
                zone_statistics.append({
                    'zone_id': zone_id,
                    'time': step,
                    'occupancy': zone_occupancy[zone_id],
                    'vacancy': zone_vacancy[zone_id],
                    'mean_speed': zone_avg_speed[zone_id],
                    'searching_vehicles': zone_searching_rate[zone_id],
                    'parking_success_rate': zone_parking_success_rate[zone_id]
                })
            zone_statistics_df = pd.DataFrame(zone_statistics)
            zone_statistics_df.to_csv(f"{summary_folder}/zone_statistics_summary_{step+1}.csv", index=False, encoding='utf-8-sig')
            
            # Collect zone-level parking-area details.
            zone_parking_data = []
            for zone_id in zone2edge.keys():
                zone_df_park = df_park.loc[df_park['zone_id']==zone_id]
                if not zone_df_park.empty:
                    zone_parking_data.append(zone_df_park)
            # Combine all zones into a single CSV.
            zone_parking_df = pd.concat(zone_parking_data)
            zone_parking_df.to_csv(f"{output_dir}/parking/zone_parking_summary_{step+1}.csv", index=False, encoding='utf-8-sig')

            ex_parking_vehicles = copy.copy(set(list(parking_vehicles.keys())))
            searching_vehicles.clear()
            
        
    traci.close()


    def process_chunk(chunk):
        chunk = pdx.flatten(chunk)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.pipe(pdx.flatten)
        chunk = chunk.rename({'@begin': 'begin', '@end': 'end',
                            'edge|@id': 'edge_id',
                            'edge|@sampledSeconds': 'sampledSeconds', 'edge|@density': 'density',
                            'edge|@laneDensity': 'laneDensity', 'edge|@speed': 'speed'}, axis=1)
        chunk = chunk.iloc[:, 1:]
        

        chunk['begin'] = chunk['begin'].astype(float)
        chunk['end'] = chunk['end'].astype(float)
        chunk["sampledSeconds"] = chunk["sampledSeconds"].astype(float)
        chunk["density"] = chunk["density"].astype(float)
        chunk["laneDensity"] = chunk["laneDensity"].astype(float)
        chunk["speed"] = chunk["speed"].astype(float)
        chunk = chunk.replace(np.NaN, 0)
        chunk['begin'] = chunk['begin'].astype(int)

        return chunk

    def plot_mfd(MD, MS, MF, ACCUMULATION, PRODUCTION):
        # Build a csv file
        Macro_Features = {'time': [time_interval * t/60 for t in range(len(MD))],
                        'density': MD,
                        'speed': MS,
                        'flow': MF,
                        'accumulation': ACCUMULATION,
                        'production': PRODUCTION
                        }
        mfd_edge = pd.DataFrame(Macro_Features)
        mfd_edge.to_csv(f"{output_dir}/mfd/MFD_edge.csv", index=False)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='density', y='speed', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Density (#veh/km)")
        plt.ylabel("Speed (Km/hr)")
        plt.title("V-K MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/v-k.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='accumulation', y='flow', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Accumulation (Veh)")
        plt.ylabel("Flow (Veh/hr)")
        plt.title("Q-K MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/q-k.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='accumulation', y='production', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Accumulation (Veh)")
        plt.ylabel(r"Production (Veh $\cdot$ km/hr)")
        plt.title("MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/mfd.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        mfd_edge.plot(x='speed', y='flow', kind='scatter', ax=ax, c='time', colormap='viridis')
        plt.xlabel("Speed (Km/hr)")
        plt.ylabel("Flow (Veh/hr)")
        plt.title("Q-V MFD based on edge data")
        plt.savefig(f"{output_dir}/mfd/q-v.png")
        plt.close()

    # calculating meandensity,meanflow,meanspeed (density=density)
    MD = []
    MS = []
    MF = []
    ACCUMULATION = []
    PRODUCTION = []
    i = 0
    context = ET.iterparse(edge_output_file, events=("end",), tag="interval")

    for event, elem in context:
        if elem.tag == 'interval':
            chunk = ET.tostring(elem, encoding='utf8').decode('utf8')
            df_chunk = pdx.read_xml(chunk, ['interval'])
            df = process_chunk(df_chunk)
            
            # calculation time interval
            bft = df.begin.iloc[0]
            eft = df.end.iloc[0]
            time_interval = int(eft - bft)

            # calculating total length of network
            # Exclude zero-density links from the MFD by setting their effective length to zero.
            length = df['sampledSeconds'] / (df['end']-df['begin']) / df['density'] # km
            df['Length'] = length.replace(np.NaN, 0).replace(np.inf, 0) # km
            _net = sum(df.Length)

            # Edie's definition
            # sampledSeconds (sum_k {tau_k}) = sum of the time spent by vehicles k within the time-space window, defined by link i and time interval [t, t + Δt]
            # link_accumulation (=l_i * K_i) : sampledSeconds / time_interval = average number of vehicles on link i during time interval [t, t + Δt]
            link_accumulation = (1/time_interval)*(df.sampledSeconds) # #veh
            # numofveh (sum_i {l_i * K_i}) : number of vehicles on the whole network during time interval [t, t + Δt]
            numofveh = sum(link_accumulation) # #veh
            # sampledSeconds * speed (sum_k {d_k}) = sum of the total distance traveled by vehicles k within the time-space window, defined by link i and time interval [t, t + Δt]
            # link_weighted_flow (=l_i * Q_i) : sampledSeconds * speed / time_interval = average flow on link i during time interval [t, t + Δt]
            link_weighted_flow = (1/time_interval) * (df.sampledSeconds * df.speed) * 3.6 # #veh*km/hr (3.6 = 3600/1000)
            # speedznumofveh (=sum_i {l_i * Q_i}) : total weighted flow on the whole network during time interval [t, t + Δt]
            speedznumofveh = sum(link_weighted_flow) # #veh*km/hr
            if numofveh > 0:
                meanspeed_ = speedznumofveh/numofveh # km/hr
            else:
                meanspeed_ = 0
            meandensity_ = numofveh / _net # #veh/km
            meanflow_ = speedznumofveh / _net # #veh/hr
            MD.append(meandensity_)
            MS.append(meanspeed_)
            MF.append(meanflow_)
            ACCUMULATION.append(numofveh)
            PRODUCTION.append(speedznumofveh)
            
            # edge data
            if i % (time_interval) == 0:
                edge_data = pd.DataFrame({'edge_id': df.edge_id,
                                        'density': (1/df.Length) * link_accumulation, # #veh/km
                                        'speed': df.speed * 3.6, # km/hr
                                        'flow': (1/df.Length) * link_weighted_flow, # #veh/hr
                                        'accumulation': link_accumulation, # #veh
                                        'link_weighted_flow': link_weighted_flow, # #veh*km/hr
                                        'length': df.Length # km
                                        })
                edge_data.to_csv(f"{output_dir}/edge/edge_data_{i}.csv", index=False)
            
        i += time_interval
        
        # Clear the processed element from memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        
    plot_mfd(MD, MS, MF, ACCUMULATION, PRODUCTION)
else:
    sumo_gui_cmd = [
        os.path.join(sumo_home, 'bin', 'sumo-gui'),
        '-c', sumo_config_file
    ]
    subprocess.run(sumo_gui_cmd, check=True)
