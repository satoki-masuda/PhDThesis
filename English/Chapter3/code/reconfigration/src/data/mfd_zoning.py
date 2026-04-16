"""Utilities for zone definitions, MFD fitting, and derived Chapter 3 inputs."""

import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
#import contextily as ctx
import cartopy.crs as ccrs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
import copy
import sumolib
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score, mean_squared_error
from itertools import islice
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import Polygon, MultiPolygon, shape
from tqdm import tqdm
import json


class MFD_Zoning:
    """Load a zone partition and provide aggregated network-level preprocessing helpers."""

    def __init__(self, network_path, cluster_path):
        """Read the network and zoning CSV, then build helper graphs and mappings."""
        self.network_path = network_path
        self.network = None
        self.G = None
        self.S = None
        self.edges = {}
        
        # Load the network and cluster definitions.
        self.network = gpd.read_file(self.network_path)
        cluster_df = pd.read_csv(cluster_path, encoding='utf-8')
        self.clusters = [cluster_df[column].dropna().to_list() for column in cluster_df.columns]
        self.cluster2edge = {cluster_id: cluster_df[column].dropna().to_list() for cluster_id, column in enumerate(cluster_df.columns)}
        self.edge2cluster = {edge_id: cluster_id for cluster_id, cluster in enumerate(self.clusters) for edge_id in cluster}
        
        # Extract edges as start-end coordinate pairs.
        for _, row in self.network.iterrows():
            coords = row['geometry'].coords
            if len(coords) > 1:
                start = coords[0]
                end = coords[-1]
                edge = (tuple(start), tuple(end))
                self.edges[row['id']] = edge
        print("Number of edges in the original SUMO graph:", len(self.edges))
        
        self.output_dir = "../../output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.create_graph()
    
    def create_graph(self):
        self.G = nx.MultiGraph()
        for edge_id, (start, end) in self.edges.items():
            self.G.add_edges_from(
                [(start, end)], id=edge_id
            )
        self._remove_small_clusters(edge_threshold=100)
        self._create_line_graph()
        self.id2node_dict = {edge_id: (s, t, x) for (s, t, x), edge_id in self.S.nodes(data='id')}
        
        print("Created graphs successfully")
    
    def _remove_small_clusters(self, edge_threshold):
        connected_components = list(nx.connected_components(self.G))
        nodes_to_remove = []
        for component in connected_components:
            subgraph = self.G.subgraph(component)
            if subgraph.number_of_edges() <= edge_threshold:
                nodes_to_remove.extend(list(component))
        self.G.remove_nodes_from(nodes_to_remove)
        print("Number of nodes in the networkx graph:", len(list(self.G.nodes)))
        print("Number of edges in the networkx graph:", len(list(self.G.edges)))
        
    def _create_line_graph(self):
        """Create the line graph whose nodes correspond to edges in the base graph."""
        self.S = nx.line_graph(self.G)
        for s, t, x in self.G.edges:
            self.S.nodes[(s, t, x)]['id'] = self.G.edges[(s, t, x)]['id']
    
    def clean_zone_polygone(self, polygon_path):
        """Remove interior rings from zone polygons and save a cleaned GeoJSON."""
        # Read polygon data.
        polygons = gpd.read_file(polygon_path)

        # Replace each geometry with an outer-boundary-only version.
        for idx, feature in polygons.iterrows():
            geom = shape(feature["geometry"])
            
            # Handle MultiPolygon geometries.
            if geom.geom_type == "MultiPolygon":
                new_polygons = []
                for polygon in geom.geoms:
                    new_polygons.append(Polygon(polygon.exterior))
                new_geom = MultiPolygon(new_polygons)

            # Handle Polygon geometries.
            elif geom.geom_type == "Polygon":
                new_geom = Polygon(geom.exterior)
            
            else:
                new_geom = geom

            # Update the geometry in place.
            polygons.loc[idx, "geometry"] = new_geom

        # Save the cleaned GeoJSON.
        fixed_file_path = polygon_path.replace(".geojson", "_fixed.geojson")
        # Sort by cluster id before saving.
        polygons = polygons.sort_values("cluster")
        
        polygons.to_file(fixed_file_path, driver="GeoJSON")
        
        
    def calculate_all_pairs_shortest_paths(self):
        return dict(nx.all_pairs_shortest_path_length(self.G))
    
    def find_adjacent_clusters(self):
        '''
        Find the adjacent clusters of each cluster in the graph
        
        Parameters
        ----------
        clusters: list
            The list of clusters of the graph (each cluster is a list of nodes)
        
        Returns
        -------
        cluster_adjacency: dict
            The dictionary of the adjacent clusters of each cluster
        '''
        cluster_adjacency = {}            
        for i, cluster in enumerate(self.clusters):
            cluster_adjacency[i] = set()
            for edge_id in cluster:
                for neighbor in self.S.neighbors(self.id2node_dict[edge_id]):
                    for j, other_cluster in enumerate(self.clusters):
                        if j != i and self.S.nodes[neighbor].get('id') in other_cluster:
                            cluster_adjacency[i].add(j)
        
        return cluster_adjacency
    
    def find_paths(self, trip_lenth_file: pd.DataFrame = None):
        '''
        Find all paths between each pair of clusters
        
        Parameters
        ----------
        clusters: list
            The list of clusters of the graph (each cluster is a list of nodes)
        trip_lenth_file: pd.DataFrame
            The file containing the trip lengths within each cluster
        
        Returns
        -------
        path_dict: dict
            The dictionary of all paths between each pair of clusters
        '''
        adjacent_clusters = self.find_adjacent_clusters()
        num_zones = len(self.clusters)
        adj_matrix = np.array([[1 if j in adjacent_clusters[i] else 0 for j in range(num_zones)] for i in range(num_zones)])
        zone_graph = nx.from_numpy_array(adj_matrix)
        if trip_lenth_file is not None:
            # Set edge weights using average trip lengths when available.
            for i, edge in enumerate(zone_graph.edges):
                s,t = edge
                zone_graph.edges[edge]['weight'] = trip_lenth_file["avg_trip_length_km"].values[s] + trip_lenth_file["avg_trip_length_km"].values[t]
            
        
        def k_shortest_paths(G, source, target, k, weight=None):
            return list(
                islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
                )
        
        path_dict = {}
        for i, s_node in enumerate(zone_graph.nodes):
            for j, t_node in enumerate(zone_graph.nodes):
                if i != j:
                    path_dict[(i, j)] = []
                    for path in k_shortest_paths(zone_graph, s_node, t_node, 5):
                        path_dict[(i, j)].append((path, nx.path_weight(zone_graph, path, weight='weight')))
        
        return path_dict

    def plot_clusters(self, suffix: str, show=False):
        """Visualize the current cluster assignment on the network geometry."""
        if not os.path.exists(f"{self.output_dir}/clusters"):
            os.makedirs(f"{self.output_dir}/clusters")
        
        color_list = [
            'Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Pink',
            'Lime',  'Gray', 'Cyan', 'Magenta', 'Brown',
            'Maroon', 'Navy', 'Olive', 'Teal', 'Aqua', 'Silver', 'Coral', 'Black'
        ]
        colors = color_list[:len(self.clusters)]
        node_color_map = {}

        for cluster_id, cluster in enumerate(self.clusters):
            for edge_id in cluster:
                node = self.id2node_dict[edge_id]
                node_color_map[node] = colors[cluster_id]

        color_map = [node_color_map[edge] for edge in self.G.edges]
        pos = {node: (node[0], node[1]) for node in self.G.nodes()}

        plt.figure(figsize=(10, 10))
        nx.draw(self.G, pos, edge_color=color_map, with_labels=False, node_size=0, font_size=8)
        plt.title(f"{len(self.clusters)} clusters")
        # クラスター番号を記載
        for i, cluster in enumerate(self.clusters):
            #print(f"cluster_{i}: {cluster}")
            cluster_x = sum([(self.id2node_dict[edge_id][0][0] + self.id2node_dict[edge_id][1][0])/2 for edge_id in cluster])/len(cluster)
            cluster_y = sum([(self.id2node_dict[edge_id][0][1] + self.id2node_dict[edge_id][1][1])/2 for edge_id in cluster])/len(cluster)
            plt.text(cluster_x, cluster_y, f"{i}", fontsize=20, color='black')
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/clusters/{suffix}_clusters.png", bbox_inches='tight')
            plt.close()
    
    # クラスターごとのMFDの描画
    def plot_mfd_per_cluster(self, flow_output_dir_list, show=False, suffix=""):
        """
        Draw MFD（Macroscopic Fundamental Diagram）of each cluster.
        
        Parameters
        ----------
        clusters: list
            list of clusters (each cluster is a list of nodes)
        """
        if not os.path.exists(f"{self.output_dir}/MFD{suffix}"):
            os.makedirs(f"{self.output_dir}/MFD{suffix}")
        
        cluster_accumulation = {i : [] for i in range(len(self.clusters))}
        cluster_density = {i : [] for i in range(len(self.clusters))}
        cluster_production = {i : [] for i in range(len(self.clusters))}
        
        for flow_output_dir in flow_output_dir_list:
            for file_path in os.listdir(flow_output_dir):
                if file_path.endswith(".csv"):
                    df = pd.read_csv(os.path.join(flow_output_dir, file_path))
                    df = df.replace([np.inf, -np.inf], np.nan)
                    
                    for cluster_id, cluster in enumerate(self.clusters):
                        if f"zone{cluster_id}" in flow_output_dir or "zone" not in flow_output_dir:
                            mask = df['edge_id'].isin(cluster)
                            cluster_accumulation[cluster_id].append(df['accumulation'][mask].sum())
                            cluster_density[cluster_id].append(df['density'][mask].mean())
                            if df['length'][mask].sum() == 0:
                                cluster_production[cluster_id].append(0)
                            else:
                                cluster_production[cluster_id].append(df['link_weighted_flow'][mask].sum()) # #veh*km/hr                
        
        # Plotting MFD for each cluster
        num_clusters = len(self.clusters)
        num_cols = 3
        num_rows = (num_clusters + num_cols - 1) // num_cols
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        ax = ax.flatten()
        
        for i in range(num_clusters):
            ax[i].scatter(cluster_accumulation[i], cluster_production[i])
            ax[i].set_title(f'Cluster {i}')
            df_mfd_cluster = pd.DataFrame({
                'accumulation': cluster_accumulation[i],
                'production': cluster_production[i],
                'density': cluster_density[i]
            })
            df_mfd_cluster.to_csv(f"{self.output_dir}/MFD{suffix}/MFD_cluster_{i}.csv", index=False)
        
        for j in range(num_clusters, len(ax)):
            fig.delaxes(ax[j])
        
        fig.supxlabel('Accumulation (Veh)', fontsize=20)
        fig.supylabel(r"Production (Veh $\cdot$ km/hr)", fontsize=20)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/MFD{suffix}/MFD_per_cluster.png", bbox_inches='tight')
            plt.close()    
    
    def fit_cubic_regression_per_cluster(self, flow_output_dir_list, show=False, suffix=""):
        """
        Fit a cubic regression (without constant term) for accumulation and production of each cluster.
        
        Parameters
        ----------
        clusters: list
            List of clusters (each cluster is a list of nodes)
        """
        if not os.path.exists(f"{self.output_dir}/CubicRegression{suffix}"):
            os.makedirs(f"{self.output_dir}/CubicRegression{suffix}")
        
        cluster_accumulation = {i: [] for i in range(len(self.clusters))}
        cluster_production = {i: [] for i in range(len(self.clusters))}
        
        for flow_output_dir in flow_output_dir_list:
            for file_path in os.listdir(flow_output_dir):
                if file_path.endswith(".csv"):
                    df = pd.read_csv(os.path.join(flow_output_dir, file_path))
                    df = df.replace([np.inf, -np.inf], np.nan)
                    
                    for cluster_id, cluster in enumerate(self.clusters):
                        if f"zone{cluster_id}" in flow_output_dir or "zone" not in flow_output_dir:
                            mask = df['edge_id'].isin(cluster)
                            cluster_accumulation[cluster_id].append(df['accumulation'][mask].sum())
                            if df['length'][mask].sum() == 0:
                                cluster_production[cluster_id].append(0)
                            else:
                                cluster_production[cluster_id].append(df['link_weighted_flow'][mask].sum()) # #veh*km/hr
        
        num_clusters = len(self.clusters)
        num_cols = 3
        num_rows = (num_clusters + num_cols - 1) // num_cols
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        ax = ax.flatten()
        
        params_list = []
        
        for i in range(num_clusters):
            X = np.array(cluster_accumulation[i]).reshape(-1, 1)
            y = np.array(cluster_production[i])
            
            poly = PolynomialFeatures(degree=3, include_bias=False)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X_poly, y)
            
            X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_fit = model.predict(poly.transform(X_fit))
            
            ax[i].scatter(cluster_accumulation[i], cluster_production[i], label='Data')
            ax[i].plot(X_fit, y_fit, color='red', label='Cubic fit')
            ax[i].set_title(f'Cluster {i}')
            ax[i].set_xlabel('Accumulation (Veh)')
            ax[i].set_ylabel(r'Production (Veh $\cdot$ km/hr)')
            ax[i].legend()
            
            # 回帰結果を保存
            df_regression_cluster = pd.DataFrame({
                'accumulation': cluster_accumulation[i],
                'production': cluster_production[i]
            })
            df_regression_cluster.to_csv(f"{self.output_dir}/CubicRegression{suffix}/cluster_{i}_data.csv", index=False)
            pd.DataFrame({
                'accumulation': X_fit.flatten(),
                'fitted_production': y_fit
            }).to_csv(f"{self.output_dir}/CubicRegression{suffix}/cluster_{i}_fit.csv", index=False)
            
            # 回帰パラメータを保存
            params_list.append([i] + model.coef_.tolist())
        
        df_params = pd.DataFrame(params_list, columns=['cluster', 'coef1', 'coef2', 'coef3'])
        df_params.to_csv(f"{self.output_dir}/CubicRegression{suffix}/regression_params.csv", index=False)
        
        for j in range(num_clusters, len(ax)):
            fig.delaxes(ax[j])
        
        fig.suptitle('Cubic Regression per Cluster', fontsize=20)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/CubicRegression{suffix}/cubic_regression_per_cluster.png", bbox_inches='tight')
            plt.close()
    
    def fit_scaled_mfd_per_cluster(self, flow_output_dir_list, show=False, suffix=""):
        """
        Fit scaled-MFD parameters for each cluster from simulation outputs.
        
        Parameters
        ----------
        flow_output_dir_list : list
            List of directories that contain simulation outputs.
        show : bool, optional
            Whether to display the figure instead of saving it.
        suffix : str, optional
            Optional suffix appended to output filenames.
        """
        def scaled_mfd_model(N, alpha, beta, A_unit, B_unit, C_unit):
            """Scaled MFD functional form."""
            N_scaled = N / beta
            return alpha * (A_unit * N_scaled**3 + B_unit * N_scaled**2 + C_unit * N_scaled)


        # Unit MFD coefficients
        A_unit = 4.133e-11 * 3600
        B_unit = -8.282e-7 * 3600
        C_unit = 4.2e-3 * 3600
        N_jam_unit = 1.0e4
        N_critical = 3400
        Max_production = 6.3*3600
        
        if not os.path.exists(f"{self.output_dir}/ScaledMFD{suffix}"):
            os.makedirs(f"{self.output_dir}/ScaledMFD{suffix}")
        
        cluster_accumulation = {i: [] for i in range(len(self.clusters))}
        cluster_production = {i: [] for i in range(len(self.clusters))}
        
        for flow_output_dir in flow_output_dir_list:
            for file_path in os.listdir(flow_output_dir):
                if file_path.endswith(".csv"):
                    df = pd.read_csv(os.path.join(flow_output_dir, file_path))
                    df = df.replace([np.inf, -np.inf], np.nan)
                    for cluster_id, cluster in enumerate(self.clusters):
                        if f"zone{cluster_id}" in flow_output_dir or "zone" not in flow_output_dir:
                            mask = df['edge_id'].isin(cluster)
                            cluster_accumulation[cluster_id].append(df['accumulation'][mask].sum())
                            if df['length'][mask].sum() == 0:
                                cluster_production[cluster_id].append(0)
                            else:
                                cluster_production[cluster_id].append(df['link_weighted_flow'][mask].sum())# #veh*km/hr
        
        num_clusters = len(self.clusters)
        num_cols = 3
        num_rows = (num_clusters + num_cols - 1) // num_cols
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        ax = ax.flatten()
        
        params_list = []
        
        for i in range(num_clusters):
            X = np.array(cluster_accumulation[i])
            y = np.array(cluster_production[i])
            
            # Initial guesses for the scaling factors.
            initial_guess = [0.5, 0.5]
            
            # Fit the scaled MFD parameters with curve_fit.
            popt, _ = curve_fit(lambda N, alpha, beta: scaled_mfd_model(N, alpha, beta, A_unit, B_unit, C_unit),
                                X, y, p0=initial_guess, bounds=(0, [1.0, 1.0]))
            
            alpha_opt, beta_opt = popt
            coef1 = alpha_opt * A_unit * (1/beta_opt**3)
            coef2 = alpha_opt * B_unit * (1/beta_opt**2)
            coef3 = alpha_opt * C_unit * (1/beta_opt)
            N_jam_opt = N_jam_unit * beta_opt
            
            # Evaluate the fitted scaled MFD.
            X_fit = np.linspace(0, N_jam_opt, 100)
            y_fit = scaled_mfd_model(X_fit, alpha_opt, beta_opt, A_unit, B_unit, C_unit)
            
            ax[i].scatter(cluster_accumulation[i], cluster_production[i], label='Data')
            ax[i].plot(X_fit, y_fit, color='red', label='Scaled MFD Fit')
            ax[i].set_title(f'Cluster {i}')
            ax[i].set_xlabel('Accumulation (Veh)')
            ax[i].set_ylabel(r'Production (Veh $\cdot$ km/hr)')
            # Keep the axes anchored near the origin if desired.
            #ax[i].set_xlim(0, X.max())
            #ax[i].set_ylim(0, y.max())
            ax[i].legend()
            
            # Save raw points and the fitted curve.
            df_regression_cluster = pd.DataFrame({
                'accumulation': cluster_accumulation[i],
                'production': cluster_production[i]
            })
            df_regression_cluster.to_csv(f"{self.output_dir}/ScaledMFD{suffix}/cluster_{i}_data.csv", index=False)
            pd.DataFrame({
                'accumulation': X_fit.flatten(),
                'fitted_production': y_fit.flatten()
            }).to_csv(f"{self.output_dir}/ScaledMFD{suffix}/cluster_{i}_fit.csv", index=False)
            
            # Store the estimated scale factors.
            params_list.append([i, alpha_opt, beta_opt, N_jam_opt, coef1, coef2, coef3])
        
        df_params = pd.DataFrame(params_list, columns=['cluster', 'alpha', 'beta', 'N_jam', 'coef1', 'coef2', 'coef3'])
        df_params.to_csv(f"{self.output_dir}/ScaledMFD{suffix}/fitted_scale_factors.csv", index=False)
        
        for j in range(num_clusters, len(ax)):
            fig.delaxes(ax[j])
        
        fig.suptitle('Scaled MFD Fit per Cluster', fontsize=20)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/ScaledMFD{suffix}/scaled_mfd_fit_per_cluster.png", bbox_inches='tight', dpi=400)
            plt.close()    
    
    def parking_success_rate(self, output_dir_list, show=False, mono=False, edge_data_use=False, is_penalty=True):
        """
        Estimate parking-success / speed-penalty relationships by cluster.
    
        Parameters
        ----------
        simulation_output_dir: str
            Path to the output directory
        
        Returns
        -------
        params_list: list
        List of estimated parameters [Ap, eta1, eta2, eta3] for each cluster.
        """
        simulation_output_dir_list = [os.path.join(data_dir, "simulation_summary") for data_dir in output_dir_list]

        def model_func_cubic(X, Ap, eta1, eta2, eta3):
            search_veh, vacancy, avg_speed  = X
            return Ap * (search_veh ** eta1) * ((vacancy/100) ** eta2) * (avg_speed ** eta3)
        
        def model_func(params, X):
            Ap, eta1, eta2, eta3 = params
            search_veh, vacancy, avg_speed  = X
            return Ap * (search_veh ** eta1) * ((vacancy/100) ** eta2) * (avg_speed ** eta3)
        
        # Residual with an added penalty when predicted successes exceed searching vehicles.
        def residuals(params, X, y, penalty_weight=1e5):
            y_pred = model_func(params, X)
            # Fit to the observed values.
            res_main = y_pred - y
            # Penalize violations of y_pred <= search_veh.
            penalty = np.maximum(0, y_pred - X[0])
            # Append the weighted penalty term to the residual vector.
            return np.concatenate((res_main, penalty_weight * penalty))
        
        # One estimation dataframe per zone.
        df_estimate_dict = {i: [] for i in range(len(self.clusters))}

        for cluster_id in range(len(self.clusters)):
            df_estimate_dict[cluster_id] = pd.DataFrame({
                'search_veh': [0.0],
                'vacancy': [0.0],
                'avg_speed': [0.0],
                'parking_success_rate': [0.0]})
        
        if edge_data_use:
            col_num = df_estimate_dict[0].columns.get_loc('avg_speed')
            
            for output_dir in output_dir_list:
                simulation_output_dir = os.path.join(output_dir, "simulation_summary")
                flow_output_dir = os.path.join(output_dir, "edge")
                sim_zone = output_dir.split("/")[-1].split("_")[0]
                sim_zone = int(sim_zone.replace("zone", "")) if "zone" in sim_zone else None
                
                for file_path in [f for f in os.listdir(simulation_output_dir) if f.endswith(".csv")]:
                    df = pd.read_csv(os.path.join(simulation_output_dir, file_path))
                    sim_time = file_path.split("_")[-1].replace(".csv", "")
                    try:
                        edge_df = pd.read_csv(os.path.join(flow_output_dir, "edge_data_" + sim_time + ".csv"))
                        edge_df = edge_df.replace([np.inf, -np.inf], np.nan)
                    except:
                        continue
                    
                    if sim_zone is None:
                        for cluster_id in range(len(self.clusters)):
                            df_estimate_dict[cluster_id] = pd.concat([df_estimate_dict[cluster_id], pd.DataFrame({
                                'search_veh': [df.loc[df['zone_id']==cluster_id, 'searching_vehicles'].values[0]],
                                'vacancy': [df.loc[df['zone_id']==cluster_id, 'vacancy'].values[0]],
                                'avg_speed': [0],
                                'parking_success_rate': [df.loc[df['zone_id']==cluster_id, 'parking_success_rate'].values[0]]
                            })])
                            
                            cluster = self.cluster2edge[cluster_id]
                            mask = (edge_df['edge_id'].isin(cluster)) & (edge_df['accumulation'] > 0)
                            df_estimate_dict[cluster_id].iloc[-1, col_num] = edge_df['speed'][mask].mean()
                    else:
                        df_estimate_dict[sim_zone] = pd.concat([df_estimate_dict[sim_zone], pd.DataFrame({
                            'search_veh': [df.loc[df['zone_id']==sim_zone, 'searching_vehicles'].values[0]],
                            'vacancy': [df.loc[df['zone_id']==sim_zone, 'vacancy'].values[0]],
                            'avg_speed': [0],
                            'parking_success_rate': [df.loc[df['zone_id']==sim_zone, 'parking_success_rate'].values[0]]
                        })])
                        
                        cluster = self.cluster2edge[sim_zone]
                        mask = (edge_df['edge_id'].isin(cluster)) & (edge_df['accumulation'] > 0)
                        df_estimate_dict[sim_zone].iloc[-1, col_num] = edge_df['speed'][mask].mean()
        else:
            for simulation_output_dir in simulation_output_dir_list:
                sim_zone = simulation_output_dir.split("/")[-2].split("_")[0]
                sim_zone = int(sim_zone.replace("zone", "")) if "zone" in sim_zone else None
                for file_path in [f for f in os.listdir(simulation_output_dir) if f.endswith(".csv")]:
                    df = pd.read_csv(os.path.join(simulation_output_dir, file_path))
                    if sim_zone is None:
                        for i in range(len(self.clusters)):
                            df_estimate_dict[i] = pd.concat([df_estimate_dict[i], pd.DataFrame({
                                'vacancy': [df.loc[df['zone_id']==i, 'vacancy'].values[0]],
                                'avg_speed': [df.loc[df['zone_id']==i, 'mean_speed'].values[0]],
                                'search_veh': [df.loc[df['zone_id']==i, 'searching_vehicles'].values[0]],
                                'parking_success_rate': [df.loc[df['zone_id']==i, 'parking_success_rate'].values[0]]
                            })])
                    else:
                        df_estimate_dict[sim_zone] = pd.concat([df_estimate_dict[sim_zone], pd.DataFrame({
                            'vacancy': [df.loc[df['zone_id']==sim_zone, 'vacancy'].values[0]],
                            'avg_speed': [df.loc[df['zone_id']==sim_zone, 'mean_speed'].values[0]],
                            'search_veh': [df.loc[df['zone_id']==sim_zone, 'searching_vehicles'].values[0]],
                            'parking_success_rate': [df.loc[df['zone_id']==sim_zone, 'parking_success_rate'].values[0]]
                        })])
        
        params_list = []
        # Estimate one model per cluster.
        for i in range(len(self.clusters)):
            df_estimate = df_estimate_dict[i].dropna()
            X = df_estimate[['search_veh', 'vacancy', 'avg_speed']].values.T
            y = df_estimate['parking_success_rate'].values
            if is_penalty:
                initial_params = [1.0, 1.0, 1.0, 1.0]
                lower_bounds = [1e-5, 1e-5, 1e-5, 1e-5]
                upper_bounds = [1e5, 1e5, 1e5, 1e5]
                # Solve the penalized least-squares problem.
                result = least_squares(residuals, x0=initial_params, args=(X, y), bounds=(lower_bounds, upper_bounds), max_nfev=1000000)
                popt = result.x
                params_list.append(popt)
                y_pred = model_func(popt, X)
                R2 = r2_score(y, y_pred)
                RMSE = np.sqrt(mean_squared_error(y, y_pred))
            else:
                popt, _ = curve_fit(model_func_cubic, X, y, p0=[1.0, 1.0, 1.0, 1.0], maxfev=1000000, bounds=([1e-05, 1e-05, 1e-05, 1e-05], [1e05, 1e05, 1e05, 1e05]))
                params_list.append(popt)
                y_pred = model_func_cubic(X, *popt)
                R2 = r2_score(y, y_pred)
                RMSE = np.sqrt(mean_squared_error(y, y_pred))

            print(f"Cluster {i} - Estimated Parameters: Ap={popt[0]}, eta1={popt[1]}, eta2={popt[2]}, eta3={popt[3]}, R2={round(R2, 3)}, RMSE={round(RMSE, 3)}")
                        
            
        # Save the estimated parameters.
        if not os.path.exists(f"{self.output_dir}/ParkingSuccessRate"):
            os.makedirs(f"{self.output_dir}/ParkingSuccessRate")
        
        df_params = pd.DataFrame(params_list, columns=['Ap', 'eta1', 'eta2', 'eta3'])
        df_params.to_csv(f"{self.output_dir}/ParkingSuccessRate/estimated_params.csv", index=False)
        
        # contour plot the relationship between vacancy and searching vehicles count with respect to parking success rate
        if mono:
            # Create grayscale figures when requested.
            ncol = 3
            nrow = (len(self.clusters) + ncol - 1) // ncol
            fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow))
            ax = ax.flatten()
            
            scatter_list = []  # Keep a scatter handle for the colorbar.
            for i in range(len(self.clusters)):
                df_estimate = df_estimate_dict[i]
                X = df_estimate[['vacancy', 'search_veh']]
                y = df_estimate['parking_success_rate']

                # Scatter plot of observed points.
                alphas = y / y.max()  # Normalized alpha values.
                scatter = ax[i].scatter(X['vacancy'], X['search_veh'], c=y, cmap='gray_r')#, alpha=alphas)
                scatter_list.append(scatter)

                # Contour plot implied by the fitted parameters.
                #vacancy_range = np.linspace(X['vacancy'].min(), X['vacancy'].max(), 100)
                vacancy_range = np.linspace(0, X['vacancy'].max(), 100)
                #search_veh_range = np.linspace(X['search_veh'].min(), X['search_veh'].max(), 100)
                search_veh_range = np.linspace(0, X['search_veh'].max(), 100)
                V, S = np.meshgrid(vacancy_range, search_veh_range)

                Ap, eta1, eta2, eta3 = params_list[i]
                if is_penalty:
                    Z = model_func((Ap, eta1, eta2, eta3), np.array([V, S, np.ones_like(V) * df_estimate['avg_speed'].mean()]))
                else:
                    Z = model_func_cubic(np.array([V, np.ones_like(V) * df_estimate['avg_speed'].mean(), S]), Ap, eta1, eta2, eta3)

                contour_levels = np.linspace(Z.min(), Z.max(), 10)
                alphas_contour = (contour_levels - Z.min()) / (Z.max() - Z.min())  # Normalized contour alpha.
                for alpha, level in zip(alphas_contour, contour_levels):
                    contour = ax[i].contour(V, S, Z, levels=[level], colors='black', linestyles='dashed', alpha=alpha)
                    ax[i].clabel(contour, inline=True, fontsize=8, fmt=f'{level:.2f}')
                # Set common plotting ranges.
                ax[i].set_xlim(0, X['vacancy'].max())
                ax[i].set_ylim(0, X['search_veh'].max())
                ax[i].set_title(f'Region {i}')
                
                if i == 0:
                    ax[i].set_xlabel('Vacancy', fontsize=12)
                    ax[i].set_ylabel('Searching Vehicles', fontsize=12)
            
            # Delete any unused subplots.
            for j in range(len(self.clusters), len(ax)):
                fig.delaxes(ax[j])

            # Add the shared colorbar.
            cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Parking Success Rate')
            cbar.ax.yaxis.set_tick_params(labelsize=16)

            if show:
                plt.show()
            else:
                plt.savefig(f"{self.output_dir}/ParkingSuccessRate/parking_success_rate_mono.png", bbox_inches='tight')
                plt.close()


        else:
            ncol = 3
            nrow = (len(self.clusters) + ncol - 1) // ncol
            cmap = "coolwarm"
            yaxis_max = 100
            zaxis_max = 100 
            # Create one 3D figure per cluster.
            for i in range(len(self.clusters)):
                df_estimate = df_estimate_dict[i]
                X = df_estimate[['search_veh', 'vacancy', 'avg_speed']]
                y = df_estimate['parking_success_rate']

                contour_min = df_estimate['parking_success_rate'].min()
                contour_max = df_estimate['parking_success_rate'].max()

                search_veh_range = np.linspace(0, yaxis_max, 100)
                vacancy_range = np.linspace(0, X['vacancy'].max(), 100)
                avg_speed_range = np.linspace(0, zaxis_max, 100)
                S, V, AS = np.meshgrid(search_veh_range, vacancy_range, avg_speed_range)

                # --- Evaluate the fitted model ---
                
                if is_penalty:
                    values = model_func(params_list[i], np.array([S, V, AS]))
                else:
                    Ap, eta1, eta2, eta3 = params_list[i]
                    values = model_func_cubic(np.array([S, V, AS]), Ap, eta1, eta2, eta3)
                

                # --- Build the figure ---
                fig = go.Figure()
                
                # Add isosurfaces of the fitted response surface.
                fig.add_trace(go.Isosurface(
                    x=S.flatten(),
                    y=V.flatten(),
                    z=AS.flatten(),
                    value=values.flatten(),
                    isomin=contour_min,
                    isomax=contour_max,
                    surface_count=5,
                    #surface_fill=0.05,
                    colorscale="RdBu_r",
                    opacity=0.6,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showscale=False
                ))
                
                # Add observed points as a 3D scatter.
                fig.add_trace(go.Scatter3d(
                    x=X['search_veh'],
                    y=X['vacancy'],
                    z=X['avg_speed'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=y,
                        colorscale="RdBu_r",
                        opacity=0.8,
                        colorbar=dict(title='Parking Success Rate')
                    )
                ))

                # Update layout.
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Search Vehicles',
                        yaxis_title='Vacancy',
                        zaxis_title='Average Speed',
                        xaxis=dict(range=[0, yaxis_max]),
                        yaxis=dict(range=[0, X['vacancy'].max()]),
                        zaxis=dict(range=[0, zaxis_max])
                    )
                )

                # Show interactively or save to disk.
                if show:
                    #fig.show()
                    pass
                else:
                    #fig.show()
                    fig.write_html(f"{self.output_dir}/ParkingSuccessRate/Region_{i}.html")
                
            
            #### Vacancy-Search relation #######            
            for lower_speed, upper_speed in [(s, s+10) for s in range(0, 70, 10)]:
                fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow))
                ax = ax.flatten()
                for i in range(len(self.clusters)):
                    df_estimate = df_estimate_dict[i]
                    xaxis_max = df_estimate['vacancy'].max()
                    df_estimate = df_estimate[(df_estimate['avg_speed'] >= lower_speed) & (df_estimate['avg_speed'] < upper_speed)]

                    X = df_estimate[['search_veh', 'vacancy']]
                    y = df_estimate['parking_success_rate']
                    
                    # Scatter plot of the observed data.
                    scatter = ax[i].scatter(X['vacancy'], X['search_veh'], c=y, cmap=cmap, vmin=contour_min, vmax=contour_max)

                    # Contours implied by the fitted model.
                    search_veh_range = np.linspace(0, yaxis_max, 100)
                    vacancy_range = np.linspace(0, xaxis_max, 100)
                    S, V = np.meshgrid(search_veh_range, vacancy_range)
                    
                    
                    if is_penalty:
                        Z = model_func(params_list[i], np.array([S, V, np.ones_like(V) * ((lower_speed + upper_speed)/2)]))
                    else:
                        Ap, eta1, eta2, eta3 = params_list[i]
                        Z = model_func_cubic(np.array([S, V, np.ones_like(V) * ((lower_speed + upper_speed)/2)]), Ap, eta1, eta2, eta3)
                    # Filled contours using the same colormap.
                    contourf = ax[i].contourf(V, S, Z, cmap=cmap, alpha=0.3, vmin=contour_min, vmax=contour_max, levels=np.arange(0, contour_max+15, 10))
                    #contour = ax[i].contour(V, S, Z, cmap=cmap, linestyles='dashed', vmin=contour_min, vmax=contour_max, levels=np.arange(0, yaxis_max, 5))
                    contour = ax[i].contour(V, S, Z, linestyles='dashed', vmin=contour_min, vmax=contour_max, levels=np.arange(0, 200, 10), colors='black')
                    ax[i].clabel(contour, inline=True, fontsize=8)
                    # Set consistent plotting ranges.
                    ax[i].set_xlim(0, xaxis_max)
                    #ax[i].set_ylim(0, X['search_veh'].max())
                    ax[i].set_ylim(0, yaxis_max)
                    
                    ax[i].set_title(f'Region {i}')
                    if i == 0:
                        ax[i].set_xlabel('Vacancy', fontsize=12)
                        ax[i].set_ylabel('Searching Vehicles', fontsize=12)           
                # Delete any unused subplots.
                for j in range(len(self.clusters), len(ax)):
                    fig.delaxes(ax[j])
                # Add the shared colorbar.
                cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
                cbar.set_label('Parking Success Rate')
                cbar.ax.yaxis.set_tick_params(labelsize=16)           
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{self.output_dir}/ParkingSuccessRate/search_vacancy_{lower_speed}_{upper_speed}.png", bbox_inches='tight')
                    plt.close()
            
            
            #### Speed-Search relation #######
            #yaxis_max = contour_max*2
            yaxis_max = 100
            for lower_ratio, upper_ratio in [(v, v+0.1) for v in np.arange(0.0, 1.0, 0.1)]:
                fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow))
                ax = ax.flatten()
                for i in range(len(self.clusters)):
                    df_estimate = df_estimate_dict[i]
                    xaxis_max = df_estimate['avg_speed'].max()
                    
                    lower_vacancy = df_estimate['vacancy'].quantile(lower_ratio)
                    upper_vacancy = df_estimate['vacancy'].quantile(upper_ratio)
                    df_estimate = df_estimate[(df_estimate['vacancy'] >= lower_vacancy) & (df_estimate['vacancy'] < upper_vacancy)]
                    X = df_estimate[['search_veh', 'avg_speed']]
                    y = df_estimate['parking_success_rate']
                    
                    # Scatter plot of the observed data.
                    scatter = ax[i].scatter(X['avg_speed'], X['search_veh'], c=y, cmap=cmap, vmin=contour_min, vmax=contour_max)

                    # Contours implied by the fitted model.
                    search_veh_range = np.linspace(0, yaxis_max, 100)
                    avg_speed_range = np.linspace(0, xaxis_max, 100)
                    S, AS = np.meshgrid(search_veh_range, avg_speed_range)
                    
                    
                    if is_penalty:
                        Z = model_func(params_list[i], np.array([S, np.ones_like(S) * ((lower_vacancy + upper_vacancy)/2), AS]))
                    else:
                        Ap, eta1, eta2, eta3 = params_list[i]
                        Z = model_func_cubic(np.array([S, np.ones_like(S) * ((lower_vacancy + upper_vacancy)/2), AS]), Ap, eta1, eta2, eta3)
                    # Filled contours using the same colormap.
                    contourf = ax[i].contourf(AS, S, Z, cmap=cmap, alpha=0.3, vmin=contour_min, vmax=contour_max, levels=np.arange(0, contour_max+15, 10))
                    #contour = ax[i].contour(V, S, Z, cmap=cmap, linestyles='dashed', vmin=contour_min, vmax=contour_max, levels=np.arange(0, yaxis_max, 5))
                    contour = ax[i].contour(AS, S, Z, linestyles='dashed', vmin=contour_min, vmax=contour_max, levels=np.arange(0, 200, 10), colors='black')
                    ax[i].clabel(contour, inline=True, fontsize=8)
                    # Set consistent plotting ranges.
                    ax[i].set_xlim(0, xaxis_max)
                    #ax[i].set_ylim(0, X['search_veh'].max())
                    ax[i].set_ylim(0, yaxis_max)
                    
                    ax[i].set_title(f'Region {i}')
                    if i == 0:
                        ax[i].set_xlabel('Mean Speed', fontsize=12)
                        ax[i].set_ylabel('Searching Vehicles', fontsize=12)           
                # Delete any unused subplots.
                for j in range(len(self.clusters), len(ax)):
                    fig.delaxes(ax[j])
                # Add the shared colorbar.
                cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
                cbar.set_label('Parking Success Rate')
                cbar.ax.yaxis.set_tick_params(labelsize=16)           
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{self.output_dir}/ParkingSuccessRate/search_speed_{round(lower_ratio, 1)}_{round(upper_ratio, 1)}.png", bbox_inches='tight')
                    plt.close()
    
    def parse_vehroute_file(self, vehroute_file):
        """
        Parse the vehroute XML file to extract vehicle routes.
        
        Parameters
        ----------
        vehroute_file: str
            Path to the vehroute XML file
        
        Returns
        -------
        vehicle_routes: dict
            Dictionary of vehicle routes with vehicle ID as key and list of edge IDs as value
        """
        tree = ET.parse(vehroute_file)
        root = tree.getroot()
        vehicle_routes = {}
        
        for vehicle in root.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            route = vehicle.find('route')
            if route is not None:
                edges = route.get('edges').split()
                vehicle_routes[vehicle_id] = edges
        
        return vehicle_routes
    
    def calculate_average_trip_length_per_cluster(self, vehroute_file):
        """
        Calculate the average trip length for each cluster using vehicle routes from the vehroute file.
        
        Parameters
        ----------
        clusters: list
            List of clusters (each cluster is a list of nodes)
        vehroute_file: str
            Path to the vehroute XML file
        
        Returns
        -------
        average_trip_lengths: dict
            Dictionary of average trip lengths (km) for each cluster
        """
        vehicle_routes = self.parse_vehroute_file(vehroute_file)
        net = sumolib.net.readNet("../../data/raw/network.net.xml")
        cluster_trip_lengths = {i: [] for i in range(len(self.clusters))}
        
        for vehicle_id, route in vehicle_routes.items():
            current_cluster = None
            trip_length = 0

            for edge_id in route:
                if edge_id in self.edge2cluster:
                    cluster_id = self.edge2cluster[edge_id]
                    if current_cluster is None:
                        current_cluster = cluster_id

                    if cluster_id == current_cluster:
                        trip_length += net.getEdge(edge_id).getLength()
                    else:
                        if current_cluster is not None:
                            cluster_trip_lengths[current_cluster].append(trip_length)
                        current_cluster = cluster_id
                        trip_length = net.getEdge(edge_id).getLength()
                else:
                    if current_cluster is not None:
                        cluster_trip_lengths[current_cluster].append(trip_length)
                    current_cluster = None
                    trip_length = 0
            
            if current_cluster is not None:
                cluster_trip_lengths[current_cluster].append(trip_length)
        
        average_trip_lengths = {cluster_id: (round(sum(lengths) * (1/len(lengths)) * (1/1000), 4) if lengths else None) for cluster_id, lengths in cluster_trip_lengths.items()} # m to km
        
        return average_trip_lengths
    
    def shelter_capacity(self, shelter_file):
        """Attach shelter capacities to zones and save the processed table."""
        df_park = pd.read_csv(shelter_file, encoding='utf-8', usecols=['name', 'address', 'lat', 'lon', 'capacity', 'edge_id'])
        # ゾーンごとの駐車場情報を集計
        zone_parking_data = []
        for zone_id, edges in self.cluster2edge.items():
            zone_df_park = df_park[df_park['edge_id'].isin(edges)]
            if not zone_df_park.empty:
                zone_parking_data.append(zone_df_park.assign(zone_id=zone_id))
        zone_parking_df = pd.concat(zone_parking_data)
        zone_parking_df.to_csv("../../data/processed/shelter_list.csv", index=False, encoding='utf-8-sig')
    
    def max_boundary_capacity(self):
        """Estimate directional boundary capacities between neighboring zones."""
        # 各ゾーン間のエッジの本数を数えて、境界容量を計算
        max_boundary_capacity = np.zeros((len(self.clusters), len(self.clusters)))
        for i, cluster in enumerate(self.clusters):
            for edge_id in cluster:
                for neighbor in self.S.neighbors(self.id2node_dict[edge_id]):
                    for j, other_cluster in enumerate(self.clusters):
                        if j != i and self.S.nodes[neighbor].get('id') in other_cluster:
                            max_boundary_capacity[i][j] += 1000 # veh/h
        max_boundary_capacity = max_boundary_capacity / 60 # veh/min
        np.savetxt("../../data/processed/boundary_capacity.csv", max_boundary_capacity, delimiter=",")
        

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # Usage
    network_analysis = MFD_Zoning(
        network_path="../../data/raw/network.geojson",
        cluster_path="../../data/raw/zoning.csv",
        )

    # outputディレクトリにあるディレクトリのパスのリスト
    data_dir_list = os.listdir("../../../sumo/output")
    data_dir_list = [os.path.join("../../../sumo/output", data_dir) for data_dir in data_dir_list if (os.path.isdir(os.path.join("../../../sumo/output", data_dir)) and data_dir.startswith("zone"))]
    
    #network_analysis.plot_clusters(suffix="", show=False)
    
    flow_output_dir_list = [os.path.join(data_dir, "edge") for data_dir in data_dir_list]
    #network_analysis.plot_mfd_per_cluster(flow_output_dir_list=flow_output_dir_list, suffix="", show=False)
    
    # カーブフィッティング
    network_analysis.fit_scaled_mfd_per_cluster(flow_output_dir_list=flow_output_dir_list, suffix="", show=False)
    #network_analysis.fit_cubic_regression_per_cluster(flow_output_dir_list=flow_output_dir_list, suffix="", show=False)
    
    # クラスターごとの平均トリップ長を計算して表示 (現実的な避難時のODで計算する)
    #vehroute_file = os.path.join('../../../sumo/output/0.0normal_1.0evac/vehroute_data.xml')
    #avg_trip_lengths = network_analysis.calculate_average_trip_length_per_cluster(vehroute_file)
    #pd.DataFrame(avg_trip_lengths.items(), columns=['cluster', 'avg_trip_length_km']).to_csv("../../data/processed/average_trip_lengths.csv", index=False)
    
    # 避難所の容量を計算し出力
    #shelter_file = "../../data/raw/shelter_list.csv"
    #network_analysis.shelter_capacity(shelter_file)
    
    # 境界容量を計算
    #network_analysis.max_boundary_capacity()
    
    # vacancy, avg_speed, search_veh, parking_success_rateの関係
    data_dir_list = os.listdir("../../../sumo/output")
    data_dir_list = [os.path.join("../../../sumo/output", data_dir) for data_dir in data_dir_list if os.path.isdir(os.path.join("../../../sumo/output", data_dir))]
    #data_dir_list = [os.path.join("../../../sumo/output", data_dir) for data_dir in data_dir_list if os.path.isdir(os.path.join("../../../sumo/output", data_dir)) and data_dir.startswith("zone")]
    #data_dir_list = [os.path.join("../../../sumo/output", data_dir) for data_dir in data_dir_list if os.path.isdir(os.path.join("../../../sumo/output", data_dir)) and (data_dir.startswith("zone") or data_dir.startswith("random"))]
    
    #network_analysis.parking_success_rate(data_dir_list, show=False, mono=False, edge_data_use=False, is_penalty=True) # edge_data_use=Trueの場合はedge_dataを使用して平均速度を計算
    
    #polygon_path = "../../data/processed/zone_polygon.geojson"
    #network_analysis.clean_zone_polygone(polygon_path)
    
