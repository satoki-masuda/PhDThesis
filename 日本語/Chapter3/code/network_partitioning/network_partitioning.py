"""Build zone partitions and cluster-level MFD summaries from SUMO edge data."""

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
import contextily as ctx
import cartopy.crs as ccrs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
import copy


class NetworkPartitioning:
    """Cluster a road network into homogeneous zones using edge-density similarity."""

    def __init__(self, network_path, flow_path, range_start, range_end, sigma_i, sigma_x=1, output_dir_name=None):
        """Load network and flow snapshots, then prepare merged edge-density data."""
        self.network_path = network_path
        self.flow_path = flow_path
        self.range_start = range_start
        self.range_end = range_end
        self.df_flow = pd.read_csv(self.flow_path + sorted(os.listdir(self.flow_path))[range_start])
        self.network = None
        self.G = None
        self.S = None
        self.edges = {}
        self.sigma_i = sigma_i
        self.sigma_x = sigma_x
        
        if output_dir_name:
            self.output_dir = os.path.join(os.path.dirname(__file__), output_dir_name) if output_dir_name else os.path.join(os.path.dirname(__file__), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else: 
            os.system(f"rm -rf {self.output_dir}/*")
            
                
        # 変数をテキストファイルで保存
        with open(os.path.join(self.output_dir, "variables.txt"), "w") as f:
            f.write(f"network_path: {network_path}\n")
            f.write(f"flow_path: {flow_path}\n")
            f.write(f"range_start: {range_start}\n")
            f.write(f"range_end: {range_end}\n")
            f.write(f"sigma_i: {sigma_i}\n")
            f.write(f"sigma_x: {sigma_x}\n")
            
        # merge_csv_files
        
        for i in range(self.range_start+1, self.range_end):
            df = pd.read_csv(self.flow_path + sorted(os.listdir(self.flow_path))[i])
            self.df_flow = self.df_flow.merge(df, on='edge_id', how='left', suffixes=('', f'_{i}'))
        self.df_flow = self.df_flow.replace([np.inf, -np.inf], np.nan)
        density_columns = [col for col in self.df_flow.columns if 'density' in col]
        self.df_flow['density_mean'] = self.df_flow[density_columns].mean(axis=1, skipna=True)
        self.df_flow = self.df_flow[['edge_id', 'density_mean']]
        
        # load_network
        self.network = gpd.read_file(self.network_path)
        self.network = self.network.merge(self.df_flow, left_on='id', right_on='edge_id', how='left')
        self.network = self.network.drop(columns=['edge_id'])
        
        # def extract_edges
        for _, row in self.network.iterrows():
            coords = row['geometry'].coords
            if len(coords) > 1:
                start = coords[0]
                end = coords[-1]
                edge = (tuple(start), tuple(end))
                self.edges[row['id']] = edge
        print("Number of edges in the original SUMO graph:", len(self.edges))
        
    def display_density_statistics(self):
        """Print basic summary statistics for the merged edge density."""
        print("min:", self.network.density_mean.min())
        print("5%:", self.network.density_mean.quantile(0.05))
        print("mean:", self.network.density_mean.mean())
        print("median:", self.network.density_mean.median())
        print("95%:", self.network.density_mean.quantile(0.95))
        print("max:", self.network.density_mean.max())
    
    def plot_density(self, show=False):
        """Plot mean edge density on the geographic network."""
        vmin = self.network.density_mean.quantile(0.25)
        vmax = self.network.density_mean.quantile(0.75)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        network_planar = self.network.to_crs(epsg=3857)
        network_planar.plot(column='density_mean', ax=ax, legend=True, cmap='viridis', vmin=vmin, vmax=vmax)
        # Background map
        ctx.add_basemap(ax, zoom=14, source=ctx.providers.CartoDB.Positron) # CartoDB.Voyager, CartoDB.Positron, OpenStreetMap.Mapnik
        plt.title('Density on Network')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/mean_density_map.png", bbox_inches='tight')
            plt.close()
    
    def create_graph(self):
        """Create the edge graph and its line graph used for partitioning."""
        self.df_flow['density_mean'] = self.df_flow['density_mean'].fillna(0)
        self.G = nx.MultiGraph()
        for edge_id, (start, end) in self.edges.items():
            self.G.add_weighted_edges_from(
                [(start, end, self.df_flow.loc[self.df_flow['edge_id'] == edge_id, 'density_mean'].values[0])],
                weight='density', id=edge_id
            )
        self._remove_small_clusters(edge_threshold=100)
        self._create_line_graph()
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
    
    def _calculate_weight(self, d_u, d_v):
        d_u = np.log(1 + d_u)
        d_v = np.log(1 + d_v)
        #d_u = d_u/100 if d_u < 20 else 20/100
        #d_v = d_v/100 if d_v < 20 else 20/100
        weight = np.exp(-((d_u - d_v) ** 2) / self.sigma_i)
        return weight
    
    def _create_line_graph(self):
        '''
        Create a line graph from a graph G. Line graph is a graph where the nodes represent the edges of the original graph G.
        
        Parameters
        ----------
        G : nx.Graph
            A graph object.
        
        Returns
        -------
        S : nx.Graph
            A line graph object.
        '''
        self.S = nx.line_graph(self.G)
        for s, t, x in self.G.edges:
            self.S.nodes[(s, t, x)]['id'] = self.G.edges[(s, t, x)]['id']
            self.S.nodes[(s, t, x)]['density'] = self.G.edges[(s, t, x)]['density']

        for u, v, key, data in self.S.edges(keys=True, data=True):
            d_u = self.S.nodes[u]['density']
            d_v = self.S.nodes[v]['density']
            data['weight'] = self._calculate_weight(d_u, d_v)

    def plot_line_graph_weights(self, show=False):
        weight_list = [self.S.get_edge_data(e[0], e[1])[0]['weight'] for e in self.S.edges]
        print("weight mean:", np.mean(weight_list))
        print("weight std:", np.std(weight_list))
        pd.Series(weight_list).hist(bins=100)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/line_graph_weights.png", bbox_inches='tight')
            plt.close()
        
    def calculate_all_pairs_shortest_paths(self):
        return dict(nx.all_pairs_shortest_path_length(self.G))
    
    def create_similarity_matrix(self, threshold, shortest_paths):
        '''
        Create a similarity matrix based on the spatial and attribute similarity between edges.
        
        Parameters
        ----------
        linegraph : nx.Graph
            Line graph of the original graph.
        shortest_paths : dict
            All pairs shortest paths between nodes.
        threshold : int
            Threshold for the similarity calculation.
        
        Returns
        -------
        similarity_matrix : np.ndarray
            Similarity matrix between edges.
        '''
        similarity_matrix = np.zeros((len(list(self.S.nodes)), len(list(self.S.nodes))))

        for edge_i, attr_i in self.S.nodes(data=True):
            i = list(self.S.nodes).index(edge_i)
            u_start, u_end, _ = edge_i
            d_u = attr_i['density']

            for edge_j, attr_j in self.S.nodes(data=True):
                if edge_i != edge_j:
                    j = list(self.S.nodes).index(edge_j)
                    v_start, v_end, _ = edge_j
                    d_v = attr_j['density']
                    shortest_path_length = shortest_paths[u_end].get(v_start, float('inf'))

                    if shortest_path_length < threshold:
                        spatial_weight = np.exp(- (shortest_path_length ** 2) / self.sigma_x)
                    else:
                        spatial_weight = 0

                    weight = self._calculate_weight(d_u, d_v) * spatial_weight
                    similarity_matrix[i, j] = weight

        return similarity_matrix
    
    def normalized_cut(self, similarity_matrix):
        '''
        Calculate the normalized cut of the graph using the similarity matrix
        
        Parameters
        ----------
        similarity_matrix: np.ndarray
            The similarity matrix of the graph
        
        Returns
        -------
        best_partition: np.ndarray
            The resulting partition of the graph
        
        Notes
        -----
        The normalized cut is calculated by solving the (generalized) eigenvalue problem of the Laplacian matrix and the degree matrix.
        Please read the paper "Normalized Cuts and Image Segmentation" by Shi and Malik (2000) for more details.
        '''
        # weighted degree matrix D
        D = np.diag(np.sum(similarity_matrix, axis=1))
        # Laplacian matrix L
        L = D - similarity_matrix
        # Add a small value to the diagonal of D to avoid division by zero
        D += np.eye(D.shape[0]) * 1e-10
        # Solve the generalized eigenvalue problem Lu = lambda D u
        _, eigvec = scipy.linalg.eigh(L, D) # eigh is used for the eigenvalue problem of symmetric matrices
        y = eigvec[:, 1]
        
        '''
        # Solve the normalized eigenvalue problem L_norm y = lambda y
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        _, eigvec = scipy.linalg.eigh(L_norm)
        y = np.diag(np.sqrt(np.diag(D))) @ np.array(eigvec[:,1])
        '''
        # find the best partition that minimizes the normalized cut
        partition_points = np.linspace(y.min(), y.max(), 100)
        min_Ncut = np.inf
        for point in partition_points:
            # calculate the Ncut between cluster A and B
            A = y <= point
            B = y > point
            cutAB = similarity_matrix[A, :][:, B].sum()
            cutAV = similarity_matrix[A, :].sum()
            cutBV = similarity_matrix[B, :].sum()
            Ncut = np.inf if cutAV == 0 or cutBV == 0 else cutAB / cutAV + cutAB / cutBV
            if Ncut < min_Ncut:
                min_Ncut = Ncut
                best_partition = A

        return best_partition
    
    def find_adjacent_clusters(self, clusters):
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
        for i, cluster in enumerate(clusters):
            cluster_adjacency[i] = set()
            for node in cluster:
                for neighbor in self.S.neighbors(node):
                    for j, other_cluster in enumerate(clusters):
                        if j != i and neighbor in other_cluster:
                            cluster_adjacency[i].add(j)
        return cluster_adjacency

    def calculate_NSk(self, A, B):
        '''
        Calculate the NS_k(A, B) of the cluster A and B
        
        Parameters
        ----------
        A: list
            The list of nodes in cluster A
        B: list
            The list of nodes in cluster B
        
        Returns
        -------
        NSk_A: float
        '''
        densities_A = np.array([self.S.nodes[node]['density'] for node in A])
        densities_B = np.array([self.S.nodes[node]['density'] for node in B])
        u_A = np.mean(densities_A)
        u_B = np.mean(densities_B)
        var_A = np.var(densities_A)
        var_B = np.var(densities_B)
        NSk_A = (2 * var_A) / (var_A + var_B + (u_A - u_B) ** 2)
        return NSk_A
    
    def calculate_NSk_total(self, clusters):
        total_NSk = 0
        adjacent_cluster_dict = self.find_adjacent_clusters(clusters)
        for i, A in enumerate(clusters):
            min_NSk = np.inf
            adjacent_clusters = adjacent_cluster_dict[i]
            if len(adjacent_clusters) == 0:
                continue
            else:
                for j, B in enumerate(clusters):
                    if j in adjacent_clusters:
                        NSk_A = self.calculate_NSk(A, B)
                        if NSk_A < min_NSk:
                            min_NSk = NSk_A
                total_NSk += min_NSk
        total_NSk /= len(clusters)
        
        return total_NSk
    
    def calculate_TV(self, clusters):
        '''
        Calculate the total density variance of the clusters
        
        Parameters
        ----------
        clusters: list
            The list of clusters of the graph (each cluster is a list of nodes)
        
        Returns
        -------
        total_variance: float
        '''
        TV = 0
        for cluster in clusters:
            densities = [self.S.nodes[node]['density'] for node in cluster]
            TV += np.var(densities) * len(cluster)
        return TV

    def recursive_bipartition(self, num_desired_clusters, threshold):
        '''
        Recursively bipartition the graph until the desired number of clusters is reached
        
        Parameters
        ----------
        graph: nx.Graph
            The graph object of the network
        num_desired_clusters: int
            The number of clusters to be created
        threshold: int
            The threshold for the similarity calculation. 
            If the threshold is 1, the similarity is based on the adjacent edges. 
            Otherwise, the similarity is based on the shortest path between edges.
        
        Returns
        -------
        best_clusters: list
            The best clusters based on the normalized cut
        max_clusters: list
            The clusters with the maximum number of clusters
        '''
        clusters = []
        clusters_dict = {}
        NS_dict = {}
        nodes_to_partition = list(self.S.nodes)
        if threshold > 1:
            shortest_paths = self.calculate_all_pairs_shortest_paths()
            similarity_matrix = self.create_similarity_matrix(threshold, shortest_paths)
            plt.hist(similarity_matrix[similarity_matrix > 0].flatten(), bins=100, range=(0, 1))
            plt.title('Histogram of Similarity index')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/similarity_index.png", bbox_inches='tight')
            plt.close()

        def bipartition_and_recurse(nodes):
            if len(clusters) >= num_desired_clusters:
                return
            subgraph = self.S.subgraph(nodes)
            weighted_adj_matrix = nx.adjacency_matrix(subgraph, weight='weight').todense()
            idx = [i for i, node in enumerate(nodes)]
            if threshold > 1:
                sub_similarity_matrix = similarity_matrix[idx, :][:, idx]
            elif threshold == 1:
                sub_similarity_matrix = weighted_adj_matrix
            else:
                raise ValueError("Threshold must be greater than 0")

            partition = self.normalized_cut(sub_similarity_matrix)
            part_A = [node for i, node in enumerate(subgraph.nodes) if partition[i]]
            part_B = [node for i, node in enumerate(subgraph.nodes) if not partition[i]]
            #print("part_A:", len(part_A))
            #print("part_B:", len(part_B))

            clusters.append(part_A)
            clusters.append(part_B)
            
            # NSを計算
            NS = self.calculate_NSk_total(clusters)
            NS_dict[len(clusters)] = NS
            clusters_dict[len(clusters)] = copy.deepcopy(clusters) # deep copy
            print("クラスタ数:", len(clusters))
            print("NS:", NS)

            if len(clusters) < num_desired_clusters:
                largest_cluster = max(clusters, key=len)
                clusters.remove(largest_cluster)
                bipartition_and_recurse(largest_cluster)

        bipartition_and_recurse(nodes_to_partition)
        print("NS_dict:", NS_dict)
        # NS_dictを記録
        with open(os.path.join(self.output_dir, "output.txt"), "w") as f:
            f.write("NS for initial partition\n")
            f.write(str(NS_dict))
            
        best_clusters = clusters_dict[min(NS_dict, key=NS_dict.get)]
        max_clusters = clusters_dict[max(NS_dict.keys())]
        return best_clusters, max_clusters

    def plot_clusters(self, clusters: list, suffix: str, show=False):
        if not os.path.exists(f"{self.output_dir}/clusters"):
            os.makedirs(f"{self.output_dir}/clusters")
        
        color_list = [
            'Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Pink',
            'Brown', 'Black', 'Gray', 'Cyan', 'Magenta', 'Lime',
            'Maroon', 'Navy', 'Olive', 'Teal', 'Aqua', 'Silver', 'Coral'
        ]
        colors = color_list[:len(clusters)]
        node_color_map = {}

        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                node_color_map[node] = colors[cluster_id]

        color_map = [node_color_map[edge] for edge in self.G.edges]
        pos = {node: (node[0], node[1]) for node in self.G.nodes()}

        plt.figure(figsize=(10, 10))
        nx.draw(self.G, pos, edge_color=color_map, with_labels=False, node_size=0, font_size=8)
        plt.title(f"{len(clusters)} clusters")
        # クラスター番号を記載
        for i, cluster in enumerate(clusters):
            #print(f"cluster_{i}: {cluster}")
            cluster_x = sum([(node[0][0] + node[1][0])/2 for node in cluster])/len(cluster)
            cluster_y = sum([(node[0][1] + node[1][1])/2 for node in cluster])/len(cluster)
            plt.text(cluster_x, cluster_y, f"{i}", fontsize=10, color='black')
        if show:
            plt.show()
        else:
            plt.savefig(f"{self.output_dir}/clusters/{suffix}_clusters.png", bbox_inches='tight')
            plt.close()
    
    def merge_clusters(self, clusters):
        """Greedily merge adjacent clusters and record candidate merged partitions."""
        NS_dict = {}
        clusters_dict = {}
        # NSを計算
        NS = self.calculate_NSk_total(clusters)
        NS_dict[len(clusters)] = NS
        clusters_dict[len(clusters)] = copy.deepcopy(clusters) # deep copy
        print("クラスタ数:", len(clusters))
        print("NS:", NS)
        
        while len(clusters) > 2:
            cluster_means = []
            for cluster in clusters:
                densities = [self.S.nodes[node]['density'] for node in cluster]
                cluster_means.append(np.mean(densities))
                
            min_distance = np.inf
            merge_idx = (0, 1)
            cluster_adjacency = self.find_adjacent_clusters(clusters)
            
            for i in range(len(clusters)):
                for j in cluster_adjacency[i]:
                    distance = np.abs(cluster_means[i] - cluster_means[j])
                    if distance < min_distance:
                        min_distance = distance
                        merge_idx = (i, j)

            i, j = merge_idx
            clusters[i].extend(clusters[j])
            del clusters[j]
            
            # NSを計算
            NS = self.calculate_NSk_total(clusters)
            NS_dict[len(clusters)] = NS
            clusters_dict[len(clusters)] = copy.deepcopy(clusters) # deep copy
            print("クラスタ数:", len(clusters))
            print("NS:", NS)
        
        print("NS_dict:", NS_dict)
        # NS_dictを記録
        with open(os.path.join(self.output_dir, "output.txt"), "a") as f:
            f.write("\nNS for merging\n")
            f.write(str(NS_dict))
        
        return clusters_dict#[min(NS_dict, key=NS_dict.get)]
    
    # クラスターごとのMFDの描画
    def plot_mfd_per_cluster(self, clusters, show=False, suffix=""):
        """
        Draw MFD（Macroscopic Fundamental Diagram）of each cluster.
        
        Parameters
        ----------
        clusters: list
            list of clusters (each cluster is a list of nodes)
        """
        if not os.path.exists(f"{self.output_dir}/MFD{suffix}"):
            os.makedirs(f"{self.output_dir}/MFD{suffix}")
        
        cluster_accumulation = {i : [] for i in range(len(clusters))}
        cluster_density = {i : [] for i in range(len(clusters))}
        cluster_production = {i : [] for i in range(len(clusters))}
        
        # list all csv files in the flow_path
        for file_path in os.listdir(self.flow_path):
            if file_path.endswith(".csv"):
                df = pd.read_csv(self.flow_path + file_path)
                df = df.replace([np.inf, -np.inf], np.nan)
                
                for cluster_id, cluster in enumerate(clusters):
                    mask = df['edge_id'].isin([self.G.edges[edge]['id'] for edge in cluster])
                    cluster_accumulation[cluster_id].append(df['accumulation'][mask].sum())
                    cluster_density[cluster_id].append(df['density'][mask].mean())
                    if df['length'][mask].sum() == 0:
                        cluster_production[cluster_id].append(0)
                    else:
                        cluster_production[cluster_id].append(df['link_weighted_flow'][mask].sum()) # #veh*km/hr    
        
        # Plotting MFD for each cluster
        num_clusters = len(clusters)
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
            
    def clusters2csv(self, clusters, suffix=""):
        """Save each cluster as a column of edge ids in CSV format."""
        if not os.path.exists(f"{self.output_dir}/clusters"):
            os.makedirs(f"{self.output_dir}/clusters")
                
        max_length = max(len(cluster) for cluster in clusters)
        cluster_dict = {f"cluster_{i}": [self.G.edges[edge]['id'] for edge in cluster] + [None]*(max_length - len(cluster)) for i, cluster in enumerate(clusters)}
        df = pd.DataFrame(cluster_dict)
        df.to_csv(f"{self.output_dir}/clusters/clusters{suffix}.csv", index=False)
    
    def fit_cubic_regression_per_cluster(self, clusters, show=False, suffix=""):
        """
        Fit a cubic regression (without constant term) for accumulation and production of each cluster.
        
        Parameters
        ----------
        clusters: list
            List of clusters (each cluster is a list of nodes)
        """
        if not os.path.exists(f"{self.output_dir}/CubicRegression{suffix}"):
            os.makedirs(f"{self.output_dir}/CubicRegression{suffix}")
        
        cluster_accumulation = {i: [] for i in range(len(clusters))}
        cluster_production = {i: [] for i in range(len(clusters))}
        
        for file_path in os.listdir(self.flow_path):
            if file_path.endswith(".csv"):
                df = pd.read_csv(self.flow_path + file_path)
                df = df.replace([np.inf, -np.inf], np.nan)
                
                for cluster_id, cluster in enumerate(clusters):
                    mask = df['edge_id'].isin([self.G.edges[edge]['id'] for edge in cluster])
                    cluster_accumulation[cluster_id].append(df['accumulation'][mask].sum())
                    if df['length'][mask].sum() == 0:
                        cluster_production[cluster_id].append(0)
                    else:
                        cluster_production[cluster_id].append(df['link_weighted_flow'][mask].sum() * (1 / df['length'][mask].sum()))
        
        num_clusters = len(clusters)
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

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # Usage
    # random_veh7054_end7200_period1.0, random_veh14004_end7200_period0.5, random_veh22783_end7200_period0.3, random_veh69146_end7200_period0.1
    # random_DUE_veh6953_end7200_period1.0, random_DUE_veh13772_end7200_period0.5, random_DUE_veh23184_end7200_period0.3, random_DUE_veh69657_end7200_period0.1
    sigma_i = 3.5
    network_analysis = NetworkPartitioning(
        network_path="../sumo/network/data/output.geojson",
        flow_path="../sumo/output/random_DUE_veh14295_end7200_period0.5/edge/",
        range_start=10, # 0
        range_end=30, # 79
        sigma_i = sigma_i,
        output_dir_name=f"log_norm/sigma_i_{sigma_i}"
        )

    network_analysis.display_density_statistics()
    network_analysis.plot_density()
    network_analysis.create_graph()
    network_analysis.plot_line_graph_weights()
    best_initial_clusters, max_initial_clusters = network_analysis.recursive_bipartition(num_desired_clusters=20, threshold=1)
    network_analysis.plot_clusters(clusters=best_initial_clusters, suffix="initial")
    
    merged_dict = network_analysis.merge_clusters(max_initial_clusters)
    #best_merged_clusters = merged_dict[min(merged_dict, key=merged_dict.get)]
    for key, clusters in merged_dict.items():
        network_analysis.plot_clusters(clusters=clusters, suffix=f"merged_{key}")
        network_analysis.plot_mfd_per_cluster(clusters, suffix=f"_{key}")
        network_analysis.clusters2csv(clusters, suffix=f"_{key}")
        network_analysis.fit_cubic_regression_per_cluster(clusters, suffix=f"_{key}")
    
