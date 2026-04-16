"""Run the network partitioning pipeline for several ``sigma_i`` settings."""

import os
import numpy as np

from network_partitioning import NetworkPartitioning

os.chdir(os.path.dirname(__file__))
for sigma_i in [2,3,4]:
    network_analysis = NetworkPartitioning(
        network_path="../sumo/network/data/output.geojson",
        flow_path="../sumo/output/random_DUE_veh14295_end7200_period0.5/edge/",
        range_start=10, # 0
        range_end=30, # 80
        sigma_i = sigma_i,
        output_dir_name=f"log_norm/sigma_i_{sigma_i}"
        )

    #network_analysis.display_density_statistics()
    #network_analysis.plot_density()
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
    
    # クラスターごとの平均トリップ長を計算して表示
    #vehroute_file = os.path.join('../sumo/output/random_DUE_veh14295_end7200_period0.5/vehroute_data.xml')
    #avg_trip_lengths = network_analysis.calculate_average_trip_length_per_cluster(best_merged_clusters, vehroute_file)
    #print("Average trip lengths per cluster:", avg_trip_lengths)
