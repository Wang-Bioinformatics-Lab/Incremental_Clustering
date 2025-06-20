import os
import pandas as pd
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Summarizing Falcon Results')
    parser.add_argument('falcon_clusters', help='falcon_clusters')
    parser.add_argument('output_summary_folder', help='output_summary_folder')
    args = parser.parse_args()

    clusterinfo_df = pd.read_csv(args.falcon_clusters, sep=',', comment='#')

    print(args)
    print(clusterinfo_df)

    clusterinfo_df = clusterinfo_df.sort_values(by='cluster', key=lambda x: x.replace(-1, np.inf))

    # Filtering out not in clusters data
    #clusterinfo_df = clusterinfo_df[clusterinfo_df["cluster"] != -1]

    # Grouping by cluster
    grouped_cluster_df = clusterinfo_df.groupby(["cluster"])
    # cluster_summary_list = []
    # for cluster, cluster_group_df in grouped_cluster_df:
    #     #TODO :Read these from mgf, as the representative is a medoid
    #
    #     cluster_count = len(cluster_group_df)
    #     cluster_mz = cluster_group_df["precursor_mz"].mean()
    #     cluster_rt = cluster_group_df["retention_time"].mean()
    #     cluster_charge = cluster_group_df["precursor_charge"].mean()
    #
    #     output_dict = {}
    #     output_dict["count"] = cluster_count
    #     output_dict["precursor_mz"] = cluster_mz
    #     output_dict["retention_time"] = cluster_rt
    #     output_dict["charge"] = cluster_charge
    #     output_dict["scan"] = cluster[0] + 1
    #
    #     cluster_summary_list.append(output_dict)
    #
    # # Creating a cluster summary
    # cluster_summary_df = pd.DataFrame(cluster_summary_list)
    # cluster_summary_df.to_csv(os.path.join(args.output_summary_folder, "cluster_summary.tsv"), sep='\t', index=False)

    # Creating cluster info
    clusterinfo_df["filename"] = clusterinfo_df["identifier"].apply(lambda x: x.split(":")[2] + ".mzML")
    clusterinfo_df["scan"] = clusterinfo_df["identifier"].apply(lambda x: x.split(":")[-1])
    clusterinfo_df.to_csv(os.path.join(args.output_summary_folder, "cluster_info.tsv"), sep='\t', index=False)

    #TODO: Rewriting MGF files

    # TODO: Maybe make this compatible with FBMN, since its already clustered.

if __name__ == "__main__":
    main()