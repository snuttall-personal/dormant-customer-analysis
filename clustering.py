from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp


def number_clusters_vis(preference_matrix, min_clusters, max_clusters):
    # Silhouette Score - use this to determine optimal clustes - well separated and cohesive clusters. too many clusters means the clusters are not well separated.
    # too few means they are not cohesive -1 - 0 - 1, b(i)-a(i)/(max(a(i),b(i)))
    silhouette_scores = []
    for k in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(preference_matrix)
        silhouette_scores.append(silhouette_score(preference_matrix, labels))

    plt.figure(figsize=(6, 3))
    plt.plot(range(min_clusters, max_clusters), silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show() # we choose k = 8

    # Elbow Method
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(preference_matrix)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(6, 3))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show() # we choose k = 8


def kmeans_cluster(user_category_df, preference_matrix, optimal_k):
    # Apply K-Means clustering with the optimal number of clusters (k = 7) ?? why use KMeans? Easy to understand + computantionally efficient.
    # choose 8 random points, assign data to clusters based on euc distance. once all clusters decided, to mean of data points in each cluster as new centroid
    # repeat until convergence
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    user_clusters = kmeans.fit_predict(preference_matrix)

    # Add the cluster labels to the original dataframe
    user_category_df['cluster'] = user_category_df['ACCOUNT_ID'].map(dict(zip(preference_matrix.index, user_clusters)))

    # Aggregate preference scores by cluster and category
    cluster_category_preferences = user_category_df.groupby(['cluster', 'New Category'])['preference_score'].mean().unstack()
    # Normalise each row (cluster) to sum to 1
    cluster_category_preferences = cluster_category_preferences.div(cluster_category_preferences.sum(axis=1), axis=0)

    # Count the number of accounts in each cluster
    cluster_account_counts = user_category_df['ACCOUNT_ID'].groupby(user_category_df['cluster']).nunique()
    return cluster_category_preferences, cluster_account_counts


def cluster_count_vis(cluster_account_counts):
    # Plotting the number of accounts in each cluster
    plt.figure(figsize=(6, 3))
    cluster_account_counts.plot(kind='bar')
    plt.title('Number of Accounts in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Accounts')
    plt.xticks(rotation=0)
    plt.show()

def cluster_preferences_vis(cluster_category_preferences):
    # Plotting the normalised preference scores for each cluster individually
    num_clusters = cluster_category_preferences.shape[0]

    fig, axes = plt.subplots(num_clusters, 1, figsize=(14, num_clusters * 4), sharex=True)
    fig.suptitle('Normalised Average Preference Scores by Cluster and Category', fontsize=16)

    for i in range(num_clusters):
        cluster_preferences = cluster_category_preferences.iloc[i]
        axes[i].bar(cluster_preferences.index, cluster_preferences)
        axes[i].set_title(f'Cluster {cluster_category_preferences.index[i]}')
        axes[i].set_ylabel('Normalised Preference Score')
        axes[i].set_ylim(0, 1)

    plt.xlabel('New Category')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def ks_clustering_test(user_category_df):
    # Kolmogorov-Smirnov Test for Distribution Comparison
    def compare_distributions(cluster_data, overall_data):
        ks_stats = {}
        for cluster in cluster_data['cluster'].unique():
            cluster_scores = cluster_data[cluster_data['cluster'] == cluster]['preference_score']
            overall_scores = overall_data['preference_score']
            ks_stat, p_value = ks_2samp(cluster_scores, overall_scores)
            ks_stats[cluster] = {'ks_stat': ks_stat, 'p_value': p_value}
        return ks_stats

    # Compare distributions
    ks_stats = compare_distributions(user_category_df, user_category_df)
    for cluster in sorted(ks_stats.keys()):
        print(f"""Cluster: {cluster} {ks_stats[cluster]}""")


    # p-value, probability of obtaining ks_stat as extreme as one from cluster if we were sampling from the entire dataset.
    # All p-values are small, means is is highly unlikely null hypothesis is true, and that two samples come from the same distribution