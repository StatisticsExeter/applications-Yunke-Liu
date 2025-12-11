from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(df_scaled)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)  # adjust this value based on dendrogram scale
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _fit_dendrogram(df):
    """Given a dataframe containing only suitable values
    Return a scipy.cluster.hierarchy hierarchical clustering solution to these data"""
    tree = linkage(df, method = "ward")
    return tree


def _plot_dendrogram(df):
    """Given a dataframe df containing only suitable variables
    Use plotly.figure_factory to plot a dendrogram of these data"""
    fig = ff.create_dendrogram(
        df,
        linkagefun=lambda x: linkage(x, method="ward")
    )
    fig.update_layout(title="Interactive Hierarchical Clustering Dendrogram")
    return fig


def _cutree(tree, height):
    """Given a scipy.cluster.hierarchy hierarchical clustering solution and a float of the height
    Cut the tree at that hight and return the solution (cluster group membership) as a
    data frame with one column called 'cluster'"""
    labels = fcluster(tree, t = height, criterion="distance")
    clusters_df = pd.DataFrame({"cluster": labels})
    return clusters_df


def _pca(df):
    """Given a dataframe of only suitable variables
    return a dataframe of the first two pca predictions (z values) with columns 'PC1' and 'PC2'"""
    pca = PCA(n_components=2)
    comps = pca.fit_transform(df)
    df_pca = pd.DataFrame(comps, columns=["PC1", "PC2"])
    return df_pca


def _scatter_clusters(df):
    """Given a data frame containing columns 'PC1' and 'PC2' and 'cluster'
      (the first two principal component projections and the cluster groups)
    return a plotly express scatterplot of PC1 versus PC2
    with marks to denote cluster group membership"""
    fig = px.scatter(
       df,
       x = "PC1",
       y = "PC2",
       color = "cluster",
       title = "PCA Scatter Plot Colored by Cluster Labels"
    )
    return fig
  
  
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd

def dbscan_analysis():
    """Perform DBSCAN clustering on the collision dataset and produce a PCA scatter plot."""

    from course.utils import find_project_root
    base_dir = find_project_root()
    data_path = base_dir / "data_cache" / "la_collision.csv"

    # Load data
    df = pd.read_csv(data_path)

    # Standardise
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Run PCA for plotting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    df_plot = pd.DataFrame({
        "PC1": df_pca[:, 0],
        "PC2": df_pca[:, 1],
    })

    # DBSCAN clustering
    db = DBSCAN(eps=0.8, min_samples=5).fit(df_scaled)
    df_plot["cluster"] = db.labels_.astype(str)

    # Plot clusters
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="cluster",
        title="DBSCAN Clustering on PCA Projection"
    )

    # Save output
    outpath = base_dir / "data_cache" / "vignettes" / "unsupervised_classification" / "dbscan_scatter.html"
    fig.write_html(outpath)

    print(f"DBSCAN plot saved to {outpath}")
