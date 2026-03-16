from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from static import FEATURES
from utils import read_dataframe

SCALED_FEATURES = [f'{c}_n' for c in FEATURES]

def read_standarized_data() -> pd.DataFrame:
    iris_df = read_dataframe()
    # TODO: Normalize the features
    features = ['Sepal_length','Sepal_width','Petal_length','Petal_width']
    scaled_features = ['Sepal_length_n','Sepal_width_n','Petal_length_n','Petal_width_n']
    scaler = StandardScaler()
    iris_df[scaled_features] = scaler.fit_transform(iris_df[features])
    return iris_df

def run_elbow_method():
    standardized_df = read_standarized_data()
    mse_data = []
    # TODO: Run KMeans for k = 1 to 10 and calculate the MSE (inertia) for each k
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=5963)
        kmeans.fit(standardized_df[SCALED_FEATURES])
        mse_data.append({'k': k,'mse': kmeans.inertia_})
    mse_df = pd.DataFrame(mse_data)
    # TODO: Visualize the result with a line plot (k on x-axis and MSE on y-axis)
    fig = px.line(mse_df, x='k', y='mse',markers=True)
    fig.write_image('q1_part2.png')
def run_kmeans(k: int = 3):
    standardized_df = read_standarized_data()
    # TODO: Run KMeans with k clusters and get the cluster labels for each data point
    kmeans = KMeans(n_clusters=k, random_state=5963)
    clusters = kmeans.fit_predict(standardized_df[SCALED_FEATURES])
    standardized_df['cluster'] = [f'Cluster {c + 1}' for c in clusters]
    # TODO: Visualize the result with a scatter plot (Petal_length on x-axis and Sepal_length on y-axis, color by cluster)
    iris_figure = px.scatter(data_frame=standardized_df, x='Petal_length', y='Sepal_length', color='cluster')
    iris_figure.write_image('q1_part3.png')
if __name__ == '__main__':
    print('[Q1][Part 1] The normalized dataframe looks like this:')
    print(read_standarized_data().head())
    print('[Q1][Part 2] Plot a line chart to show how to find the best K using the Elbow method')
    run_elbow_method()
    print('[Q1][Part 3] Visualize K-means with k=3')
    run_kmeans(k=3)
