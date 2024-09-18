import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.linear_model import LinearRegression


###################################################################### import dataset

df=pd.read_csv("https://raw.githubusercontent.com/ggkurklu/bill_authentication/refs/heads/main/bill_authentication.csv")

###################################################################### General Methods
def clean_dataset_(data):
    """
    Cleans the given DataFrame by handling missing values,
    removing duplicates, and filtering out outliers.
    """
    # handeling missing values in the dataset
    data.fillna(data.mean(), inplace=True)

    # removing any duplicates
    data.drop_duplicates(inplace=True)

    # using zscore to handle outliers
    numeric_df = data.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(numeric_df))

    # use 3 as zscore threshold
    threshold = 3
    outlier_mask = (z_scores > threshold).any(axis=1)

    # remove rows with outliers
    clean_df = data[~outlier_mask]

    # return cleaned dataset
    return clean_df

# method to visualise the datasets according to their plot type lable and title
def create_plot(plot_type, x=None, y=None, data=None, xlabel='', ylabel='', title='', filename='', color=None, marker='o', cmap=None, grid=True, **kwargs):
    plt.figure(figsize=kwargs.get('figsize', (8, 5)))

    if plot_type == 'bar':
        sns.barplot(x=x, y=y, data=data, color=color)

    elif plot_type == 'line':
        plt.plot(x, y, marker=marker, color=color)

    elif plot_type == 'scatter':
        plt.scatter(x, y, c=color, cmap=cmap, marker=marker)

    elif plot_type == 'sns_scatter':
        sns.scatterplot(x=x, y=y, data=data, color=color)

    elif plot_type == 'tree':
        plot_tree(kwargs['model'], feature_names=kwargs['feature_names'], class_names=kwargs['class_names'], filled=True, rounded=True)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if title:
        plt.title(title)

    if grid:
        plt.grid(True)

    # save plot filename
    if filename:
        plt.tight_layout()
        plt.savefig(filename)

    # show plot
    plt.show()

###################################################################### clean dataset
cleaned_df=clean_dataset_(df)
print(cleaned_df.head(100))

###################################################################### decision tree
# Features and target variable
X = cleaned_df.iloc[:, :-1]  # remove one column from the array
y = cleaned_df['Class']      # use Class real/fake bill as focus variable

# split data to training and testing
test_size = 0.3
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# start decision tree
tree_clf = DecisionTreeClassifier(random_state=random_state)
tree_clf.fit(X_train, y_train)

# predections data
y_pred = tree_clf.predict(X_test)

# accuracy, confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy Score: {accuracy:.2f}\n\n"
      f"Confusion Matrix:\n{conf_matrix}\n\n"
      f"Classification Report:\n{class_report}")

# Visualizing the Decision Tree
create_plot(
    plot_type='tree',
    xlabel='Decision Tree',
    ylabel='Class Labels',
    title='Decision Tree Visualization',
    filename='decision_tree.png',
    figsize=(20, 10),
    model=tree_clf,
    feature_names=X.columns.tolist(),
    class_names=['0', '1']
)

# visualize feature Importance Variance in this case
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': tree_clf.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

create_plot(
    plot_type='bar',
    x='Importance',
    y='Feature',
    data=feature_importance.sort_values('Importance', ascending=False),
    xlabel='Importance',
    ylabel='Feature',
    title='Feature Importance',
    filename='feature_importance.png',
    figsize=(10, 6)
)


###################################################################### KMEAN CLASTER

X = cleaned_df.values  # Convert DataFrame to NumPy array for KMeans
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

create_plot(
    plot_type='line',
    x=k_range,
    y=inertia,
    xlabel='Number of Clusters',
    ylabel='Inertia',
    title='Elbow Method for Optimal k',
    filename='elbow_method.png',
    marker='o',
    figsize=(8, 5)
)

# Determine the optimal number of clusters using the Silhouette Score
silhouette_avg = []
for k in k_range[1:]:  # Start from 2 because silhouette score is not defined for k=1
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    silhouette_avg.append(silhouette_score(X_scaled, clusters))

create_plot(
    plot_type='line',
    x=k_range[1:],
    y=silhouette_avg,
    xlabel='Number of Clusters',
    ylabel='Silhouette Score',
    title='Silhouette Score for Optimal k',
    filename='silhouette_score.png',
    marker='o',
    figsize=(8, 5)
)

# two possiblities 3 clusters according to elbow point and 5 according to silhouette
# n_clusters = 3
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
cleaned_df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_df['Variance'], cleaned_df['Skewness'], c=cleaned_df['Cluster'], cmap='viridis', marker='o')
plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.title('Clusters Visualization 5')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('clusters_visualization.png')
plt.show()

# Print cluster centers and DataFrame with cluster labels
print(f"Cluster Centers (scaled):\n\n"
      f"{kmeans.cluster_centers_}\n\n"
      f"\nCluster Centers (original scale):\n\n"
      f"\n{scaler.inverse_transform(kmeans.cluster_centers_)}\n\n"
      f"\nData with Cluster Labels:\n\n"
      f"{cleaned_df}")

######################################################################### classification algorithm


######################################################################### linear regression
# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred = lin_reg.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

sorted_idx = np.argsort(X_test['Variance'])
X_test_sorted = X_test['Variance'].iloc[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# Re-plot
plt.figure(figsize=(12, 6))
plt.scatter(X_test['Variance'], y_test, color='blue', label='True values')
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression line')
plt.title('Linear Regression: Variance vs Log Variance')
plt.xlabel('Variance')
plt.ylabel('Log Variance')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression_variance.png')
plt.show()

# Plotting residuals
plt.figure(figsize=(12, 6))
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.savefig('residuals_plot.png')
plt.show()
print("Residuals plot saved as 'residuals_plot.png'")