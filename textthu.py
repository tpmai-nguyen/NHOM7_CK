import streamlit as st
st.header("Nhóm 7")
st.write("""
1. Nguyễn Thị Phương Mai K224131544  
2. Nguyễn Hồ Minh Nguyệt K224131549  
3. Phạm Thị Hoài Thư K224131560
""")
st.header("Phân tích mô tả")
import pandas as pd
import requests
from io import StringIO

sheet_id = '1L8HOtCvDeGdtLOmWPKrF-5YtkR1ubX-4lnMcaoPZQdU'
sheet_name = 'Preprocessing data Export'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    st.write(df)
else:
    st.write("Lỗi khi tải dữ liệu")

st.subheader("Thông tin DataFrame")
st.write(df.dtypes)
st.subheader("Mô tả thống kê cho các biến số")
st.write(df.describe())
st.subheader("Số lượng giá trị null trong mỗi cột")
st.write(df.isnull().sum())
st.title("Xử lý giá trị thiếu trong DataFrame")

for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
        else:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

st.subheader("DataFrame sau khi xử lý")
st.write(df)
st.write(df.isnull().sum())
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
df.rename(columns=lambda x: x.replace("(", "").replace(")", ""), inplace=True)
st.write(df.info())
columns_to_drop = ["customer_email", "customer_fname", "customer_lname", "customer_id", 
                   "customer_password", "customer_street", "customer_city", "customer_state", 
                   "customer_zip", "customer_country"]

df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
st.write(df)
st.write(df.duplicated().sum())
st.write(df[df.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1)])
for i in df.columns:
    for j in range(len(df[i])):
        value = str(df[i][j])
        if '?' in value:
            df.at[j, i] = value.replace('?', '')
import matplotlib.pyplot as plt
st.subheader("Biểu đồ phân phối và hộp cho 'days_for_shipping_real")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
df[["days_for_shipping_real"]].hist(bins=50, ax=axes[0])
df[["days_for_shipping_real"]].boxplot(ax=axes[1], vert=False)
st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin

def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5) -> Tuple[float, float]:
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper

class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: float = 1.5):
        self.whisker = whisker_coeff
        self.lower = None
        self.upper = None

    def fit(self, X: pd.Series):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)

clipped_benefit_per_order = BoxplotOutlierClipper().fit_transform(df["days_for_shipping_real"])

st.subheader("Biểu đồ phân phối và hộp cho 'days_for_shipping_real' sau khi xử lý outliers")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

clipped_benefit_per_order.hist(bins=50, ax=axes[0])
clipped_benefit_per_order.to_frame().boxplot(ax=axes[1], vert=False)
st.pyplot(fig)

st.subheader("Biểu đồ phân phối và hộp cho 'days_for_shipment_scheduled'")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

df[["days_for_shipment_scheduled"]].hist(bins=50, ax=axes[0])
df[["days_for_shipment_scheduled"]].boxplot(ax=axes[1], vert=False)

st.pyplot(fig)

import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin

def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5) -> Tuple[float, float]:
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper

class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: float = 1.5):
        self.whisker = whisker_coeff
        self.lower = None
        self.upper = None

    def fit(self, X: pd.Series):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)

clipped_order_profit_per_order = BoxplotOutlierClipper().fit_transform(df["days_for_shipment_scheduled"])

st.subheader("Biểu đồ phân phối và hộp cho 'days_for_shipment_scheduled' sau khi xử lý outliers")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
clipped_order_profit_per_order.hist(bins=50, ax=axes[0])
clipped_order_profit_per_order.to_frame().boxplot(ax=axes[1], vert=False)
st.pyplot(fig)

df_1=df.copy()
import streamlit as st
import seaborn as sns

st.subheader('Market Distribution')
fig, ax = plt.subplots()
df['market'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax)
ax.set_ylabel('')
ax.set_title('Market Distribution')
st.pyplot(fig)

st.subheader('Customer Segment Distribution')
fig, ax = plt.subplots()
df['customer_segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax)
ax.set_ylabel('')
ax.set_title('Customer Segment Distribution')
st.pyplot(fig)

import seaborn as sns

st.subheader('Count of Each Category Name')
category_counts = df['category_name'].value_counts()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='coolwarm', ax=ax)
ax.set_title('Count of Each Category Name')
ax.set_xlabel('Category Name')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

st.subheader('Top 10 Most Common Products')
product_counts = df['product_name'].value_counts().nlargest(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=product_counts.index, y=product_counts.values, palette='coolwarm', ax=ax)
ax.set_title('Top 10 Most Common Products')
ax.set_xlabel('Product Name')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

from sklearn.preprocessing import LabelEncoder

object_columns = df.select_dtypes(include=['object']).columns.tolist()

for column in object_columns:
    df[column] = df[column].astype(str)
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])

st.write(df.info())

df['late_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
df['late_days']

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader('Distribution of Days for Shipping (real)')
fig, ax = plt.subplots(figsize=(14, 6))
sns.histplot(df['days_for_shipping_real'], kde=True, color='skyblue', ax=ax)
ax.set_title('Distribution of Days for Shipping (real)')
ax.set_xlabel('days_for_shipping_real')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.subheader('Distribution of Days for Shipment (scheduled)')
fig, ax = plt.subplots(figsize=(14, 6))
sns.histplot(df['days_for_shipment_scheduled'], kde=True, color='skyblue', ax=ax)
ax.set_title('Distribution of Days for Shipment (scheduled)')
ax.set_xlabel('days_for_shipment_scheduled')
ax.set_ylabel('Frequency')
st.pyplot(fig)

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

st.write("Correlation Matrix:")
st.write(correlation_matrix)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8}, fmt=".2f", ax=ax)
ax.set_title('Correlation Heatmap')

st.pyplot(fig)
st.header("Phân tích dự đoán")
st.subheader('Rủi ro giao hàng trễ')

existing_columns = [col for col in ['type', 'days_for_shipment_scheduled', 'delivery_status', 'late_delivery_risk', 'category_id',
              'customer_city', 'customer_country', 'customer_segment', 'customer_state', 'latitude', 'longitude',
              'order_country', 'order_city', 'order_item_product_price', 'order_item_quantity', 'order_status',
              'product_card_id', 'product_price', 'shipping_date_dateorders', 'shipping_mode', 'late_days',
              'order_date_dateorders', 'order_region', 'market'] if col in df.columns]
df_copy = df[existing_columns]

c = df.corr()  

if 'late_delivery_risk' in c.columns:
    st.write(c['late_delivery_risk'])
else:
    st.write("Column 'late_delivery_risk' not found in correlation matrix.")

if 'delivery_status' in df_1.columns:
    delivery_status_data = df_1['delivery_status']
    delivery_status_summary = delivery_status_data.value_counts()

    st.write("Delivery Status Summary:")
    st.write(delivery_status_summary)
else:
    st.write("Column 'delivery_status' not found in `df_1`.")


plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.countplot(data=df_1, x='delivery_status', order=delivery_status_summary.index, palette='coolwarm')
plt.title('Delivery Status Distribution')
plt.xlabel('Delivery Status')
plt.ylabel('Count')
st.pyplot(plt.gcf())

df_copy['late_days'].value_counts().plot.pie(legend = ["0", "1"])

delivery_status_data = df_1['delivery_status']
shipping_mode_data = df_1['shipping_mode']

cross_tab = pd.crosstab(shipping_mode_data, delivery_status_data)
cross_tab_percent = cross_tab.div(cross_tab.sum(1), axis=0) * 100

st.write("Cross-Tabulation (Contingency Table) with Percentages:")
st.write(cross_tab_percent)

st.subheader('Delivery Status by Shipping Mode (Percentage)')
fig, ax = plt.subplots(figsize=(12, 8))
cross_tab_percent.plot(kind="bar", stacked=True, colormap='Set2', ax=ax)
ax.set_title('Delivery Status by Shipping Mode (Percentage)')
ax.set_xlabel('Shipping Mode')
ax.set_ylabel('Percentage')
st.pyplot(fig)

late_delivery_data = df_1[df_1['delivery_status'] == 'Late delivery']
late_by_product = late_delivery_data['category_name'].value_counts().nlargest(10).reset_index()

plt.figure(figsize=(12, 6))

st.write("Columns in late_by_product:", late_by_product.columns.tolist())
if 'index' in late_by_product.columns and 'category_name' in late_by_product.columns:
    sns.barplot(x=late_by_product['index'], y=late_by_product['category_name'], palette='coolwarm')
else:
    st.write("The required columns ('index', 'category_name') are not found in the 'late_by_product' DataFrame.")
plt.xlabel('Category Name')
plt.ylabel('Quantity')
plt.xticks(rotation=45, ha='right')

st.pyplot(plt)

group = df_1.groupby(['market', 'delivery_status']).market.count().unstack()

plt.figure(figsize=(10, 6))
group.plot(kind='bar', ax=plt.gca())
plt.title('Delivery Status by Market')
plt.xlabel('Market')
plt.ylabel('Count')

st.pyplot(plt)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

target1 = 'late_delivery_risk'
if target1 not in df_1.columns:
    raise ValueError(f"The target variable '{target1}' is not present in the dataset.")

from imblearn.over_sampling import SMOTE

X = df_copy.drop(target1, axis=1)
y = df_copy[target1]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

y = y_balanced
X = X_balanced

class_counts = pd.Series(y_balanced).value_counts()

st.write("Số lượng lớp 0 sau khi cân bằng:", class_counts[0])
st.write("Số lượng lớp 1 sau khi cân bằng:", class_counts[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression()),
    ("KNeighbors", KNeighborsClassifier())
]

output = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

for model_name, model in models:
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    precision_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')
    recall_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    output['Model'].append(model_name)
    output['Accuracy'].append(np.mean(accuracy_scores))
    output['Precision'].append(np.mean(precision_scores))
    output['Recall'].append(np.mean(recall_scores))
    output['F1-score'].append(np.mean(f1_scores))

output_df = pd.DataFrame(output)

st.title("Model Performance Evaluation")
st.write("Here are the performance metrics for various machine learning models:")

st.dataframe(output_df)

from sklearn.decomposition import PCA
from sklearn.model_selection import  KFold, cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd

pca = PCA(n_components=0.8)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression()),
    ("KNeighbors", KNeighborsClassifier())
]

output = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}
for model_name, model in models:
    model.fit(X_train_pca, y_train)
    accuracy_scores = cross_val_score(model, X_test_pca, y_test, scoring='accuracy')
    precision_scores = cross_val_score(model, X_test_pca, y_test, scoring='precision')
    recall_scores = cross_val_score(model, X_test_pca, y_test, scoring='recall')
    f1_scores = cross_val_score(model, X_test_pca, y_test, scoring='f1')
    
    output['Model'].append(model_name)
    output['Accuracy'].append(np.mean(accuracy_scores))
    output['Precision'].append(np.mean(precision_scores))
    output['Recall'].append(np.mean(recall_scores))
    output['F1-score'].append(np.mean(f1_scores))

output_df = pd.DataFrame(output)

st.write(output_df)

def frequency_encoding(df, column):
    frequency_map = df[column].value_counts(normalize=True)
    df[column + '_freq_encoded'] = df[column].map(frequency_map)
    return df

instances_to_predict = pd.DataFrame({
    'Type': ['DEBIT','DEBIT'],
    'Days for shipment (scheduled)': [5,6],
    'Delivery Status': ['Advance shipping','Advance shipping'],
    'Category Id': [17,17],
    'Category Name': ['Cleats','Cleats'],
    'Customer City': ['Los Angeles','Los Angeles'],
    'Customer Country': ['EE. UU.', 'EE. UU.'],
    'Customer Segment': ['Corporate','Corporate'],
    'Customer State': ['NY','NY'],
    'Latitude': [17.24253835,17.24253835],
    'Longitude': [-65.03704823,-65.03704823],
    'Order City': ['Bikaner','Bikaner'],
    'Order Country': ['India','India'],
    'Order Item Product Price': [327.75,327.75],
    'Order Item Quantity': [2,2],
    'Order Status': ['COMPLETE','COMPLETE'],
    'Product Card Id': [1360,1360],
    'Product Price': [327.75,327.75],
    'Order Region': ['Southeast Asia','Southeast Asia'],
    'Market': ['Asia','Asia']
})

columns_to_encode = ['Type', 'Delivery Status','Category Name','Customer City','Customer Country','Customer Segment',
                    'Customer State','Order City','Order Country','Order Status','Order Region','Market']

for column in columns_to_encode:
    instances_to_predict = frequency_encoding(instances_to_predict, column)

instances_to_predict.drop(columns=columns_to_encode, inplace=True)

scaler = StandardScaler()
instances_to_predict_scaled = scaler.fit_transform(instances_to_predict)

X_train = np.random.rand(10, instances_to_predict_scaled.shape[1])
y_train = np.random.randint(0, 2, 10)  # Random binary labels

gb_model = XGBClassifier()
gb_model.fit(X_train, y_train)

predictions = gb_model.predict(instances_to_predict_scaled)

st.title("Late Delivery Prediction")

st.write("### Input Data (Instances to Predict):")
st.write(instances_to_predict)
st.write("### Predictions:")
st.write("Case 1: Predicted Late Delivery:", "Yes" if predictions[0] == 1 else "No")
st.write("Case 2: Predicted Late Delivery:", "Yes" if predictions[1] == 1 else "No")

st.subheader("Dự đoán doanh số mỗi khách hàng tạo ra")

df_copy2=df[["sales","delivery_status","category_id","customer_city","customer_country","customer_segment","customer_state","order_country","order_city",
            "order_item_product_price","order_item_quantity","order_status","product_card_id","product_price","order_region","market"]]

c = df.corr()

st.write("Correlation with 'sales':")
st.write(c['sales'])

df['order_date_dateorders'] = pd.to_datetime(df['order_date_dateorders'])

plt.figure(figsize=(14, 6))
df.groupby('order_date_dateorders')['sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')

st.pyplot(plt)

market_sales = df.groupby('market')['sales'].sum()

top_n_markets = market_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_n_markets.plot(kind='bar')
plt.title('Top Markets by Total Sales')
plt.xlabel('Market')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(plt)

plt.figure(figsize=(12, 6))
sns.barplot(x='category_name', y='sales', data=df_1)
plt.title('Doanh thu theo loại hàng')
plt.xlabel('Loại hàng')
plt.ylabel('Doanh thu')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(plt)

target2 = 'sales'
if target2 not in df_1.columns:
    raise ValueError(f"The target variable '{target2}' is not present in the dataset.")


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

X = df_copy2.drop(target2, axis=1)
y = df_copy2[target2]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

models = {
    'Multiple Linear Regression': LinearRegression(),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Support Vector Regression': SVR(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X2_train, y2_train)
    y2_pred = model.predict(X2_test)

    mse = mean_squared_error(y2_test, y2_pred)
    r_square = r2_score(y2_test, y2_pred)
    results[name] = {'MSE': mse, 'R^2': r_square}

st.title("Model Performance Evaluation")

for model_name, metrics in results.items():
    st.subheader(f"{model_name} performance:")
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")

def frequency_encoding(df, column):
    frequency_map = df[column].value_counts(normalize=True)
    df[column + '_freq_encoded'] = df[column].map(frequency_map)
    return df

instances_to_predict = pd.DataFrame({
    'Delivery Status': ['Advance shipping','Advance shipping'],
    'Category Id': [17,17],
    'Customer City': ['Los Angeles','Los Angeles'],
    'Customer Country': ['EE. UU.', 'EE. UU.'],
    'Customer Segment': ['Corporate','Corporate'],
    'Customer State': ['NY','NY'],
    'Order City': ['Bikaner','Bikaner'],
    'Order Country': ['India','India'],
    'Order Item Product Price': [327.75,327.75],
    'Order Item Quantity': [2,2],
    'Order Status': ['COMPLETE','COMPLETE'],
    'Product Card Id': [1360,1360],
    'Product Price': [327.75,327.75],
    'Order Region': ['Southeast Asia','Southeast Asia'],
    'Market': ['Asia','Asia']
})

columns_to_encode = ['Delivery Status','Category Name','Customer City','Customer Country','Customer Segment',
                    'Customer State','Order City','Order Country','Order Status','Order Region','Market']

for column in columns_to_encode:
    instances_to_predict = frequency_encoding(instances_to_predict, column)

instances_to_predict.drop(columns=columns_to_encode, inplace=True)

scaler = StandardScaler()
instances_to_predict_scaled = scaler.fit_transform(instances_to_predict)

X_train = np.random.rand(10, instances_to_predict_scaled.shape[1])
y_train = np.random.randint(0, 2, 10)  # Random binary labels

gb_model = XGBClassifier()
gb_model.fit(X_train, y_train)

predictions = gb_model.predict(instances_to_predict_scaled)

st.write("Case 1: Predicted Sales:", predictions[0])
st.write("Case 2: Predicted Sales:", predictions[1])
