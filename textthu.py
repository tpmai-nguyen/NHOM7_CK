import streamlit as st
st.header("Nhóm 7")
st.write("""
1. Nguyễn Thị Phương Mai K224131544  
2. Nguyễn Hồ Minh Nguyệt K224131549  
3. Phạm Thị Hoài Thư K224131560
""")
st.header("Phân tích mô tả")
import seaborn as sns
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

import matplotlib.pyplot as plt
st.subheader("Biểu đồ phân phối và hộp cho 'days_for_shipping_real")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
df[["days_for_shipping_real"]].hist(bins=50, ax=axes[0])
df[["days_for_shipping_real"]].boxplot(ax=axes[1], vert=False)
st.pyplot(fig)

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

import matplotlib.pyplot as plt
from typing import Tuple

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

delivery_status_data = df['delivery_status']
delivery_status_summary = delivery_status_data.value_counts()

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.countplot(data=df, x='delivery_status', order=delivery_status_summary.index, palette='coolwarm')
plt.title('Delivery Status Distribution')
plt.xlabel('Delivery Status')
plt.ylabel('Count')
st.pyplot(plt.gcf())

df_copy['late_days'].value_counts().plot.pie(legend = ["0", "1"])

delivery_status_data = df['delivery_status']
shipping_mode_data = df['shipping_mode']

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

late_delivery_data = df[df['delivery_status'] == 'Late delivery']
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

group = df.groupby(['market', 'delivery_status']).market.count().unstack()

plt.figure(figsize=(10, 6))
group.plot(kind='bar', ax=plt.gca())
plt.title('Delivery Status by Market')
plt.xlabel('Market')
plt.ylabel('Count')

st.pyplot(plt)



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
sns.barplot(x='category_name', y='sales', data=df)
plt.title('Doanh thu theo loại hàng')
plt.xlabel('Loại hàng')
plt.ylabel('Doanh thu')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(plt)


