# Problem Statement
'''
# Excess inventory and storage issue

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Model Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Maximize storage capacity
# 
# **Constraints:** Minimize wastage   

# **Success Criteria**
# - **Business Success Criteria**: Increase revenue by 20%
# - **ML Success Criteria**: ML accuracy of atleast 90%
# - **Economic Success Criteria**: Save $100000 to customer

# **Proposed Plan:**
# Forecasting Method

# ### Data Dictionary
#Time:	A  particular period of time 
#Date:	A numbered day in a month, often given with the name of the month or with the month and the year 
#Location:	A location is a fixed place or position 
#Sales volume in Tonnes:	 The number of units, company sells during a specific reporting period .
#Price/ unit	: How much customer is charged for each item sold 
#Customer ID: 	A unique number that's assigned to each Customer
#Diameter (mm):	Diameter of Rod
#Length(meter):	Length of Rod
#Grade:	used to identify and distinguish different types of steel 
#Weather: 	The state of the air and atmosphere at a particular time and place
#Current stock:	Goods already on hand 
#Re-order:	Supply, or deliver the same goods again
#Lead time:	Time interval between the start and completion of a certain task
#Production time	: The total time required to produce a component(items)
#Units Produced: 	Calculates depreciation based on the unit of production
#Production cost	: Generates revenue for the company 
'''

#Libraries
import matplotlib.backends
from sklearn.compose import make_column_transformer
import numpy as np
import pandas as pd # data manipulation
import sweetviz # autoEDA
import matplotlib.pyplot as plt # data visualization
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # machine learning algorithms
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
import joblib
import pickle
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import os
import pymysql
import mysql.connector as connector
#from datacleaner import autoclean
from Dora import Dora 
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing



steel_df1 = pd.read_csv(r"/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv")
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
steel_df1["Location"] = pd.DataFrame(median_imputer.fit_transform(steel_df1[["Location"]]))
steel_df1["Location"].isna().sum()  # all records replaced by median 
steel_df4 = steel_df1[['Date']]
# Creating engine which connect to MySQL
user = 'root' # user name
pw = '######' # password
db = '######' # database

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
steel_df1.to_sql('steel', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from steel'

steel_df2 = pd.read_sql_query(sql, con = engine)

print(steel_df2)

# Show sample of data set
steel_df2.head()

# ## EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
steel_df2.describe()

# ***1st Moment Business Decision (Measures of Central Tendency)***
# 1) Mean
# 2) Median
# 3) Mode
print(steel_df2.mean())
print('\n')
print(steel_df2.median())
print('\n')
print(steel_df2.mode())

# ***2nd Moment Business Decision (Measures of Dispersion)***
# 1) Variance
# 2) Standard deviation
# 3) Range (maximum - minimum)
print(steel_df2.var())
print('\n')
print(steel_df2.std())

# ***3rd Business Moment Decision (Skewness)***
# Measure of asymmetry in the data distribution
steel_df2.skew()

# ***4th Business Moment Decision (Kurtosis)***
# Measure of peakedness - represents the overall spread in the data
steel_df2.kurt()


# AutoEDA
# ## Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([steel_df2, "steel_df"])
my_report.show_html('Report.html')

#pandas-profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(pd.read_csv('/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv'), explorative=True)
#Saving results to a HTML file
profile.to_file("output.html")
'''
#Autoviz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
%matplotlib inline # AutoViz no displays plots automatically. You must perform %matplotlib inline just before you run AutoViz on your data.
# Generate visualizations
filename = "/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv"
sep = ","
dft = AV.AutoViz(
    filename,
    sep=",",
    depVar="",
    dfte=None,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)

#DTALE
import dtale
dtale.show(pd.read_csv("/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv")) 


# ## Data Preprocessing and Cleaning
#1 DATACLEAN AUTO PRE PROCESSING
#clean_df = autoclean(steel_df)
#2 AUTOCLEAN
#3 DORA
#from Dora import Dora
#dora = Dora()
#dora.configure(output = 'A', data = '/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv')
#dora.data
#dora = Dora(output = 0, data = '/Users/gaurav/Documents/360digitmg/Iron rod project/Data100.csv')
#dora.impute_missing_values()
#4 TABULATE
#from tabulate import tabulate
#clean=tabulate(steel_df)
'''

#**Cleaning Unwanted columns**
steel_df=steel_df2.drop(['Customer ID','Sales in Rs/T','time','Date'], axis = 1)

steel_df = steel_df[['Location', 'Climate','Grade','Sales volume in Tonnes', 'Price/ kg', 'Diameter', 'Length',
       'Current stock', 'Re-order', 'Lead time', 'Production time',
       'units produced', 'Production cost']]
steel_df.info()

#**Handling duplicates:**
duplicate = steel_df.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate
sum(duplicate)

# Removing Duplicates
steel_df = steel_df.drop_duplicates() # Returns DataFrame with duplicate rows removed.

#**Missing Value Analysis**
steel_df.isnull().sum() # Check for missing values


# Segregating Non-Numeric features
categorical_features = steel_df.select_dtypes(include = ['object']).columns
print(categorical_features)


# Segregating Numeric features
numeric_features = steel_df.select_dtypes(exclude = ['object']).columns
print(numeric_features)


num_pipeline1 = Pipeline(steps=[('impute1', SimpleImputer(strategy = 'most_frequent'))])

# Mean imputation for Continuous (Float) data
num_pipeline2 = Pipeline(steps=[('impute2', SimpleImputer(strategy = 'mean'))])


# 1st Imputation Transformer
preprocessor = ColumnTransformer([
        ('mode', num_pipeline1, categorical_features),
        ('mean', num_pipeline2, numeric_features)])

print(preprocessor)

# Fit the data to train imputation pipeline model
impute_data = preprocessor.fit(steel_df)

# Save the pipeline
joblib.dump(impute_data, 'impute')

# Transform the original data
cleandata = pd.DataFrame(impute_data.transform(steel_df), columns = steel_df.columns).convert_dtypes()

cleandata.isna().sum()


## Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list([
                                 'Sales volume in Tonnes', 'Price/ kg', 'Diameter', 'Length',
                                        'Current stock', 'Re-order', 'Lead time', 'Production time',
                                        'units produced', 'Production cost']))

winsor

clean = winsor.fit(cleandata)

# Save winsorizer model
joblib.dump(clean, 'winsor')

cleandata1 = clean.transform(cleandata)

# Boxplot
cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

cleandata2 = pd.DataFrame(cleandata1.iloc[:, 3:])
cat_steel_df = pd.DataFrame(cleandata1.iloc[:, :3]) 
# Scaling
## Scaling with MinMaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata2)

# Save Minmax scaler pipeline model
joblib.dump(scale, 'minmax')
scaled_data = pd.DataFrame(scale.transform(cleandata2), columns = cleandata2.columns).convert_dtypes()
scaled_data.describe()
cat_steel_df = pd.get_dummies(cat_steel_df)


clean_data = pd.concat([steel_df4 , cleandata2 , cleandata1.iloc[:,:3]], axis = 1)  # concatenated data will have new sequential index
clean_data1.info()
#convert date column to datetime and subtract one week
clean_data['Date'] = pd.to_datetime(clean_data['Date']) - pd.to_timedelta(7, unit='d')
clean_data.rename(columns = {'Sales volume in Tonnes':'Sales_volume_in_Tonnes'}, inplace = True)
#calculate sum of values, grouped by week
clean_data1=clean_data.groupby([pd.Grouper(key = 'Date', freq = 'W')]).agg(Sales_volume_in_Tonnes = ('Sales_volume_in_Tonnes' , 'mean'))
#clean_data.groupby([pd.Grouper(key='Date', freq='W')])['Sales_volume_in_Tonnes'].mean()
#clean_data = clean_data.sort_values(by=['Date'], ascending=True)
#clean_data = clean_data.to_frame()
clean_data1['Date'] = clean_data1.index
clean_data1.reset_index(inplace = True, drop = True)
clean_data1['Sales_volume_in_Tonnes'] = clean_data1['Sales_volume_in_Tonnes'].fillna(clean_data1['Sales_volume_in_Tonnes'].mean())
clean_data1.to_csv(r'/Users/gaurav/Documents/360digitmg/Arima/test1_arima.csv')



#ARIMA
import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from sqlalchemy import create_engine


user = 'root' # user name
pw = 'gaurav1234' # password
db = 'recommenddb' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
steelrod = pd.read_csv(r"/Users/gaurav/Documents/360digitmg/Arima/test1_arima.csv")
# dumping data into database 
steelrod.to_sql('steelrod', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
# loading data from database
sql = 'select * from steelrod'
steelrod = pd.read_sql_query(sql, con = engine)

steelrod.Sales_volume_in_Tonnes.plot()

# Data Partition
Train = steelrod.head(1456)
Test = steelrod.tail(52)


tsa_plots.plot_acf(steelrod.Sales_volume_in_Tonnes, lags = 52)
tsa_plots.plot_pacf(steelrod.Sales_volume_in_Tonnes, lags = 52)



model1 = ARIMA(Train.Sales_volume_in_Tonnes, order = (1, 1, 52))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 52 weeks
start_index = len(Train)
start_index
end_index = start_index + 51
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Sales_volume_in_Tonnes, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

def MAPE(pred, actual):
    temp = np.abs((pred - actual)/actual)*100
    return np.mean(temp)

MAPE(forecast_test, Test.Sales_volume_in_Tonnes) 

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales_volume_in_Tonnes)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.Sales_volume_in_Tonnes, start_p = 0, start_q = 0,
                      max_p = 52, max_q = 52, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA
model = ARIMA(Train.Sales_volume_in_Tonnes, order = (1,0,0))
res = model.fit()
print(res.summary())


# Forecast for next 52 weeks
start_index = len(Train)
end_index = start_index + 51
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Sales_volume_in_Tonnes, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
MAPE(forecast_best, Test.Sales_volume_in_Tonnes) 
# plot forecasts against actual outcomes
pyplot.plot(Test.Sales_volume_in_Tonnes)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# checking both rmse of with and with out autoarima

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)
# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model.pickle")
import os
os.getcwd()
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Forecast for future 52 weeks
start_index = len(steelrod)
end_index = start_index + 51
forecast = model.predict(start = start_index, end = end_index)

print(forecast)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales_volume_in_Tonnes"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
ses = MAPE(pred_ses, Test.Sales_volume_in_Tonnes) 

# Holt method 
hw_model = Holt(Train["Sales_volume_in_Tonnes"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
hw = MAPE(pred_hw, Test.Sales_volume_in_Tonnes) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales_volume_in_Tonnes"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
hwe = MAPE(pred_hwe_add_add, Test.Sales_volume_in_Tonnes) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales_volume_in_Tonnes"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
hwe_w = MAPE(pred_hwe_mul_add, Test.Sales_volume_in_Tonnes) 

# comparing all mape's
di = pd.Series({'Simple Exponential Method':ses, 'Holt method ':hw, 'hw_additive seasonality and additive trend':hwe, 'hw_multiplicative seasonality and additive trend':hwe_w})
mape = pd.DataFrame(di, columns=['mape'])
mape



