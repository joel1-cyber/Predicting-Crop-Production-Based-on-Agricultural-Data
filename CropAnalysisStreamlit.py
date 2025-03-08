import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import streamlit as st
import os
import plotly.express as px
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
# from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet  [Testing which model performs well]
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# from sklearn.tree import DecisionTreeRegressor    Testing which model performs well
#import xgboost as xgb  Just for testing which model performs well
import joblib



st.title("Crop Production Analyzer: Insights & Forecasting🌾")
def CleaningAndPreprocessing(CropData):
   #Null Checking
    print(CropData.isnull().sum())
    #only omitting the Columns that are not relevant and empty
    CropData.drop(columns=["Note","Flag Description","Flag"],axis=1,inplace=True)
    
    print(CropData.isnull().sum())
    #print('duplicated values is :',CropData.duplicated().sum()) 
   
   #
    CropData_Pivoted = CropData.pivot_table(index=['Year', 'Area','Item'], columns=['Element'], values='Value',aggfunc='first')
    CropData_Pivoted.reset_index(inplace=True)
    #after Pivoting has been done
    CropData_Pivoted.drop(columns=['Yield/Carcass Weight','Stocks','Laying','Milk Animals','Producing Animals/Slaughtered'],axis=0,inplace=True)
    CropData_Pivoted= CropData_Pivoted.dropna(subset=['Yield', 'Area harvested', 'Production'],how='any')
    
    #Data Standardization
    CropData_Pivoted['Year'] = CropData_Pivoted['Year'].astype(int)
    CropData_Pivoted[['Area harvested', 'Production', 'Yield']] = CropData_Pivoted[['Area harvested', 'Production', 'Yield']].astype(float)
    CropData_Pivoted.rename(columns={'Area harvested':'Area_Harvested_(in ha)','Yield':'Yield_(in kg/ha)','Production':'Production_(in tons)'},inplace=True)
    print(CropData_Pivoted.info())

    # Encoding has been done for categorical variables 
    Encoding=LabelEncoder()
    CropData_Pivoted['Area_Encoded']=Encoding.fit_transform(CropData_Pivoted['Area'])
    CropData_Pivoted['Item_Encoded']=Encoding.fit_transform(CropData_Pivoted['Item'])
    
    #Checked and attached the separate image for that 
    sns.boxplot(x=CropData_Pivoted["Production_(in tons)"])
    plt.title("Boxplot Before Outlier Treatment")
    plt.show()

    # Outlier Detection & Correction
    ColumnstoCheck=['Area_Harvested_(in ha)','Yield_(in kg/ha)','Production_(in tons)']
    for col in ColumnstoCheck:
        Q1=CropData_Pivoted[col].quantile(0.25)
        Q3=CropData_Pivoted[col].quantile(0.75)
        IQR=Q3-Q1
        lowerbound=Q1-IQR*1.5
        upperbound=Q3+IQR*1.5
        outliers = CropData_Pivoted[(CropData_Pivoted[col] < lowerbound) | (CropData_Pivoted[col] > upperbound)]
        CropData_Pivoted[col]=CropData_Pivoted[col].clip(lowerbound,upperbound)
    
        print(f"Feature: {col}")
        print(f"Lower Bound: {lowerbound}, Upper Bound: {upperbound}")
        print(f"Total Outliers: {outliers.shape[0]}\n")
        sns.boxplot(x=CropData_Pivoted[col])
        plt.title("Boxplot After Outlier Removal (IQR)")
        plt.show()
    CropData_Pivoted.to_csv('ProcessedCropData.csv',index=False)
    return CropData_Pivoted

#Streamlit
def create_sidebar():
    # Navigation
    st.sidebar.markdown(f'Hello {os.getlogin()} !! :smile:')
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to",
        ["EDA", "Predicting Crop Production"]
    )
    return selected_tab   

def filter(CropData):
    st.divider()
    st.header("Exploratory Data Analysis - Understanding Crop Trends/Distribution")
    
    st.sidebar.subheader("Filter Section") 
    Area_List=CropData['Area'].unique()
    SelectedArea=st.sidebar.multiselect("Area",Area_List)
    st.sidebar.write('🗺️ Selected Area is ',','.join(SelectedArea))
    minyear,maxyear=CropData['Year'].min(),CropData['Year'].max()
    Year_Range=st.sidebar.slider(" Year Range:", min_value=minyear, max_value=maxyear, value=(minyear, maxyear))
    st.sidebar.write(f"Showing data from {Year_Range[0]} to {Year_Range[1]} 📅 ")
    CropType=CropData['Item'].unique()
    SelectedCropType=st.sidebar.multiselect("CropType",CropType)
    st.sidebar.write('Selected Crop Type 🌱 is ',','.join(SelectedCropType))


    return SelectedArea,Year_Range,SelectedCropType


def EDA(CropData):
    print('using visualization ')
    SelectedArea,Year_Range,SelectedCropType=filter(CropData)
    
    #grouped = df.groupby(['Region', 'Crop'])['Area_Harvested'].sum().reset_index()
    CropData["Year"] = CropData["Year"].astype(int)
    if SelectedArea:
        CropData=CropData[CropData['Area'].isin(SelectedArea)]
    if SelectedCropType:
        CropData=CropData[CropData['Item'].isin(SelectedCropType)]
    if Year_Range:
        CropData=CropData[(CropData['Year']>=Year_Range[0]) & (CropData['Year']<=Year_Range[1])]
    col1,col2=st.columns(2)
    #
    Crop_Totals=CropData.groupby('Item')['Area_Harvested_(in ha)'].sum()
    most_cultivatedCrop = Crop_Totals.idxmax()
    mostAreaHarvested=Crop_Totals.max()
    least_cultivatedCrop = Crop_Totals.idxmin()
    leastAreaHarvested=Crop_Totals.min()
    col1.metric('🌾 Most Cultivated Crop  (hectares)  is :',most_cultivatedCrop+' '+ str(mostAreaHarvested))
    col2.metric('🌾 Least Cultivated Crop  (hectares)  is :',least_cultivatedCrop+' '+ str(leastAreaHarvested))


    ## Geographical Distribution
    st.subheader("🌍 Geographical Distribution of Agriculture 🚜🌾")
    geo_df = CropData.groupby("Area")["Production_(in tons)"].sum().reset_index()
    fig = px.choropleth(geo_df, locations="Area", locationmode="country names", color="Production_(in tons)",
                        title="Total Crop Production by Region", color_continuous_scale="Viridis")
    st.plotly_chart(fig)

    #Temporal Analysis 
    st.subheader("Yearly Trends Over Area harvested, Yield, and Production ")
    year_df=CropData.groupby("Year")[['Area_Harvested_(in ha)','Yield_(in kg/ha)','Production_(in tons)']].sum().reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Area Harvested and Yield on the left y-axis
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Area Harvested (Mha) & Yield (tons/ha)", color="black")
    ax1.plot(year_df['Year'],year_df['Area_Harvested_(in ha)'], "bo-", label="Area Harvested")  # Blue
    ax1.plot(year_df['Year'],year_df['Yield_(in kg/ha)'], "ro-", label="Yield")  # Red
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.tick_params(axis="y", labelcolor="black")
    # Set X-axis to show whole numbers only (no decimals)
    ax1.set_xticks(year_df['Year']) 
    # Create a second y-axis for Production
    ax2 = ax1.twinx()
    ax2.set_ylabel("Production (tons)", color="green")
    ax2.plot(year_df['Year'],year_df['Production_(in tons)'], "go-", label="Production")  # Green
    ax2.tick_params(axis="y", labelcolor="green")

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Title
    plt.title("📅 Yearly Trends: Area Harvested, Yield, and Production 📊🌾")

    st.pyplot(fig)
    
    # Scatter plot: Area Harvested vs Yield
    st.subheader("Environmental Relationship between Area harvested and Yield")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=CropData["Area_Harvested_(in ha)"], y=CropData["Yield_(in kg/ha)"], color="blue",alpha=0.3, s=10)
    plt.xlabel("Area Harvested (Mha)")
    plt.ylabel("Yield (tons/ha)")
    plt.title("Area Harvested vs Yield")
    st.pyplot(plt)

    #input-output  RelaionShips
    st.subheader("Correlation between Area_harvested, Yield,Production,Area,Crop Tpe and Year  ")
    corr_matrix=CropData[['Area_Harvested_(in ha)','Yield_(in kg/ha)','Production_(in tons)','Area_Encoded','Item_Encoded','Year']].corr()
    fig,ax=plt.subplots()
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    #Comparision Analysis

    st.subheader("Comparison of Crop Yields")
    fig,ax=plt.subplots(figsize=(10, 5))
    avg_yield = CropData.groupby("Item")["Yield_(in kg/ha)"].mean().reset_index()
    fig = px.bar(avg_yield, x="Item", y="Yield_(in kg/ha)", title="Average Yield per Crop")
    st.plotly_chart(fig)



    # Productivity Ratio Analysis
    st.subheader("Productivity Analysis")

    # Group data by Item and sum numeric columns
    Itemdata = CropData.groupby("Item")[["Area_Harvested_(in ha)", "Production_(in tons)"]].sum().reset_index()

    # Compute weighted average yield
    Itemdata["Yield_(in kg/ha)"] = CropData.groupby("Item").apply(
        lambda x: np.average(x["Yield_(in kg/ha)"], weights=x["Area_Harvested_(in ha)"])
    ).reset_index(drop=True)

    # Handle NaN values
    Itemdata["Yield_(in kg/ha)"].fillna(0, inplace=True)

    # Compute Productivity Ratio
    Itemdata["Productivity Ratio"] = Itemdata["Production_(in tons)"] / Itemdata["Area_Harvested_(in ha)"]
    Itemdata['Variation(Production & Yield)']=Itemdata["Productivity Ratio"] *1000 - Itemdata['Yield_(in kg/ha)']
    # Display results
    st.dataframe(Itemdata[['Item', 'Production_(in tons)', 'Area_Harvested_(in ha)', 'Productivity Ratio', 'Yield_(in kg/ha)','Variation(Production & Yield)']])


def FeatureSelection(CropData):

    #Linear using Filter Method
    target="Production_(in tons)"
    NumericalCropData=CropData.drop(columns=['Area','Item'])
    Correlationmatrix=NumericalCropData.corr()
    target_corr = Correlationmatrix[target].drop(target)
    selectedFeatures=target_corr[abs(target_corr)>0.3].index.tolist()
    print('using Pierson Correaltion matrix we found the most important feature',selectedFeatures)

    #using Wrapper Method 

    print('using Wrapper Method RFE')
    x=CropData.drop(columns=['Area','Item','Production_(in tons)'])
    y=CropData['Production_(in tons)']

    model=RandomForestRegressor(n_estimators=100,n_jobs=-1)

    rfe=RFE(estimator=model,n_features_to_select=3)
    xselected=rfe.fit_transform(x,y)

    selectedFeatures=x.columns[rfe.support_]
    print('using RFE MEthod ',selectedFeatures)

    #using Embedded Method
    print('Random forest method using embedded method')
    model=RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=42)
    model.fit(x,y)
    feature_importances = pd.DataFrame({'Feature': x.columns, 'Importance': model.feature_importances_})
    feature_importances=feature_importances.sort_values(by="Importance",ascending=False)
    top_features = feature_importances['Feature'][:4]  # Select top 3 features


    print("Selected Features:", list(top_features))


def ModelTraining_Predicting(CropData):

    X=CropData.drop(columns=['Area','Item','Production_(in tons)'])
    #print(X)
    Y=CropData['Production_(in tons)']

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    # Below Models are just for testing 
    # #Model 1
    # print('Linear Regression')
    # model=LinearRegression()
    # model.fit(X_train,Y_train)
    # Y_pred=model.predict(X_test)
    # print(Y_pred)
    # ModelEvaluation(Y_test,Y_pred,"Linear Regression")

    # #Model-2
    # print('Ridge Regression ')
    # ridge=Ridge(alpha=10)
    # ridge.fit(X_train,Y_train)
    # Y_pred=ridge.predict(X_test)
    # print(Y_pred)
    # ModelEvaluation(Y_test,Y_pred,"Ridge Regression")


    # # #Model-3
    # print('Lasso Regression ')
    # lasso=Lasso(alpha=10)
    # lasso.fit(X_train,Y_train)
    # Y_pred=lasso.predict(X_test)
    # print(Y_pred)
    # ModelEvaluation(Y_test,Y_pred,"Lasso Regression")

    #  # #Model-4
    # print('ElasticNet Regression ')
    # elastic=ElasticNet(alpha=100,l1_ratio=1)
    # elastic.fit(X_train,Y_train)
    # Y_pred=elastic.predict(X_test)
    # print(Y_pred)
    # ModelEvaluation(Y_test,Y_pred,"ElasticNet Regression")


     #Model5
    # print('Descision Tree Regression ')
    # dt=DecisionTreeRegressor(max_depth=10,random_state=42,min_samples_split=10, min_samples_leaf=5)
    # dt.fit(X_train,Y_train)
    # Y_pred=dt.predict(X_test)
    # print(Y_pred)
    # ModelEvaluation(Y_test,Y_pred,"Decision Tree Regression")

    #Finally the below model is performing well compared to the other models
    print('RandomForest Tree Regression ')
    rfr=RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rfr.fit(X_train,Y_train)
    Y_pred=rfr.predict(X_test)
    print(Y_pred)
    PredictionVisualization (Y_test,Y_pred,"Random Tree Regression")
    ModelEvaluation(Y_test,Y_pred)
    
    ModelPerformanceChecking(X_train,Y_train,rfr)

    joblib.dump(rfr, "crop_production_model.pkl")  
    print("Model saved successfully!")  

    




#Evaluation Metrics 
def ModelEvaluation(Y_test,Y_pred):
    print('Model Evaluation Started...')
    mae=mean_absolute_error(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    rmse=np.sqrt(mse)
    r2score=r2_score(Y_test,Y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2score:.2f}")
    print("-" * 40)


def ModelPerformanceChecking(X_train,Y_train,rfr):
    # Checking the model performance using cross validation 
    scores = cross_val_score(rfr, X_train, Y_train, cv=5, scoring="neg_mean_absolute_error")
    print("MAE Scores:", -scores)
    print("Average MAE:", -scores.mean())

    mae_scores = np.array(scores)
    std_dev = np.std(mae_scores)

    print(f"Standard Deviation of MAE: {std_dev}")

    #Since standard deviation of mae is less than 500 the model perform consistenly around different dataset 

# Verifying the trained model using visualization 
def PredictionVisualization (Y_test,Y_pred,ModelName):
    plt.plot(Y_test, Y_pred, 'o', label=ModelName)

    plt.title("Model Predictions vs Actual Values")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.show()

def Streamlitinput_Prediction(CropData):

    Model = joblib.load("crop_production_model.pkl") 
    st.divider()
    st.header("Crop Forecast - Predicting Production")
    
    year = st.number_input('Year',min_value=2000, max_value=2030, step=1)  
    area_harvested = st.number_input("Area Harvested (in ha)",min_value=0.0)  
    yield_value = st.number_input("Yield (in kg/ha): ",min_value=0.0) 

    AreaMapping=CropData[["Area","Area_Encoded"]].drop_duplicates()
    ItemMapping=CropData[["Item","Item_Encoded"]].drop_duplicates()

    area=st.selectbox("Area",AreaMapping['Area'].tolist())
    item=st.selectbox("Crop Type",ItemMapping['Item'].tolist())

    area_encoded=AreaMapping.loc[AreaMapping['Area']==area,"Area_Encoded"].values[0]
    item_encoded=ItemMapping.loc[ItemMapping['Item']==item,"Item_Encoded"].values[0]
    print(AreaMapping.head())
    print(ItemMapping.head())

    st.write("Selected Values are ",year, area_harvested, yield_value,area,item)

    if st.button("🔮 Predict Production", key="predict_btn"):
        user_data = np.array([[year,area_harvested, yield_value,area_encoded,item_encoded]])  
        if area_harvested < 0.00 or yield_value < 0.00:
            st.error("Values must be non-negative!")
        else:
            prediction = Model.predict(user_data)
            st.success(f"Predicted Production: {prediction[0]:,.2f} tons")  


    

def ModelBuilding(CropData):
    FeatureSelection(CropData)  
    ModelTraining_Predicting(CropData)
    Streamlitinput_Prediction(CropData)
    


def main():
    
    #Step 1 Preprocessing and cleaning the dataset
    CropData=pd.read_csv("FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv")
    CleaningAndPreprocessing(CropData)
    CleanedCropData=pd.read_csv("ProcessedCropData.csv")
    Selectedtab=create_sidebar()
    if Selectedtab=="EDA":
        EDA(CleanedCropData)
    else:
        print("Model Building Started...")
        ModelBuilding(CleanedCropData)


if __name__=="__main__":
    main()
