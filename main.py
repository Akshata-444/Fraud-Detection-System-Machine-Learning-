# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import streamlit_authenticator as stauth

st.title('Credit Card Fraud Detection!')

# Load the dataset from the csv file using pandas
df=st.cache(pd.read_csv)('credit card fraud data.csv')
df = df.sample(frac=0.1, random_state = 48)

# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data decription: \n',df.describe())
# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

    #Obtaining X (features) and y (labels)
X=df.drop(['Class'], axis=1)
y=df.Class
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)

    #Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
logreg=LogisticRegression()
svm=SVC()
knn=KNeighborsClassifier()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)

#Feature selection through feature importance
@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp
#Classifiers for feature importance
#clf=['Decision Trees','Random Forest']
#mod_feature = st.sidebar.selectbox('Which model to use?', clf)
#start_time = timeit.default_timer()
#if mod_feature=='Decision Trees':
    #model=etree
    #importance=feature_sort(model,X_train,y_train)
#elif mod_feature=='Random Forest':
     #model=rforest
     #importance=feature_sort(model,X_train,y_train)
     #elapsed = timeit.default_timer() - start_time
     #st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))
# #Plot of feature importance
#if st.sidebar.checkbox('Show plot of feature importance'):
     #st.set_option('deprecation.showPyplotGlobalUse', False)
    # plt.bar([x for x in range(len(importance))], importance)
     #plt.title('Feature Importance')
     #plt.xlabel('Feature (Variable Number)')
     #plt.ylabel('Importance')
     #st.pyplot()

#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations
smt = SMOTE()
nr = NearMiss()
def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    mcc= matthews_corrcoef(y_test, y_pred)
    'Matthews Correlation Coefficient: ',mcc
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)
#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Decision Trees']
    classifier = st.sidebar.selectbox('Which Algorithm?', alg)
    #rectifier=['SMOTE','Near Miss','No Rectifier']
    #imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier) 
    
    if classifier == 'Decision Trees':
        model=etree
        compute_performance(model, X_train, y_train,X_test,y_test)


if st.sidebar.checkbox('Submit Fradulent Report'):
    with st.form("form1",clear_on_submit=True):
        st.sidebar.name = st.text_input("Name")
        st.sidebar.Credentials = st.text_input("Credential ID")
        st.sidebar.Password = st.text_input("Password")
    #st.sidebar.hashed_Password = stauth.Hasher(Password).generate()
        st.sidebar.email = st.text_input("Enter email")
        st.sidebar.message = st.text_input("Enter message")
        uploaded_files = st.file_uploader("Choose a image file", accept_multiple_files=True, type=["png","jpg","jpeg"])

        filteredImages = [] # your images here
        NAME=[]
        for uploaded_file in uploaded_files:
            NAME.append(uploaded_file.name)
            bytes_data = uploaded_file.read()
    # save_img(bytes_data)
    # print(uploaded_file, bytes_data)
            filteredImages.append(bytes_data)
    # save_img(filteredImages[0])
            st.image(uploaded_file)

        #st.sidebar.uploaded_files = st.form_button.file_uploader("Choose a image file", accept_multiple_files=True, type=["png","jpg","jpeg"])
        #st.button = ('Submit')
        
        if st.form_submit_button("SUBMIT"):
            st.title('Fraudulent Report Submitted. The data will be feeded to the dataset')
        #submit = st.form("Submit this form")
        
    #st.sidebar.uploaded_files = st.form_button.file_uploader("Choose a image file", accept_multiple_files=True, type=["png","jpg","jpeg"])
    #st.sidebar.uploaded_files = st.form_submit_button.file_uploader("Choose a document file", accept_multiple_files=True, type=["pdf,word"])
    
    #submit = st.form("Submit this form")
