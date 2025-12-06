import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    layout='wide'
)

SEED = 12345

@st.cache_data
def load_and_split_data():
    data = pd.read_csv('users_behavior.csv')

    # Train y Temporal
    train, temp = train_test_split(data,test_size=0.4,random_state=SEED)

    # Validacion y Prueba
    validation, test = train_test_split(temp,test_size=0.5,random_state=SEED)

    return train, validation, test

@st.cache_data
def features_target(data,target_column):
    features = data.drop(columns=target_column)
    target = data[target_column]

    return features, target

@st.cache_data
def decision_tree_classifier(train,validation):
    train_features,train_target = features_target(data=train,target_column='is_ultra')
    validation_features, validation_target = features_target(validation,target_column='is_ultra')

    best_depth = 0
    best_score = 0

    x = []
    y = []

    for depth in range(1,16):
        model = DecisionTreeClassifier(random_state=SEED,max_depth=depth)
        model.fit(train_features,train_target)
        prediction =  model.predict(validation_features)
        score = accuracy_score(validation_target,prediction)

        y.append(score)
        x.append(depth)
        
        if score > best_score:
            best_score = score
            best_depth = depth
    
    x = np.array(x)
    y = np.array(y)

    fig = px.line(x=x,y=y,title='DecisionTreeClassifier Accuracy Score Through Different Depths',labels={'x': 'Depth per Iteration','y': 'Accuracy Score'},markers=1)
    st.plotly_chart(fig)

    st.markdown(f'Best Depth: `{best_depth}`. Accuracy Score: `{best_score}`.')

    return model

def random_forest_classifier(train,validation):
    train_features,train_target = features_target(data=train,target_column='is_ultra')
    validation_features, validation_target = features_target(data=validation,target_column='is_ultra')

    best_estimator = 0
    best_score = 0

    x = []
    y = []

    for estimator in range(1,51,5):
        model = RandomForestClassifier(n_estimators=estimator,random_state=SEED)
        model.fit(train_features,train_target)
        prediction = model.predict(validation_features)
        score = accuracy_score(validation_target,prediction)

        y.append(score)
        x.append(estimator)

        if score > best_score:
            best_score = score
            best_estimator = estimator
    
    x = np.array(x)
    y = np.array(y)

    fig = px.line(x=x,y=y,title='RandomForestClassifier Accuracy Score Through Different Estimators',labels={'x': 'Number of Estimators','y': 'Accuracy Score'},markers=1)
    st.plotly_chart(fig)

    st.markdown(f'Best Estimator: `{best_estimator}`. Accuracy Score: `{best_score}`.')

    return model

def logistic_regression(train, validation):
    train_features, train_target = features_target(data=train,target_column='is_ultra')
    validation_features, validation_target = features_target(data=validation,target_column='is_ultra')

    model = LogisticRegression(random_state=SEED,solver='liblinear')
    model.fit(train_features,train_target)
    prediction = model.predict(validation_features)
    score = accuracy_score(validation_target,prediction)

    x = np.array(['Threshold','Model\'Score'])
    y = np.array([0.75,score])

    fig = px.bar(x=x,y=y,color=x,title='LogisticRegression Score',labels={'x': 'Categories','y': 'Accuracy Score'})
    st.plotly_chart(fig)

    st.markdown(f'LogisticRegression\'s Accuracy Score: `{score}`.')
    
    return model

# ----------- INTERFAZ
st.title('Testing Different Classifier Models (Megaline Company)')
train,validation,test = load_and_split_data()
st.markdown(f'Train shape: :red[${train.shape}$], Validation shape: :red[${validation.shape}$], Test shape: :red[${test.shape}$]. Data ratio $= 6:2:2$')
st.markdown(f'Balance of Classes in Train Set: :green[${train['is_ultra'].value_counts().tolist()}$]')

col_1, col_2, col_3 = st.columns([1.2,1.2,0.8])

with col_1:
    decision_tree_classifier_model = decision_tree_classifier(train=train,validation=validation)
with col_2:
    random_forest_classifier_model = random_forest_classifier(train=train,validation=validation)
with col_3:
    logistic_regression_model = logistic_regression(train=train,validation=validation)

test_features, test_target = features_target(data=test,target_column='is_ultra')
prediction = random_forest_classifier_model.predict(test_features)
score = accuracy_score(test_target,prediction)
matrix = confusion_matrix(test_target,prediction)

col_4, col_5 = st.columns([2,1])

with col_4:
    st.divider()
    st.subheader('Testing RandomForestClassifier on Test Set')
    st.markdown(f'Final Score: `{score}`.')
    st.dataframe(pd.DataFrame(matrix,['Real: 0 (No Ultra)', 'Real: 1 (Ultra)'],columns=['Prediction: 0 (No Ultra)', 'Prediction: 1 (Ultra)']))
    st.markdown(':green[**Conclusion**:] The model shows strong performance detecting \'No Ultra\' users (Class 0, True Negatives), but its effectiveness with \'Ultra\' users (Class 1) is likely lower due to the class imbalance. Accuracy might improve by oversampling the minority class (\'Ultra\') in the training data to improve the model\'s recall for that specific plan.')

with col_5:
    labels = ['0 (No Ultra)', '1 (Ultra)']
    fig_cm = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        labels=dict(x="Prediction", y="Real Value", color="Frecuencia"),
        color_continuous_scale='blues'
    )
    fig_cm.update_xaxes(title_text='Prediction')
    fig_cm.update_yaxes(title_text='Real Value', autorange="reversed")
    fig_cm.update_layout(title_text='Confussion Matrix')
    st.plotly_chart(fig_cm)