import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

def main():
    """
    Main function to run the Streamlit app for binary classification.
    Allows users to select a classifier, adjust hyperparameters, and view performance metrics.
    """
    # Set the title and sidebar title of the app
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Are your mushrooms edible or poisonous?')
    st.sidebar.markdown('Are your mushrooms edible or poisonous?')

    @st.cache(persist=True)
    def load_data():
        """
        Load and preprocess the mushroom dataset.
        Encodes categorical variables using LabelEncoder.

        Returns:
            pd.DataFrame: Preprocessed mushroom dataset.
        """
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        """
        Split the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): The preprocessed dataset.

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        """
        Plot selected evaluation metrics.

        Args:
            metrics_list (list): List of metrics to plot.
        """
        if 'confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            cm_display = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(cm_display.figure_)

        if 'ROC curve' in metrics_list:
            st.subheader('ROC Curve')
            roc_display = RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(roc_display.figure_)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            pr_display = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(pr_display.figure_)

    # Load and split the data
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    # Sidebar options for classifier selection
    st.sidebar.subheader('Choose classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest'))

    # Support Vector Machine (SVM) classifier
    if classifier == 'Support Vector Machine':
        st.sidebar.subheader('Model Hyperparameters')
        c = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio('Gamma (Kernel coefficient)', ('scale', 'auto'), key='gamma')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C=c, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='binary', pos_label=1), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='binary', pos_label=1), 2))
            plot_metrics(metrics_list=metrics)

    # Logistic Regression classifier
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        c = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider('Maximum number of iterations', 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C=c, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='binary', pos_label=1), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='binary', pos_label=1), 2))
            plot_metrics(metrics_list=metrics)

    # Random Forest classifier
    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input('The number of trees in the forest', 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input('The maximum depth of the tree', 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio('Bootstrap samples when building trees', [(True, 'True'), (False, 'False')],
                                     format_func=lambda x: x[1], key='bootstrap')[0]

        metrics = st.sidebar.multiselect('What metrics to plot?', ('confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Random Forest Results')
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(precision_score(y_test, y_pred, average='binary', pos_label=1), 2))
            st.write('Recall:', round(recall_score(y_test, y_pred, average='binary', pos_label=1), 2))
            plot_metrics(metrics_list=metrics)
           
    # Press this button if we want to display the raw dataset
    if st.sidebar.checkbox('Show raw data', False):
       st.subheader('Mushroom Data set classification')
       st.write(df)

if __name__ == '__main__':
    main()


