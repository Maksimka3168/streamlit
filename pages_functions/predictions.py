import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, rand_score
from functions.init_df import X_test, y_test
from functions.load_model import get_models


def get_predictions():
    uploaded_file = st.file_uploader("Загрузите ваш файл CSV формата", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}

        feature_names = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
                         'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                         'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'Year', 'Month', 'Day']

        feature_names_unique_validate = ['RainToday', 'Year', 'Month', 'Day']
        for feature in feature_names:
            if feature not in feature_names_unique_validate:
                input_data[feature] = st.number_input(f"{feature}", min_value=0, value=10)

        input_data['RainToday'] = st.number_input('RainToday', min_value=0, value=1, step=1)
        input_data['Year'] = st.number_input('Year', min_value=2000, value=2024, step=1)
        input_data['Month'] = st.number_input('Month', min_value=1, value=12, step=1)
        input_data['Day'] = st.number_input('Day', min_value=1, value=31, step=1)

        if st.button('Сделать предсказание'):
            model1, model2, model3, model4, model5, model6 = get_models()
            input_df = pd.DataFrame([input_data])
            st.write("Входные данные:", input_df)

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model1.predict(input_df)
            predictions_ml2 = model2.fit_predict(input_df)
            predictions_ml3 = model3.predict(input_df)
            predictions_ml4 = model4.predict(input_df)
            predictions_ml5 = model5.predict(input_df)
            probabilities_ml6 = model6.predict(input_df)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)
            pred6 = "[Yes]" if predictions_ml6 == 1 else "[No]"

            st.success(f"Предсказанние LogisticRegression: {predictions_ml1}")
            st.success(f"Предсказанние KMeans: {predictions_ml2}")
            st.success(f"Предсказанние GradientBoostingClassifier: {predictions_ml3}")
            st.success(f"Предсказанние BaggingClassifier: {predictions_ml4}")
            st.success(f"Предсказанние StackingClassifier: {predictions_ml5}")
            st.success(f"Предсказанние Tensorflow: {pred6}")
    else:
        try:
            model1, model2, model3, model4, model5, model6 = get_models()

            predictions_ml1 = model1.predict(X_test)
            predictions_ml2 = model2.fit_predict(X_test)
            predictions_ml3 = model3.predict(X_test)
            predictions_ml4 = model4.predict(X_test)
            predictions_ml5 = model5.predict(X_test)
            probabilities_ml6 = model6.predict(X_test)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = accuracy_score(y_test, predictions_ml2)
            rand_score_ml3 = round(rand_score(y_test, predictions_ml3))
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)

            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность KMeans: {accuracy_ml2}")
            st.success(f"Rand Score GradientBoostingClassifier: {rand_score_ml3}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml4}")
            st.success(f"Точность StackingClassifier: {accuracy_ml5}")
            st.success(f"Точность Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")