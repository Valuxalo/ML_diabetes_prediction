import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_NAMES = [
    'gender', 
    'age', 
    'hypertension', 
    'heart_disease', 
    'smoking_history', 
    'bmi', 
    'HbA1c_level', 
    'blood_glucose_level'
]

CATEGORICAL_FEATURES = {
    'gender': ['Female', 'Male'],
    'smoking_history': ['No Info', 'never', 'former', 'current', 'not current', 'ever']
}

BINARY_FEATURES = {
    'hypertension': ['Нет', 'Да'],
    'heart_disease': ['Нет', 'Да']
}

load_dotenv()


class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.model_name = os.getenv("MODEL_NAME")
        self.model_path = os.getenv("MODEL_PATH")
        self.root_dir = Path(__file__).parent.parent.parent
        self.load_model()
        self.feature_types = {
            'gender': 'category',
            'age': 'int',
            'hypertension': 'int',
            'heart_disease': 'int',
            'smoking_history': 'category',
            'bmi': 'float',
            'HbA1c_level': 'float',
            'blood_glucose_level': 'int'
        }

    def load_model(self):
        """Загрузка сохраненной модели"""
        try:
            path=f'{self.root_dir}/{self.model_path}/{self.model_name}.pkl'
            with open(path, 'rb') as file:
                self.model = pickle.load(file)
            return True
        except FileNotFoundError:
            st.warning(f"Модель не найдена. Путь: {path}")
            return False
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}")
            return False
    
    def predict(self, features_dict):
        """Предсказание на основе словаря признаков"""
      
        
        # Предсказание
        try:
            df = pd.DataFrame([features_dict])
            
            # Приводим типы данных
            # Категориальные признаки - как строки
            df['gender'] = df['gender'].astype(str)
            df['smoking_history'] = df['smoking_history'].astype(str)
            
            # Числовые признаки - как числа
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['hypertension'] = pd.to_numeric(df['hypertension'], errors='coerce')
            df['heart_disease'] = pd.to_numeric(df['heart_disease'], errors='coerce')
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
            df['HbA1c_level'] = pd.to_numeric(df['HbA1c_level'], errors='coerce')
            df['blood_glucose_level'] = pd.to_numeric(df['blood_glucose_level'], errors='coerce')
            
            # Проверяем на NaN
            if df.isnull().any().any():
                st.error("Обнаружены пропущенные значения (NaN)")
                st.write(df.isnull().sum())
                return None, None
            # Предсказание
            prediction = self.model.predict(df)
            probability = self.model.predict_proba(df)
            
            return prediction[0], probability[0]
            
        except Exception as e:
            st.error(f"Ошибка предсказания: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None, None

    

def create_input_form():
    """Создание формы ввода данных"""
    with st.form("prediction_form"):
        st.subheader("Данные пациента")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender_map = {
                'Женщина': 'Female',
                'Мужчина': 'Male'
            }
            gender_ru = st.selectbox(
                "Пол",
                options=['Женщина', 'Мужчина'],
                help="Пол пациента"
            )
            age = st.slider(
                "Возраст",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Возраст пациента в годах"
            )
            
            hypertension_ru = st.radio(
                "Гипертония",
                options=['Нет', 'Да'],
                horizontal=True,
                help="Наличие гипертонии"
            )
            
            heart_disease_ru = st.radio(
                "Болезни сердца",
                options=['Нет', 'Да'],
                horizontal=True,
                help="Наличие сердечных заболеваний"
            )
        
        with col2:
            smoking_ru = st.selectbox(
                "История курения",
                options=['Нет информации', 'Никогда', 'Бывший', 'Текущий', 'Не курит сейчас', 'Когда-то'],
                help="История курения"
            )
            smoking_map = {
                'Нет информации': 'No Info',
                'Никогда': 'never',
                'Бывший': 'former',
                'Текущий': 'current',
                'Не курит сейчас': 'not current',
                'Когда-то': 'ever'
            }
            bmi = st.number_input(
                "ИМТ (Индекс массы тела)",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1,
                help="Индекс массы тела (вес/рост²)"
            )
            
            hba1c = st.number_input(
                "Уровень HbA1c",
                min_value=3.0,
                max_value=15.0,
                value=5.7,
                step=0.1,
                help="Уровень гликированного гемоглобина (%)"
            )
            
            blood_glucose = st.number_input(
                "Уровень глюкозы в крови",
                min_value=50,
                max_value=400,
                value=120,
                step=1,
                help="Уровень глюкозы в крови (мг/дл)"
            )

            binary_map = {
                        'Нет': 0,
                        'Да': 1
                        }
            gender = gender_map[gender_ru]
            hypertension = binary_map[hypertension_ru]  # 'Нет' -> 0, 'Да' -> 1
            heart_disease = binary_map[heart_disease_ru]  # 'Нет' -> 0, 'Да' -> 1
            smoking_history = smoking_map[smoking_ru]
        
        submitted = st.form_submit_button(
            "Рассчитать вероятность диабета",
            use_container_width=True,
            type="primary"
        )

    features = {
        'gender': gender,  # 'Female' или 'Male'
        'age': age,  # число
        'hypertension': hypertension,  # 0 или 1
        'heart_disease': heart_disease,  # 0 или 1
        'smoking_history': smoking_history,  # 'No info', 'never', 'former' и т.д.
        'bmi': bmi,  # число
        'HbA1c_level': hba1c,  # число
        'blood_glucose_level': blood_glucose  # число
    }
    return features, submitted

def display_prediction_result(prediction, probability):
    """Отображение результатов предсказания"""
    st.divider()
    st.subheader("Результат предсказания")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if prediction == 1:
            st.markdown("**Высокий риск диабета**")
        else:
            st.markdown("**Низкий риск диабета**")
        
        # Вероятности
        st.metric(
            "Вероятность диабета",
            f"{probability[1]*100:.2f}%",
        )
        
        
def display_sidebar_info():
    """Отображение информации в боковой панели"""
    with st.sidebar:
        st.header("Информация")
        
        st.markdown("""
        ### О модели
        Модель машинного обучения для предсказания риска диабета на основе 8 клинических признаков.
        
        ### Признаки модели:
        - **Пол** (Женский/Мужской)
        - **Возраст** (18-100 лет)
        - **Гипертония** (Да/Нет)
        - **Болезни сердца** (Да/Нет)
        - **История курения** (Никода/Бывший/Текущий/Не сейчас/Когда-то)
        - **ИМТ** (10-60)
        - **HbA1c** (3-15%)
        - **Глюкоза крови** (50-400 мг/дл)
        """)
        
        st.divider()
        
        st.markdown("""
        ### Интерпретация
        - **HbA1c < 5.7%**: Норма
        - **HbA1c 5.7-6.4%**: Преддиабет
        - **HbA1c ≥ 6.5%**: Диабет
                    
        - **Глюкоза < 100 мг/дл**: Норма
        - **Глюкоза 100-125 мг/дл**: Преддиабет
        - **Глюкоза ≥ 126 мг/дл**: Диабет
                    
        - **ИМТ 18.5 - 24.9**: Норма
        - **ИМТ < 18.4**: Дифицит массы
        - **ИМТ > 24.9**: Избыточная масса                  
        """)
        
        st.divider()
        

def main():
    # Заголовок
    st.title("Система прогнозирования диабета")
    st.markdown("*На основе модели машинного обучения (Random Forest)*")
    
    # Инициализация предсказателя
    predictor = DiabetesPredictor()
    
    # Боковая панель
    display_sidebar_info()
    
    # Основная форма
    features, submitted = create_input_form()
    
    if submitted:
        # Проверка загрузки модели
        if predictor.model is None:
            st.error("Модель не загружена! Пожалуйста, обучите модель сначала.")
            st.info("Для обучения модели запустите основной пайплайн или загрузите готовую модель.")
            return
        
        # Предсказание
        with st.spinner("Выполняется расчёт..."):
            prediction, probability = predictor.predict(features)
        
            
            # Отображаем результат
            display_prediction_result(prediction, probability)
            

if __name__ == "__main__":
    main()
