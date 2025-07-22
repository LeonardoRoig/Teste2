import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cross-validation scores
try:
    mean_cv_accuracy = joblib.load('mean_cv_accuracy.joblib')
    std_cv_accuracy = joblib.load('std_cv_accuracy.joblib')
except FileNotFoundError:
    mean_cv_accuracy = None
    std_cv_accuracy = None
except Exception as e:
    mean_cv_accuracy = None
    std_cv_accuracy = None

# Load test set accuracy
try:
    test_accuracy = joblib.load('test_accuracy.joblib')
except FileNotFoundError:
    test_accuracy = None
except Exception as e:
    test_accuracy = None

# Function for the main prediction page
def main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy):
    st.title('Previsão de Obesidade')

    # Initialize session state for prediction status
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    st.header("Insira seus dados para a previsão de Obesidade")

    # Define the questions and widgets for input, including 'peso' and 'altura'
    perguntas_amigaveis_widgets = {
        'vegetais': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'ref_principais': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1, "max_value": 4, "step": 1},
        'agua': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'atv_fisica': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0, "max_value": 3, "step": 1},
        'atv_eletronica': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0, "max_value": 2, "step": 1},
        'idade': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 18, "step": 1},
        'peso': {"pergunta": "Qual o seu peso em kg? (inteiro): ", "tipo": "number_input", "min_value": 0, "step": 1}, # Added peso
        'altura': {"pergunta": "Qual a sua altura em metros? (ex: 1.75): ", "tipo": "number_input", "min_value": 0.0, "format": "%.2f"}, # Added altura
        'historico': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'al_calorico': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'ctrl_caloria': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'entre_ref': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'fumante': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'alcool': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'transporte': {"pergunta": "Seu meio de transporte principal envolve caminhada ou bicicleta? ", "tipo": "radio", "opcoes": {0: 'Sim', 1: 'Não'}},
    }

    # Define the order of features expected by the model
    # This list MUST match the order and names of features the model was trained on
    colunas_features = [
        'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
        'idade', 'peso', 'altura', # Added peso and altura
        'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
        'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
    ]

    dados_entrada = {}

    # Add a single input for Gender
    genero_selecionado = st.radio("Qual o seu gênero?", ['Feminino', 'Masculino'], key='genero_input_pred')

    # Map the single gender input back to the 'feminino' and 'masculino' columns
    if genero_selecionado == 'Feminino':
        dados_entrada['feminino'] = 1
        dados_entrada['masculino'] = 0
    else:
        dados_entrada['feminino'] = 0
        dados_entrada['masculino'] = 1

    # Add input widgets for all features except 'feminino' and 'masculino' and 'grupo_idade'
    for coluna in colunas_features:
        # Skip 'feminino' and 'masculino' as they are handled by the single gender input
        if coluna in ['feminino', 'masculino']:
            continue

        if coluna in perguntas_amigaveis_widgets:
            widget_info = perguntas_amigaveis_widgets[coluna]
            pergunta = widget_info["pergunta"]
            tipo_widget = widget_info["tipo"]

            if tipo_widget == "number_input":
                min_value = widget_info.get("min_value")
                max_value = widget_info.get("max_value")
                step = widget_info.get("step")
                format_str = widget_info.get("format")
                dados_entrada[coluna] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, format=format_str, key=f'{coluna}_input')
            elif tipo_widget == "radio":
                opcoes = list(widget_info["opcoes"].keys())
                opcoes_labels = list(widget_info["opcoes"].values())
                selected_label = st.radio(pergunta, opcoes_labels, key=f'{coluna}_input')
                dados_entrada[coluna] = opcoes[opcoes_labels.index(selected_label)]
        else:
             # This case should ideally not be reached if colunas_features and perguntas_amigaveis_widgets are aligned
             st.warning(f"Widget not defined for column: {coluna}. Using text input as fallback.")
             dados_entrada[coluna] = st.text_input(f"Enter value for '{coluna}': ", key=f'{coluna}_input_fallback')


    # Display test set accuracy below the input fields
    if test_accuracy is not None:
        st.subheader('Performance do Modelo no Conjunto de Teste:')
        st.write(f"Acurácia no Teste: **{test_accuracy:.2f}**")
        st.info("Este valor indica a performance do modelo em dados que ele não viu durante o treinamento.")

    # Add a button to trigger the prediction
    if st.button('Prever Obesidade', key='predict_button'):
        if model is not None:
            # Create a DataFrame with the input data
            # Ensure the order of columns matches the training data
            novo_dado_df = pd.DataFrame([dados_entrada])
            novo_dado_df = novo_dado_df.reindex(columns=colunas_features, fill_value=0)

            # Make the prediction
            previsao = model.predict(novo_dado_df)
            previsao_proba = model.predict_proba(novo_dado_df)[:, 1]

            # Display the prediction result
            st.subheader('Resultado da Previsão:')
            if previsao[0] == 1:
                st.write(f"A previsão é: **Obeso**")
                st.subheader('Recomendações para Obesidade:')
                st.markdown("""
                *   Consulte um nutricionista para um plano alimentar individualizado.
                *   Inicie um programa de exercícios físicos regular, com acompanhamento profissional.
                *   Gerencie o estresse com técnicas de relaxamento ou terapia.
                *   Priorize o sono de qualidade, visando 7-9 horas por noite.
                *   Participe de grupos de apoio ou procure terapia para lidar com questões emocionais relacionadas à alimentação.
                """)
            else:
                st.write(f"A previsão é: **Não Obeso**")
                st.subheader('Recomendações para Manter um Peso Saudável:')
                st.markdown("""
                *   Mantenha uma dieta balanceada com variedade de frutas, vegetais e proteínas magras.
                *   Continue praticando atividade física regularmente.
                *   Monitore seu peso e hábitos alimentares periodicamente.
                *   Beba água suficiente ao longo do dia.
                *   Evite o consumo excessivo de alimentos processados e açucarados.
                """)

            st.write(f"Probabilidade de ser Obeso: **{previsao_proba[0]:.2f}**")

            # Display overall model performance metrics below the prediction
            st.subheader('Performance Geral do Modelo (Cross-Validation):')
            if mean_cv_accuracy is not None and std_cv_accuracy is not None:
                 st.write(f"Acurácia Média: {mean_cv_accuracy:.2f} (+/- {std_cv_accuracy*2:.2f})")
                 st.info("Estes valores indicam a performance geral do modelo em diferentes subconjuntos dos dados, não a confiança desta previsão específica.")

            # --- Add Bar Chart Visualization ---
            st.subheader("Seus Hábitos e Fatores de Risco:")

            # Define features to visualize and their positive/negative status
            features_to_visualize = {
                'vegetais': {'label': 'Consumo de Vegetais', 'positive': True, 'type': 'numerical'}, # Higher is generally positive
                'ref_principais': {'label': 'Refeições Principais/Dia', 'positive': True, 'type': 'numerical'}, # Within a healthy range is positive
                'agua': {'label': 'Consumo de Água (Litros)', 'positive': True, 'type': 'numerical'}, # Higher is generally positive
                'atv_fisica': {'label': 'Atividade Física', 'positive': True, 'type': 'numerical'}, # Higher is generally positive
                'atv_eletronica': {'label': 'Tempo de Tela', 'positive': False, 'type': 'numerical'}, # Higher is generally negative
                'historico': {'label': 'Histórico Familiar', 'positive': False, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'al_calorico': {'label': 'Alimentos Calóricos', 'positive': False, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'ctrl_caloria': {'label': 'Controle de Calorias', 'positive': True, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'entre_ref': {'label': 'Comer Entre Refeições', 'positive': False, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'fumante': {'label': 'Fumante', 'positive': False, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'alcool': {'label': 'Consumo de Álcool', 'positive': False, 'type': 'binary', 'binary_labels': {0: 'Não', 1: 'Sim'}},
                'transporte': {'label': 'Transporte Ativo (Caminhar/Bike)', 'positive': True, 'type': 'binary', 'binary_labels': {0: 'Sim', 1: 'Não'}}, # 0 is positive here
            }

            # Prepare data for the bar chart
            chart_data = []
            x_labels = []
            bar_colors = []

            for feature_name, info in features_to_visualize.items():
                if feature_name in dados_entrada:
                    value = dados_entrada[feature_name]
                    label = info['label']
                    is_positive = info['positive']
                    feature_type = info['type']
                    binary_labels = info.get('binary_labels')

                    x_labels.append(label)

                    if feature_type == 'binary':
                        # For binary features, the value (0 or 1) is the height of the bar
                        chart_data.append(value)
                        # Determine color based on the value (0 or 1) and positive/negative status
                        # If positive=True, 1 is blue, 0 is red
                        # If positive=False, 1 is red, 0 is blue
                        color = 'blue' if (is_positive and value == 1) or (not is_positive and value == 0) else 'red'
                        bar_colors.append(color)
                    elif feature_type == 'numerical':
                         # For numerical features, use the raw value as the bar height
                         chart_data.append(value)
                         # For simplicity, color numerical features based on whether higher is positive or negative
                         color = 'blue' if is_positive else 'red'
                         bar_colors.append(color)


            # Create the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(x_labels, chart_data, color=bar_colors)

            # Add labels and title
            ax.set_ylabel('Valor/Resposta do Usuário (0/1 para binário)')
            ax.set_title('Seus Hábitos e Fatores de Risco')
            plt.xticks(rotation=45, ha='right') # Rotate labels for better readability

            # Add text labels on top of the bars for binary features (Sim/Não)
            for bar, feature_name in zip(bars, features_to_visualize.keys()):
                 info = features_to_visualize[feature_name]
                 if info['type'] == 'binary':
                     height = bar.get_height()
                     binary_labels = info.get('binary_labels')
                     if binary_labels is not None:
                         # Find the text label corresponding to the numerical value (height)
                         text_label = binary_labels.get(height)
                         if text_label is not None:
                             ax.text(bar.get_x() + bar.get_width()/2., height,
                                     text_label,
                                     ha='center', va='bottom')
                 else: # For numerical features, display the numerical value
                     height = bar.get_height()
                     ax.text(bar.get_x() + bar.get_width()/2., height,
                              f'{height}',
                              ha='center', va='bottom')


            # Adjust layout to prevent labels overlapping
            plt.tight_layout()

            # Display the chart in Streamlit
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory


            st.session_state.prediction_made = True

        else:
            st.error("Model not loaded. Cannot make prediction.")

# Load the trained model
try:
    model = joblib.load('obesity_model.joblib')
except FileNotFoundError:
    st.error("Model file 'obesity_model.joblib' not found. Please ensure the trained model is saved in the same directory as app.py")
    model = None

# Load cross-validation scores
try:
    mean_cv_accuracy = joblib.load('mean_cv_accuracy.joblib')
    std_cv_accuracy = joblib.load('std_cv_accuracy.joblib')
except FileNotFoundError:
    mean_cv_accuracy = None
    std_cv_accuracy = None
except Exception as e:
    mean_cv_accuracy = None
    std_cv_accuracy = None

# Load test set accuracy
try:
    test_accuracy = joblib.load('test_accuracy.joblib')
except FileNotFoundError:
    test_accuracy = None
except Exception as e:
    test_accuracy = None

# Display the main page content directly
main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy)
