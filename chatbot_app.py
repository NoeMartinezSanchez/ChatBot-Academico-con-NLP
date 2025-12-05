#!/usr/bin/env python3
"""
ChatBot Educativo - Versión Streamlit

"""

import streamlit as st
import json
from datetime import datetime
import os

# Configuración de página
st.set_page_config(
    page_title="🤖 ChatBot Educativo - Prepa en Línea SEP",
    page_icon="🎓",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.bot {
        background-color: #f5f5f5;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🎓 Asistente Virtual - Módulo Propedéutico")
st.markdown("""
### 🤖 ChatBot especializado en el módulo propedéutico de **Prepa en Línea SEP**
*Resuelve tus dudas académicas 24/7 con Inteligencia Artificial*
""")

# Inicializar estado de la sesión
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar con información
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("ℹ️ Información")
    
    st.markdown("""
    ### 📚 Temas que puedo explicar:
    
    • **Módulo Propedéutico**: Qué es, objetivos, duración
    """)
    
    # Botón para limpiar historial
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Conversación limpiada!")
    
    # Estado del sistema
    st.markdown("---")
    st.subheader("📊 Estado del sistema")
    
    # Verificar modelos
    model_files = ['models/intent_classifier.pkl', 'models/tfidf_vectorizer.pkl']
    models_exist = all(os.path.exists(f) for f in model_files)
    
    if models_exist:
        st.success("✅ ML: Modelos cargados")
    else:
        st.warning("⚠️ ML: Modelos no encontrados")
        st.info("Ejecuta: `python train_model.py`")

# Importar tu chatbot (AJUSTA ESTAS IMPORTACIONES)
try:
    from chatbot.nl_engine import nl_engine
    from chatbot.intent_classifier import intent_classifier
    from chatbot.response_generator import ResponseGenerator
    
    # Cargar intents
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    
    available_intents = intents_data.get("intents", [])
    response_generator = ResponseGenerator(available_intents)
    
    st.sidebar.success("✅ ChatBot cargado correctamente")
    
except Exception as e:
    st.sidebar.error(f"❌ Error cargando chatbot: {e}")
    available_intents = []
    response_generator = None

# Mostrar historial de chat
st.subheader("💬 Conversación")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
user_input = st.chat_input("Escribe tu pregunta sobre el módulo propedéutico...")

if user_input:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Procesar con tu chatbot
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            try:
                if response_generator and intent_classifier.is_trained:
                    # Usar tu NLP engine existente
                    nlp_results = nl_engine.process_query(user_input, available_intents)
                    bot_response = response_generator.get_response(nlp_results)
                    response_text = bot_response["response"]
                    
                    # Metadata (opcional)
                    with st.expander("📊 Detalles técnicos"):
                        st.json({
                            "intención_detectada": nlp_results['final_intent']['tag'] if nlp_results['final_intent'] else 'unknown',
                            "confianza": f"{nlp_results['confidence']:.2%}",
                            "método": nlp_results['winning_method']
                        })
                else:
                    # Fallback si no está cargado el ML
                    response_text = "⚠️ El sistema ML no está completamente cargado. Ejecuta `python train_model.py` primero."
                    
            except Exception as e:
                response_text = f"❌ Error: {str(e)}"
            
            # Mostrar respuesta
            st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Sección de preguntas rápidas
st.markdown("---")
st.subheader("🚀 Preguntas Rápidas")

# Crear columnas para botones
col1, col2, col3 = st.columns(3)

quick_questions = [
    ("📖 ¿Qué es el módulo propedéutico?", "¿Qué es el módulo propedéutico?"),
    ("👋 Saludo inicial", "Hola, buen día")
]

# Botones de preguntas rápidas
cols = st.columns(3)
for idx, (btn_text, question) in enumerate(quick_questions):
    with cols[idx % 3]:
        if st.button(btn_text, key=f"quick_{idx}"):
            # Simular input del usuario
            with st.chat_message("user"):
                st.markdown(question)
            
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Procesar respuesta
            with st.chat_message("assistant"):
                with st.spinner("🤔 Pensando..."):
                    try:
                        if response_generator and intent_classifier.is_trained:
                            nlp_results = nl_engine.process_query(question, available_intents)
                            bot_response = response_generator.get_response(nlp_results)
                            response_text = bot_response["response"]
                        else:
                            response_text = "⚠️ Sistema no listo"
                        
                        st.markdown(response_text)
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Pie de página
st.markdown("---")
st.caption("🤖 ChatBot Educativo v2.0 | Prepa en Línea SEP | IA con NLP y Machine Learning")

if __name__ == "__main__":
    # Esto es para ejecutar localmente
    # Streamlit automáticamente ejecuta el script
    pass