from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import logging

from config import Config
from chatbot.nl_engine import nl_engine
from chatbot.intent_classifier import intent_classifier
from chatbot.response_generator import ResponseGenerator


import sys
import io

# Configurar stdout para UTF-8 (solo en Windows)
if sys.platform == "win32":
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Inicializar componentes del chatbot
try:
    # Cargar datos de intenciones
    with open(Config.INTENTS_FILE, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    
    available_intents = intents_data.get("intents", [])

    print(f"🔍 DEBUG: {len(available_intents)} intents cargados")
    for intent in available_intents:
        print(f"  - {intent['tag']}: {len(intent['patterns'])} patrones")
    
    # Inicializar generador de respuestas
    response_generator = ResponseGenerator(available_intents)
    
    # Verificar estado de modelos NLP
    nlp_status = "✅ NLP Avanzado Activado" if Config.NLP["USE_ADVANCED_NLP"] else "⚠️ NLP Básico"
    
    if intent_classifier.is_trained:
        ml_status = "✅ ML Cargado"
        print("   • ✅ ML: Modelos cargados automáticamente")
    else:
        ml_status = "⚠️ ML No Disponible"
        print("   • ⚠️ ML: Modelos no cargados")
        
        # Diagnóstico adicional
        import os
        model_files = [
            'models/intent_classifier.pkl',
            'models/tfidf_vectorizer.pkl', 
            'models/intent_mapping.json'
        ]
        
        existing = [f for f in model_files if os.path.exists(f)]
        
        if existing:
            print(f"   • 📁 Archivos encontrados: {len(existing)}/{len(model_files)}")
            if len(existing) < len(model_files):
                missing = [f for f in model_files if not os.path.exists(f)]
                print(f"   • ❌ Faltan: {[os.path.basename(f) for f in missing]}")
            print("   • 💡 Posible error de carga, revisa logs")
        else:
            print("   • 💡 Ejecuta: python train_model.py")


    
    print("🚀 ChatBot NLP Avanzado Inicializado")
    print(f"   • {nlp_status}")
    print(f"   • {ml_status}")
    print(f"   • {len(available_intents)} intenciones cargadas")
    print(f"   • Modelo: {Config.NLP['MODEL_TYPE']}")
    
except Exception as e:
    print(f"❌ Error inicializando chatbot: {e}")
    raise e

@app.route('/')
def home():
    """Página principal del chatbot"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para procesar mensajes con NLP avanzado"""
    try:
        user_message = request.json.get('message', '').strip()
        session_id = request.json.get('session_id', 'default')
        
        if not user_message:
            return jsonify({
                'response': 'Por favor, escribe tu mensaje.',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
        
        # Procesar con motor NLP avanzado
        nlp_results = nl_engine.process_query(user_message, available_intents)
        
        # Generar respuesta
        bot_response = response_generator.get_response(nlp_results)
        
        # Agregar metadata NLP
        response_data = {
            'response': bot_response["response"],
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'nlp_metadata': {
                'detected_intent': nlp_results['final_intent']['tag'] if nlp_results['final_intent'] else 'unknown',
                'confidence': nlp_results['confidence'],
                'classification_method': nlp_results['winning_method'],
                'entities': nl_engine.extract_entities(user_message),
                'sentiment': nl_engine.analyze_sentiment(user_message),
                'processed_text': nlp_results['processed_text']
            }
        }
        
        # Log de la conversación
        app.logger.info(f"Chat - User: '{user_message}' -> Intent: {response_data['nlp_metadata']['detected_intent']} "
                       f"(Conf: {response_data['nlp_metadata']['confidence']:.3f})")
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error en endpoint /chat: {e}")
        return jsonify({
            'response': '❌ Lo siento, ha ocurrido un error en el sistema. Por favor, intenta nuevamente.',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Endpoint para análisis NLP de texto"""
    try:
        text = request.json.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Texto requerido'}), 400
        
        analysis = {
            'entities': nl_engine.extract_entities(text),
            'sentiment': nl_engine.analyze_sentiment(text),
            'features': nl_engine.text_processor.extract_features(text),
            'word_embeddings_shape': nl_engine.text_processor.get_word_embeddings(text).shape
        }
        
        return jsonify({
            'analysis': analysis,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error en análisis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint para obtener estadísticas del sistema"""
    try:
        stats = {
            'system': {
                'intents_count': len(available_intents),
                'patterns_count': sum(len(intent['patterns']) for intent in available_intents),
                'nlp_enabled': Config.NLP['USE_ADVANCED_NLP'],
                'model_type': Config.NLP['MODEL_TYPE']
            },
            'ml_model': intent_classifier.get_model_info(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({'statistics': stats, 'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check completo"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-nlp',
        'components': {
            'flask': 'ok',
            'nlp_engine': 'ok',
            'intent_classifier': 'ok' if intent_classifier.is_trained else 'not_trained',
            'models_loaded': intent_classifier.is_trained
        }
    }
    
    return jsonify(health_status)

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint para entrenar el modelo (solo desarrollo)"""
    if not Config.DEBUG:
        return jsonify({'error': 'Solo disponible en modo desarrollo'}), 403
    
    try:
        # Aquí iría la lógica para re-entrenar el modelo
        return jsonify({
            'message': 'Entrenamiento iniciado',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Asegurar directorios
    Config.ensure_directories()
    
    print("\n🎯 ENDPOINTS DISPONIBLES:")
    print("   • GET  /          - Interfaz web")
    print("   • POST /chat      - Chat con NLP")
    print("   • POST /analyze   - Análisis NLP de texto")
    print("   • GET  /stats     - Estadísticas del sistema")
    print("   • GET  /health    - Health check")
    print("   • POST /train     - Entrenar modelo (desarrollo)")
    
    print(f"\n🌐 Servidor iniciado: http://localhost:5000")
    print("💡 Ejecuta 'python train_model.py' para entrenar los modelos ML")
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=Config.DEBUG,
        threaded=True
    )