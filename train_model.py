#!/usr/bin/env python3
"""
Script para entrenar modelos de NLP y ML del chatbot
"""

import json
import os
import sys
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.intent_classifier import intent_classifier
from chatbot.utils.text_processor import text_processor
from config import Config

def main():
    print("🚀 Iniciando entrenamiento de modelos NLP...")
    
    # Asegurar que existen los directorios
    Config.ensure_directories()
    
    # Cargar datos de intenciones
    try:
        with open(Config.INTENTS_FILE, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        
        intents = intents_data.get("intents", [])
        print(f"📁 Cargados {len(intents)} intents desde {Config.INTENTS_FILE}")
        
    except Exception as e:
        print(f"❌ Error cargando intents: {e}")
        return
    
    # Preparar datos de entrenamiento para vectorizadores
    print("🔧 Preparando datos para vectorizadores...")
    all_patterns = []
    for intent in intents:
        all_patterns.extend(intent["patterns"])
    
    text_processor.fit_vectorizers(all_patterns)
    print(f"✅ Vectorizadores entrenados con {len(all_patterns)} patrones")
    
    # Entrenar modelo de clasificación
    print("🤖 Entrenando clasificador de intenciones...")
    training_results = intent_classifier.train_model(intents)
    
    if "error" in training_results:
        print(f"❌ Error en entrenamiento: {training_results['error']}")
        return
    
    print("\n📊 RESULTADOS DEL ENTRENAMIENTO:")
    print(f"   • Accuracy: {training_results['accuracy']:.3f}")
    print(f"   • Ejemplos entrenamiento: {training_results['training_size']}")
    print(f"   • Ejemplos prueba: {training_results['test_size']}")
    print(f"   • Clases: {training_results['num_classes']}")
    print(f"   • Características: {training_results['feature_count']}")
    print(f"   • Clasificador: {training_results['classifier_type']}")
    
    # Guardar modelos
    print("💾 Guardando modelos...")
    intent_classifier.save_model()
    
    # Probar el modelo con algunos ejemplos
    print("\n🧪 Probando modelo con ejemplos...")
    test_examples = [
        "Hola, buen día",
        "Esto es todo",
        "Cuanto necesito para pasar el curso?",
        "Donde puedo ver mi calificacion?",
        "Quiero aprender la prepa"
    ]
    
    for example in test_examples:
        try:
            tag, confidence = intent_classifier.predict_intent(example)
            print(f"   '{example}' -> {tag} (conf: {confidence:.3f})")
        except Exception as e:
            print(f"   '{example}' -> Error: {e}")
    
    print(f"\n✅ Entrenamiento completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()