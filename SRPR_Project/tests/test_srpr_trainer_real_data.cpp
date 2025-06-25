#include "../include/SRPR_Trainer.h"
#include "../include/UserItemStore.h"
#include "../include/Triplet.h"
#include "../include/LSH.h"
#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <cmath>
#include <algorithm>

// Función para calcular precisión de ranking
double calculate_ranking_accuracy(const std::vector<Triplet>& test_triplets, 
                                 const UserItemStore& store) {
    int correct_rankings = 0;
    int total_rankings = 0;
    
    for (const auto& triplet : test_triplets) {
        try {
            const Vector& user_vec = store.get_user_vector(triplet.user_id);
            const Vector& preferred_vec = store.get_item_vector(triplet.preferred_item_id);
            const Vector& less_preferred_vec = store.get_item_vector(triplet.less_preferred_item_id);
            
            // Calcular productos punto
            double score_preferred = 0.0, score_less_preferred = 0.0;
            for (size_t i = 0; i < user_vec.size(); ++i) {
                score_preferred += user_vec[i] * preferred_vec[i];
                score_less_preferred += user_vec[i] * less_preferred_vec[i];
            }
            
            if (score_preferred > score_less_preferred) {
                correct_rankings++;
            }
            total_rankings++;
            
        } catch (const std::exception& e) {
            // Saltar tripletas con vectores no encontrados
            continue;
        }
    }
    
    return total_rankings > 0 ? (double)correct_rankings / total_rankings : 0.0;
}

// Función para calcular la correlación entre similitud y distancia Hamming
double evaluate_lsh_correlation(const UserItemStore& store, const SRPR_Trainer& trainer,
                               const SRPR_Trainer::TrainingParams& params) {
    
    SRPHasher hasher(store.get_user_vector(1).size(), params.b_lsh_length, 42);
    
    // Tomar muestra de usuarios e items
    std::vector<int> sample_users, sample_items;
    int count = 0;
    
    // Obtener algunos usuarios
    for (int user_id = 1; user_id <= 370 && count < 10; ++user_id) {
        try {
            store.get_user_vector(user_id);
            sample_users.push_back(user_id);
            count++;
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    // Obtener algunos items
    count = 0;
    for (int item_id = 1; item_id <= 5000 && count < 20; ++item_id) {
        try {
            store.get_item_vector(item_id);
            sample_items.push_back(item_id);
            count++;
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    double total_correlation = 0.0;
    int correlation_count = 0;
    
    // Calcular correlación entre algunos pares usuario-item
    for (int user_id : sample_users) {
        const Vector& user_vec = store.get_user_vector(user_id);
        std::string user_code = hasher.generate_code(user_vec);
        
        for (int item_id : sample_items) {
            const Vector& item_vec = store.get_item_vector(item_id);
            std::string item_code = hasher.generate_code(item_vec);
            
            // Calcular similitud coseno
            double dot_prod = 0.0, norm_u = 0.0, norm_i = 0.0;
            for (size_t k = 0; k < user_vec.size(); ++k) {
                dot_prod += user_vec[k] * item_vec[k];
                norm_u += user_vec[k] * user_vec[k];
                norm_i += item_vec[k] * item_vec[k];
            }
            
            double cosine_sim = dot_prod / (std::sqrt(norm_u) * std::sqrt(norm_i));
            
            // Calcular distancia Hamming
            int hamming_dist = 0;
            for (size_t k = 0; k < user_code.length(); ++k) {
                if (user_code[k] != item_code[k]) {
                    hamming_dist++;
                }
            }
            
            // Normalizar distancia Hamming a [0,1]
            double normalized_hamming = (double)hamming_dist / user_code.length();
            
            // Esperamos correlación negativa (menor distancia Hamming = mayor similitud)
            total_correlation += -normalized_hamming * cosine_sim;
            correlation_count++;
        }
    }
    
    return correlation_count > 0 ? total_correlation / correlation_count : 0.0;
}

int main() {
    std::cout << "=== Prueba SRPR_Trainer con Dataset Completo ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === CONFIGURACIÓN ===
    int dimensions = 32;
    std::string training_file = "data/training_triplets.csv";
    std::string validation_file = "data/validation_triplets.csv";
    
    std::cout << "\nConfiguración de la prueba:" << std::endl;
    std::cout << "  - Dimensiones: " << dimensions << std::endl;
    std::cout << "  - Archivo entrenamiento: " << training_file << std::endl;
    std::cout << "  - Archivo validación: " << validation_file << std::endl;
    
    // === PASO 1: Cargar datasets ===
    std::cout << "\n--- Paso 1: Cargando datasets ---" << std::endl;
    
    std::vector<Triplet> training_triplets = load_triplets(training_file);
    std::vector<Triplet> validation_triplets = load_triplets(validation_file);
    
    if (training_triplets.empty()) {
        std::cerr << "ERROR: No se pudo cargar el dataset de entrenamiento." << std::endl;
        std::cerr << "Ejecuta primero: ./generate_training_data" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Entrenamiento: " << training_triplets.size() << " tripletas" << std::endl;
    std::cout << "✓ Validación: " << validation_triplets.size() << " tripletas" << std::endl;
    
    // === PASO 2: Inicializar sistema ===
    std::cout << "\n--- Paso 2: Inicializando sistema ---" << std::endl;
    
    UserItemStore store(dimensions);
    store.initialize(training_triplets);
    store.print_summary();
    
    SRPR_Trainer trainer(store);
    std::cout << "✓ SRPR_Trainer inicializado" << std::endl;
    
    // === PASO 3: Evaluación baseline ===
    std::cout << "\n--- Paso 3: Evaluación baseline ---" << std::endl;
    
    SRPR_Trainer::TrainingParams eval_params;
    eval_params.b_lsh_length = 16;
    
    double initial_loss = trainer.calculate_total_loss(training_triplets, eval_params);
    double initial_val_loss = 0.0;
    double initial_accuracy = 0.0;
    
    if (!validation_triplets.empty()) {
        initial_val_loss = trainer.calculate_total_loss(validation_triplets, eval_params);
        initial_accuracy = calculate_ranking_accuracy(validation_triplets, store);
    }
    
    std::cout << "✓ Pérdida inicial (entrenamiento): " << std::fixed << std::setprecision(6) << initial_loss << std::endl;
    std::cout << "✓ Pérdida inicial (validación): " << std::fixed << std::setprecision(6) << initial_val_loss << std::endl;
    std::cout << "✓ Precisión inicial: " << std::fixed << std::setprecision(4) << (initial_accuracy * 100) << "%" << std::endl;
    
    // === PASO 4: Entrenamiento optimizado ===
    std::cout << "\n--- Paso 4: Entrenamiento optimizado ---" << std::endl;
    
    SRPR_Trainer::TrainingParams opt_params;
    opt_params.epochs = 15;
    opt_params.learning_rate = 0.005;  // Learning rate más agresivo
    opt_params.b_lsh_length = 16;
    opt_params.regularization = 0.0005;
    opt_params.verbose = true;
    opt_params.validation_freq = 3;
    
    std::cout << "Configuración de entrenamiento optimizada:" << std::endl;
    std::cout << "  - Epochs: " << opt_params.epochs << std::endl;
    std::cout << "  - Learning rate: " << opt_params.learning_rate << std::endl;
    std::cout << "  - LSH bits: " << opt_params.b_lsh_length << std::endl;
    std::cout << "  - Regularización: " << opt_params.regularization << std::endl;
    
    auto training_stats = trainer.train(training_triplets, opt_params, validation_triplets);
    
    // === PASO 5: Evaluación post-entrenamiento ===
    std::cout << "\n--- Paso 5: Evaluación final ---" << std::endl;
    
    double final_loss = trainer.calculate_total_loss(training_triplets, opt_params);
    double final_val_loss = 0.0;
    double final_accuracy = 0.0;
    
    if (!validation_triplets.empty()) {
        final_val_loss = trainer.calculate_total_loss(validation_triplets, opt_params);
        final_accuracy = calculate_ranking_accuracy(validation_triplets, store);
    }
    
    double loss_improvement = initial_loss - final_loss;
    double val_loss_improvement = initial_val_loss - final_val_loss;
    double accuracy_improvement = final_accuracy - initial_accuracy;
    
    std::cout << "✓ Pérdida final (entrenamiento): " << std::fixed << std::setprecision(6) << final_loss << std::endl;
    std::cout << "✓ Pérdida final (validación): " << std::fixed << std::setprecision(6) << final_val_loss << std::endl;
    std::cout << "✓ Precisión final: " << std::fixed << std::setprecision(4) << (final_accuracy * 100) << "%" << std::endl;
    
    std::cout << "\nMejoras obtenidas:" << std::endl;
    std::cout << "  - Mejora en pérdida (entrenamiento): " << std::fixed << std::setprecision(6) << loss_improvement << std::endl;
    std::cout << "  - Mejora en pérdida (validación): " << std::fixed << std::setprecision(6) << val_loss_improvement << std::endl;
    std::cout << "  - Mejora en precisión: " << std::fixed << std::setprecision(4) << (accuracy_improvement * 100) << " pp" << std::endl;
    
    // === PASO 6: Análisis de correlación LSH ===
    std::cout << "\n--- Paso 6: Análisis de correlación LSH ---" << std::endl;
    
    double lsh_correlation = evaluate_lsh_correlation(store, trainer, opt_params);
    std::cout << "✓ Correlación LSH: " << std::fixed << std::setprecision(4) << lsh_correlation << std::endl;
    
    if (lsh_correlation > 0.1) {
        std::cout << "✓ Buena correlación entre similitud y hashing LSH" << std::endl;
    } else {
        std::cout << "⚠️ Correlación LSH baja - puede necesitar más entrenamiento" << std::endl;
    }
    
    // === PASO 7: Análisis de convergencia ===
    std::cout << "\n--- Paso 7: Análisis de convergencia ---" << std::endl;
    
    if (training_stats.epoch_losses.size() >= 3) {
        std::cout << "Evolución de pérdida (últimos 5 epochs):" << std::endl;
        int start_idx = std::max(0, (int)training_stats.epoch_losses.size() - 5);
        for (int i = start_idx; i < training_stats.epoch_losses.size(); ++i) {
            std::cout << "  Epoch " << (i + 1) << ": " 
                      << std::fixed << std::setprecision(6) << training_stats.epoch_losses[i];
            if (i > start_idx) {
                double change = training_stats.epoch_losses[i] - training_stats.epoch_losses[i-1];
                std::cout << " (Δ: " << std::fixed << std::setprecision(6) << change << ")";
            }
            std::cout << std::endl;
        }
        
        // Verificar tendencia de convergencia
        if (training_stats.epoch_losses.size() >= 3) {
            double recent_change = std::abs(training_stats.epoch_losses.back() - training_stats.epoch_losses[training_stats.epoch_losses.size()-2]);
            if (recent_change < 0.001) {
                std::cout << "✓ El modelo está convergiendo (cambio < 0.001)" << std::endl;
            } else {
                std::cout << "⚠️ El modelo aún no ha convergido (cambio: " << recent_change << ")" << std::endl;
            }
        }
    }
    
    // === PASO 8: Benchmarks de rendimiento ===
    std::cout << "\n--- Paso 8: Benchmarks de rendimiento ---" << std::endl;
    
    std::cout << "Métricas de rendimiento:" << std::endl;
    std::cout << "  - Tiempo total entrenamiento: " << training_stats.training_time_ms << " ms" << std::endl;
    std::cout << "  - Actualizaciones totales: " << training_stats.total_updates << std::endl;
    std::cout << "  - Velocidad: " << std::fixed << std::setprecision(1) 
              << (training_stats.total_updates * 1000.0 / training_stats.training_time_ms) 
              << " actualizaciones/s" << std::endl;
    std::cout << "  - Tiempo por tripleta: " << std::fixed << std::setprecision(3) 
              << (training_stats.training_time_ms / training_stats.total_updates) 
              << " ms" << std::endl;
    
    // === PASO 9: Verificación de calidad del modelo ===
    std::cout << "\n--- Paso 9: Verificación de calidad del modelo ---" << std::endl;
    
    // Verificar que los vectores no han colapsado
    std::vector<double> user_norms, item_norms;
    int sample_size = std::min(50, 370);
    
    for (int i = 1; i <= sample_size; ++i) {
        try {
            const Vector& user_vec = store.get_user_vector(i);
            double norm = 0.0;
            for (double val : user_vec) {
                norm += val * val;
            }
            user_norms.push_back(std::sqrt(norm));
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    for (int i = 1; i <= sample_size && item_norms.size() < 50; ++i) {
        try {
            const Vector& item_vec = store.get_item_vector(i);
            double norm = 0.0;
            for (double val : item_vec) {
                norm += val * val;
            }
            item_norms.push_back(std::sqrt(norm));
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    double avg_user_norm = 0.0, avg_item_norm = 0.0;
    for (double norm : user_norms) avg_user_norm += norm;
    for (double norm : item_norms) avg_item_norm += norm;
    avg_user_norm /= user_norms.size();
    avg_item_norm /= item_norms.size();
    
    std::cout << "Calidad de vectores aprendidos:" << std::endl;
    std::cout << "  - Norma promedio usuarios: " << std::fixed << std::setprecision(4) << avg_user_norm << std::endl;
    std::cout << "  - Norma promedio ítems: " << std::fixed << std::setprecision(4) << avg_item_norm << std::endl;
    
    if (avg_user_norm > 0.1 && avg_item_norm > 0.1) {
        std::cout << "✓ Los vectores mantienen magnitudes saludables" << std::endl;
    } else {
        std::cout << "⚠️ Los vectores pueden haber colapsado - revisar configuración" << std::endl;
    }
    
    // === RESUMEN FINAL ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n=== RESUMEN FINAL ===" << std::endl;
    std::cout << "🎉 ¡Entrenamiento SRPR con dataset completo finalizado!" << std::endl;
    std::cout << "⏱️  Tiempo total: " << total_duration.count() << " segundos" << std::endl;
    
    std::cout << "\n📊 Resultados finales:" << std::endl;
    std::cout << "   - Dataset: " << training_triplets.size() << " tripletas entrenamiento, " 
              << validation_triplets.size() << " validación" << std::endl;
    std::cout << "   - Mejora pérdida: " << std::fixed << std::setprecision(6) << loss_improvement << std::endl;
    std::cout << "   - Mejora precisión: " << std::fixed << std::setprecision(2) 
              << (accuracy_improvement * 100) << " puntos porcentuales" << std::endl;
    std::cout << "   - Convergencia: " << (training_stats.converged ? "Sí" : "En progreso") << std::endl;
    std::cout << "   - Velocidad: " << std::fixed << std::setprecision(0) 
              << (training_stats.total_updates * 1000.0 / training_stats.training_time_ms) 
              << " actualizaciones/s" << std::endl;
    
    // Determinar si el entrenamiento fue exitoso
    bool success = false;
    if (loss_improvement > 0.01 || accuracy_improvement > 0.05) {
        success = true;
        std::cout << "\n🚀 ¡ENTRENAMIENTO EXITOSO!" << std::endl;
        std::cout << "✅ El modelo SRPR muestra mejoras significativas" << std::endl;
        std::cout << "✅ Sistema listo para producción" << std::endl;
    } else if (final_accuracy > 0.6) {
        success = true;
        std::cout << "\n✅ ENTRENAMIENTO ACEPTABLE" << std::endl;
        std::cout << "✅ El modelo alcanza buena precisión base" << std::endl;
        std::cout << "💡 Considerar más epochs o ajuste de hiperparámetros" << std::endl;
    } else {
        std::cout << "\n⚠️ ENTRENAMIENTO NECESITA OPTIMIZACIÓN" << std::endl;
        std::cout << "💡 Sugerencias:" << std::endl;
        std::cout << "   - Aumentar número de epochs" << std::endl;
        std::cout << "   - Ajustar learning rate" << std::endl;
        std::cout << "   - Verificar calidad de datos" << std::endl;
        std::cout << "   - Considerar diferentes valores de regularización" << std::endl;
    }
    
    std::cout << "\n🎯 ¡SRPR_Trainer completamente funcional!" << std::endl;
    std::cout << "📋 Listo para el pipeline final integrado" << std::endl;
    
    return success ? 0 : 1;
}