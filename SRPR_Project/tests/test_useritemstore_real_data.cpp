#include "../include/UserItemStore.h"
#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <chrono>

int main() {
    std::cout << "=== Prueba de UserItemStore con Dataset Completo ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === CONFIGURACIÓN ===
    int dimensions = 32;  // Dimensiones más realistas para el modelo
    std::string training_file = "data/training_triplets.csv";
    std::string validation_file = "data/validation_triplets.csv";
    
    std::cout << "\nConfiguración de la prueba:" << std::endl;
    std::cout << "  - Dimensiones de vectores: " << dimensions << std::endl;
    std::cout << "  - Archivo de entrenamiento: " << training_file << std::endl;
    std::cout << "  - Archivo de validación: " << validation_file << std::endl;
    
    // === PASO 1: Cargar dataset de entrenamiento ===
    std::cout << "\n--- Paso 1: Cargando dataset de entrenamiento ---" << std::endl;
    
    std::vector<Triplet> training_triplets = load_triplets(training_file);
    
    if (training_triplets.empty()) {
        std::cerr << "ERROR: No se pudo cargar el dataset de entrenamiento." << std::endl;
        std::cerr << "Ejecuta primero: ./generate_training_data" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Cargadas " << training_triplets.size() << " tripletas de entrenamiento" << std::endl;
    
    // === PASO 2: Inicializar UserItemStore ===
    std::cout << "\n--- Paso 2: Inicializando UserItemStore ---" << std::endl;
    
    UserItemStore store(dimensions);
    
    auto init_start = std::chrono::high_resolution_clock::now();
    store.initialize(training_triplets);
    auto init_end = std::chrono::high_resolution_clock::now();
    
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
    
    store.print_summary();
    std::cout << "✓ Inicialización completada en " << init_duration.count() << " ms" << std::endl;
    
    // === PASO 3: Análisis del dataset ===
    std::cout << "\n--- Paso 3: Análisis del dataset ---" << std::endl;
    
    std::set<int> unique_users, unique_items;
    std::map<int, int> user_triplet_count;
    
    for (const auto& t : training_triplets) {
        unique_users.insert(t.user_id);
        unique_items.insert(t.preferred_item_id);
        unique_items.insert(t.less_preferred_item_id);
        user_triplet_count[t.user_id]++;
    }
    
    std::cout << "Estadísticas del dataset:" << std::endl;
    std::cout << "  ✓ Usuarios únicos: " << unique_users.size() << std::endl;
    std::cout << "  ✓ Items únicos: " << unique_items.size() << std::endl;
    std::cout << "  ✓ Tripletas totales: " << training_triplets.size() << std::endl;
    std::cout << "  ✓ Promedio tripletas/usuario: " << (double)training_triplets.size() / unique_users.size() << std::endl;
    
    // Distribución de tripletas por usuario
    int min_triplets = INT_MAX, max_triplets = 0;
    for (const auto& pair : user_triplet_count) {
        min_triplets = std::min(min_triplets, pair.second);
        max_triplets = std::max(max_triplets, pair.second);
    }
    
    std::cout << "  ✓ Rango tripletas/usuario: [" << min_triplets << ", " << max_triplets << "]" << std::endl;
    
    // === PASO 4: Verificar acceso a vectores ===
    std::cout << "\n--- Paso 4: Verificando acceso a vectores ---" << std::endl;
    
    int access_errors = 0;
    int sample_size = std::min(100, (int)unique_users.size());
    
    auto access_start = std::chrono::high_resolution_clock::now();
    
    int checked_users = 0;
    for (int user_id : unique_users) {
        if (checked_users >= sample_size) break;
        
        try {
            const Vector& user_vec = store.get_user_vector(user_id);
            if (user_vec.size() != dimensions) {
                access_errors++;
            }
            checked_users++;
        } catch (const std::exception& e) {
            access_errors++;
        }
    }
    
    int checked_items = 0;
    for (int item_id : unique_items) {
        if (checked_items >= sample_size) break;
        
        try {
            const Vector& item_vec = store.get_item_vector(item_id);
            if (item_vec.size() != dimensions) {
                access_errors++;
            }
            checked_items++;
        } catch (const std::exception& e) {
            access_errors++;
        }
    }
    
    auto access_end = std::chrono::high_resolution_clock::now();
    auto access_duration = std::chrono::duration_cast<std::chrono::microseconds>(access_end - access_start);
    
    if (access_errors == 0) {
        std::cout << "✓ Acceso exitoso a " << checked_users << " usuarios y " << checked_items << " items" << std::endl;
        std::cout << "✓ Tiempo de acceso: " << access_duration.count() << " μs" << std::endl;
    } else {
        std::cout << "❌ " << access_errors << " errores de acceso encontrados" << std::endl;
        return 1;
    }
    
    // === PASO 5: Análisis estadístico de vectores ===
    std::cout << "\n--- Paso 5: Análisis estadístico de vectores ---" << std::endl;
    
    std::vector<double> user_vector_norms, item_vector_norms;
    double user_values_sum = 0.0, item_values_sum = 0.0;
    int user_values_count = 0, item_values_count = 0;
    
    // Analizar muestra de vectores de usuarios
    int analyzed_users = 0;
    for (int user_id : unique_users) {
        if (analyzed_users >= 50) break;  // Analizar solo 50 usuarios para eficiencia
        
        const Vector& vec = store.get_user_vector(user_id);
        
        // Calcular norma
        double norm = 0.0;
        for (double val : vec) {
            norm += val * val;
            user_values_sum += val;
            user_values_count++;
        }
        user_vector_norms.push_back(std::sqrt(norm));
        analyzed_users++;
    }
    
    // Analizar muestra de vectores de items
    int analyzed_items = 0;
    for (int item_id : unique_items) {
        if (analyzed_items >= 50) break;  // Analizar solo 50 items para eficiencia
        
        const Vector& vec = store.get_item_vector(item_id);
        
        // Calcular norma
        double norm = 0.0;
        for (double val : vec) {
            norm += val * val;
            item_values_sum += val;
            item_values_count++;
        }
        item_vector_norms.push_back(std::sqrt(norm));
        analyzed_items++;
    }
    
    double user_mean = user_values_sum / user_values_count;
    double item_mean = item_values_sum / item_values_count;
    
    // Calcular estadísticas de normas
    double user_norm_sum = 0.0, item_norm_sum = 0.0;
    for (double norm : user_vector_norms) user_norm_sum += norm;
    for (double norm : item_vector_norms) item_norm_sum += norm;
    
    double avg_user_norm = user_norm_sum / user_vector_norms.size();
    double avg_item_norm = item_norm_sum / item_vector_norms.size();
    
    std::cout << "Estadísticas de vectores (muestra de 50):" << std::endl;
    std::cout << "  Usuarios:" << std::endl;
    std::cout << "    - Media de valores: " << user_mean << std::endl;
    std::cout << "    - Norma promedio: " << avg_user_norm << std::endl;
    std::cout << "  Items:" << std::endl;
    std::cout << "    - Media de valores: " << item_mean << std::endl;
    std::cout << "    - Norma promedio: " << avg_item_norm << std::endl;
    
    // === PASO 6: Prueba de modificación de vectores ===
    std::cout << "\n--- Paso 6: Prueba de modificación de vectores ---" << std::endl;
    
    if (!unique_users.empty()) {
        int test_user = *unique_users.begin();
        Vector& user_vec = store.get_user_vector(test_user);
        
        std::vector<double> original_values = user_vec;  // Copia
        
        // Modificar algunos valores
        for (int i = 0; i < std::min(5, (int)user_vec.size()); ++i) {
            user_vec[i] += 0.1;
        }
        
        // Verificar que los cambios persistieron
        const Vector& modified_vec = store.get_user_vector(test_user);
        bool changes_persisted = true;
        for (int i = 0; i < std::min(5, (int)modified_vec.size()); ++i) {
            if (std::abs(modified_vec[i] - (original_values[i] + 0.1)) > 1e-10) {
                changes_persisted = false;
                break;
            }
        }
        
        if (changes_persisted) {
            std::cout << "✓ Modificaciones de vectores persisten correctamente" << std::endl;
            
            // Restaurar valores originales
            for (int i = 0; i < std::min(5, (int)user_vec.size()); ++i) {
                user_vec[i] = original_values[i];
            }
        } else {
            std::cout << "❌ Error: Las modificaciones no persistieron" << std::endl;
            return 1;
        }
    }
    
    // === PASO 7: Cargar y verificar dataset de validación ===
    std::cout << "\n--- Paso 7: Verificando compatibilidad con dataset de validación ---" << std::endl;
    
    std::vector<Triplet> validation_triplets = load_triplets(validation_file);
    
    if (!validation_triplets.empty()) {
        std::cout << "✓ Cargadas " << validation_triplets.size() << " tripletas de validación" << std::endl;
        
        // Verificar que todos los usuarios e items de validación existen en el store
        int missing_users = 0, missing_items = 0;
        std::set<int> validation_users, validation_items;
        
        for (const auto& t : validation_triplets) {
            validation_users.insert(t.user_id);
            validation_items.insert(t.preferred_item_id);
            validation_items.insert(t.less_preferred_item_id);
        }
        
        for (int user_id : validation_users) {
            try {
                store.get_user_vector(user_id);
            } catch (const std::out_of_range&) {
                missing_users++;
            }
        }
        
        for (int item_id : validation_items) {
            try {
                store.get_item_vector(item_id);
            } catch (const std::out_of_range&) {
                missing_items++;
            }
        }
        
        std::cout << "Compatibilidad con validación:" << std::endl;
        std::cout << "  - Usuarios en validación: " << validation_users.size() << std::endl;
        std::cout << "  - Items en validación: " << validation_items.size() << std::endl;
        std::cout << "  - Usuarios faltantes: " << missing_users << std::endl;
        std::cout << "  - Items faltantes: " << missing_items << std::endl;
        
        if (missing_users == 0 && missing_items == 0) {
            std::cout << "✓ Completa compatibilidad con dataset de validación" << std::endl;
        } else {
            std::cout << "⚠️ Hay entidades en validación que no están en entrenamiento" << std::endl;
        }
    } else {
        std::cout << "⚠️ No se pudo cargar el dataset de validación" << std::endl;
    }
    
    // === PASO 8: Benchmark de rendimiento ===
    std::cout << "\n--- Paso 8: Benchmark de rendimiento ---" << std::endl;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    // Simular accesos típicos durante entrenamiento
    int benchmark_iterations = 1000;
    int operations = 0;
    
    for (int i = 0; i < benchmark_iterations; ++i) {
        for (const auto& t : training_triplets) {
            if (operations >= 10000) break;  // Limitar para que no tome demasiado tiempo
            
            // Simular acceso típico durante entrenamiento
            const Vector& user_vec = store.get_user_vector(t.user_id);
            const Vector& item1_vec = store.get_item_vector(t.preferred_item_id);
            const Vector& item2_vec = store.get_item_vector(t.less_preferred_item_id);
            
            // Operación simple para evitar optimización del compilador
            if (user_vec[0] + item1_vec[0] + item2_vec[0] > 999999) {
                std::cout << "unlikely";
            }
            
            operations += 3;  // 3 accesos por tripleta
        }
        if (operations >= 10000) break;
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    auto bench_duration = std::chrono::duration_cast<std::chrono::microseconds>(bench_end - bench_start);
    
    std::cout << "Benchmark de rendimiento:" << std::endl;
    std::cout << "  - " << operations << " accesos a vectores en " << bench_duration.count() << " μs" << std::endl;
    std::cout << "  - " << (operations * 1000000.0 / bench_duration.count()) << " accesos/segundo" << std::endl;
    std::cout << "  - " << (bench_duration.count() / (double)operations) << " μs por acceso" << std::endl;
    
    // === RESUMEN FINAL ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== RESUMEN FINAL ===" << std::endl;
    std::cout << "🎉 ¡Prueba completa de UserItemStore exitosa!" << std::endl;
    std::cout << "⏱️  Tiempo total: " << total_duration.count() << " ms" << std::endl;
    
    std::cout << "\n📊 Estadísticas finales:" << std::endl;
    std::cout << "   - Usuarios gestionados: " << unique_users.size() << std::endl;
    std::cout << "   - Items gestionados: " << unique_items.size() << std::endl;
    std::cout << "   - Dimensiones por vector: " << dimensions << std::endl;
    std::cout << "   - Memoria estimada: " << 
        ((unique_users.size() + unique_items.size()) * dimensions * sizeof(double) / 1024 / 1024) 
        << " MB" << std::endl;
    
    std::cout << "\n✅ UserItemStore está completamente preparado para:" << std::endl;
    std::cout << "   ✓ Entrenamiento con " << training_triplets.size() << " tripletas" << std::endl;
    std::cout << "   ✓ Validación con " << validation_triplets.size() << " tripletas" << std::endl;
    std::cout << "   ✓ Acceso eficiente durante optimización" << std::endl;
    std::cout << "   ✓ Modificación de vectores durante gradiente descent" << std::endl;
    
    std::cout << "\n🚀 ¡Listo para el siguiente paso: LSH Hasher!" << std::endl;
    
    return 0;
}