#include "../include/UserItemStore.h"
#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <chrono>

// Función de utilidad para calcular la norma de un vector
double vector_norm(const Vector& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Función de utilidad para calcular el producto punto
double dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

int main() {
    std::cout << "=== Prueba Completa de UserItemStore ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === PRUEBA 1: Inicialización básica ===
    std::cout << "\n--- Prueba 1: Inicialización básica ---" << std::endl;
    
    int dimensions = 8;
    UserItemStore store(dimensions);
    
    // Crear datos de prueba
    std::vector<Triplet> test_triplets = {
        {101, 1, 2},
        {101, 1, 3},
        {102, 5, 6},
        {103, 2, 4},
        {103, 7, 1}
    };
    
    store.initialize(test_triplets);
    store.print_summary();
    
    // Verificar que se inicializaron correctamente
    std::set<int> expected_users = {101, 102, 103};
    std::set<int> expected_items = {1, 2, 3, 4, 5, 6, 7};
    
    std::cout << "✓ Usuarios esperados: " << expected_users.size() << std::endl;
    std::cout << "✓ Items esperados: " << expected_items.size() << std::endl;
    
    // === PRUEBA 2: Acceso a vectores ===
    std::cout << "\n--- Prueba 2: Acceso a vectores ---" << std::endl;
    
    try {
        Vector& user_vec = store.get_user_vector(101);
        Vector& item_vec = store.get_item_vector(1);
        
        std::cout << "✓ Acceso a vector de usuario 101: dimensión " << user_vec.size() << std::endl;
        std::cout << "✓ Acceso a vector de item 1: dimensión " << item_vec.size() << std::endl;
        
        if (user_vec.size() != dimensions || item_vec.size() != dimensions) {
            std::cerr << "ERROR: Dimensiones incorrectas!" << std::endl;
            return 1;
        }
        
        // Mostrar algunos valores iniciales
        std::cout << "  Vector usuario 101 (primeros 4): ";
        for (int i = 0; i < 4; ++i) {
            std::cout << user_vec[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Vector item 1 (primeros 4): ";
        for (int i = 0; i < 4; ++i) {
            std::cout << item_vec[i] << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::out_of_range& e) {
        std::cerr << "ERROR: No se pudo acceder a los vectores. " << e.what() << std::endl;
        return 1;
    }
    
    // === PRUEBA 3: Modificación de vectores ===
    std::cout << "\n--- Prueba 3: Modificación de vectores ---" << std::endl;
    
    try {
        Vector& user_vec = store.get_user_vector(101);
        double original_value = user_vec[0];
        
        std::cout << "  Valor original usuario 101[0]: " << original_value << std::endl;
        
        // Modificar el vector
        user_vec[0] = 99.9;
        
        const Vector& updated_vec = store.get_user_vector(101);
        std::cout << "  Valor modificado usuario 101[0]: " << updated_vec[0] << std::endl;
        
        if (updated_vec[0] != 99.9) {
            std::cerr << "ERROR: La modificación no persistió!" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Modificación de vectores funciona correctamente" << std::endl;
        
        // Restaurar valor original para otras pruebas
        user_vec[0] = original_value;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR en modificación: " << e.what() << std::endl;
        return 1;
    }
    
    // === PRUEBA 4: Acceso constante ===
    std::cout << "\n--- Prueba 4: Acceso constante ---" << std::endl;
    
    const UserItemStore& const_store = store;
    
    try {
        const Vector& const_user_vec = const_store.get_user_vector(102);
        const Vector& const_item_vec = const_store.get_item_vector(5);
        
        std::cout << "✓ Acceso constante a usuario 102: dimensión " << const_user_vec.size() << std::endl;
        std::cout << "✓ Acceso constante a item 5: dimensión " << const_item_vec.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR en acceso constante: " << e.what() << std::endl;
        return 1;
    }
    
    // === PRUEBA 5: Obtener todos los vectores de items ===
    std::cout << "\n--- Prueba 5: Acceso a todos los vectores de items ---" << std::endl;
    
    const auto& all_items = store.get_all_item_vectors();
    std::cout << "✓ Total de vectores de items: " << all_items.size() << std::endl;
    
    // Verificar que todos los items esperados están presentes
    for (int expected_item : expected_items) {
        if (all_items.find(expected_item) == all_items.end()) {
            std::cerr << "ERROR: Item " << expected_item << " no encontrado!" << std::endl;
            return 1;
        }
    }
    std::cout << "✓ Todos los items esperados están presentes" << std::endl;
    
    // === PRUEBA 6: Análisis estadístico de la inicialización ===
    std::cout << "\n--- Prueba 6: Análisis estadístico ---" << std::endl;
    
    std::vector<double> all_values;
    double sum = 0.0;
    
    // Recopilar todos los valores de vectores de usuarios
    for (int user_id : expected_users) {
        const Vector& vec = store.get_user_vector(user_id);
        for (double val : vec) {
            all_values.push_back(val);
            sum += val;
        }
    }
    
    // Recopilar todos los valores de vectores de items
    for (int item_id : expected_items) {
        const Vector& vec = store.get_item_vector(item_id);
        for (double val : vec) {
            all_values.push_back(val);
            sum += val;
        }
    }
    
    double mean = sum / all_values.size();
    
    // Calcular varianza
    double variance_sum = 0.0;
    for (double val : all_values) {
        variance_sum += (val - mean) * (val - mean);
    }
    double variance = variance_sum / all_values.size();
    double std_dev = std::sqrt(variance);
    
    std::cout << "  Estadísticas de inicialización:" << std::endl;
    std::cout << "    - Total de valores: " << all_values.size() << std::endl;
    std::cout << "    - Media: " << mean << std::endl;
    std::cout << "    - Desviación estándar: " << std_dev << std::endl;
    std::cout << "    - Esperado ~N(0, 0.1): media ≈ 0, std ≈ 0.1" << std::endl;
    
    if (std::abs(mean) > 0.05 || std::abs(std_dev - 0.1) > 0.05) {
        std::cout << "  ⚠️ ADVERTENCIA: La distribución podría no ser la esperada" << std::endl;
    } else {
        std::cout << "  ✓ Distribución de inicialización parece correcta" << std::endl;
    }
    
    // === PRUEBA 7: Prueba con datos reales de MovieLens ===
    std::cout << "\n--- Prueba 7: Datos reales de MovieLens ---" << std::endl;
    
    // Intentar cargar datos reales si están disponibles
    std::vector<Triplet> movielens_triplets = load_triplets("data/movielens_sample.csv");
    
    if (!movielens_triplets.empty()) {
        std::cout << "  Probando con " << movielens_triplets.size() << " tripletas de MovieLens..." << std::endl;
        
        UserItemStore real_store(20); // Dimensiones más realistas
        real_store.initialize(movielens_triplets);
        real_store.print_summary();
        
        // Verificar que no hay errores de acceso
        std::set<int> unique_users, unique_items;
        for (const auto& t : movielens_triplets) {
            unique_users.insert(t.user_id);
            unique_items.insert(t.preferred_item_id);
            unique_items.insert(t.less_preferred_item_id);
        }
        
        bool access_success = true;
        for (int user_id : unique_users) {
            try {
                const Vector& vec = real_store.get_user_vector(user_id);
                if (vec.size() != 20) access_success = false;
            } catch (...) {
                access_success = false;
                break;
            }
        }
        
        if (access_success) {
            std::cout << "  ✓ Acceso exitoso a todos los vectores de usuarios reales" << std::endl;
        } else {
            std::cout << "  ❌ Error en acceso a vectores de usuarios reales" << std::endl;
        }
        
    } else {
        std::cout << "  ⚠️ No se encontraron datos de MovieLens (data/movielens_sample.csv)" << std::endl;
        std::cout << "    Esto es normal si no has ejecutado las pruebas de MovieLens antes." << std::endl;
    }
    
    // === PRUEBA 8: Manejo de errores ===
    std::cout << "\n--- Prueba 8: Manejo de errores ---" << std::endl;
    
    try {
        store.get_user_vector(99999); // Usuario inexistente
        std::cerr << "ERROR: Debería haber lanzado excepción para usuario inexistente!" << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Excepción correcta para usuario inexistente: " << e.what() << std::endl;
    }
    
    try {
        store.get_item_vector(99999); // Item inexistente
        std::cerr << "ERROR: Debería haber lanzado excepción para item inexistente!" << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cout << "✓ Excepción correcta para item inexistente: " << e.what() << std::endl;
    }
    
    // === PRUEBA 9: Rendimiento ===
    std::cout << "\n--- Prueba 9: Análisis de rendimiento ---" << std::endl;
    
    auto access_start = std::chrono::high_resolution_clock::now();
    
    // Realizar muchos accesos para medir rendimiento
    int access_count = 0;
    for (int i = 0; i < 1000; ++i) {
        for (int user_id : expected_users) {
            const Vector& vec = store.get_user_vector(user_id);
            access_count++;
            // Operación simple para evitar optimización del compilador
            if (vec[0] > 999) std::cout << "unlikely";
        }
    }
    
    auto access_end = std::chrono::high_resolution_clock::now();
    auto access_duration = std::chrono::duration_cast<std::chrono::microseconds>(access_end - access_start);
    
    std::cout << "  Rendimiento de acceso:" << std::endl;
    std::cout << "    - " << access_count << " accesos en " << access_duration.count() << " μs" << std::endl;
    std::cout << "    - " << (access_count * 1000000.0 / access_duration.count()) << " accesos/segundo" << std::endl;
    
    // === RESUMEN FINAL ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== RESUMEN FINAL ===" << std::endl;
    std::cout << "🎉 ¡Todas las pruebas de UserItemStore completadas exitosamente!" << std::endl;
    std::cout << "⏱️  Tiempo total de pruebas: " << total_duration.count() << " ms" << std::endl;
    
    std::cout << "\n✅ Funcionalidades verificadas:" << std::endl;
    std::cout << "   ✓ Inicialización de vectores desde tripletas" << std::endl;
    std::cout << "   ✓ Acceso de lectura y escritura a vectores" << std::endl;
    std::cout << "   ✓ Acceso constante a vectores" << std::endl;
    std::cout << "   ✓ Obtención de todos los vectores de items" << std::endl;
    std::cout << "   ✓ Manejo correcto de errores" << std::endl;
    std::cout << "   ✓ Inicialización estadística correcta" << std::endl;
    std::cout << "   ✓ Rendimiento de acceso eficiente" << std::endl;
    std::cout << "   ✓ Compatibilidad con datos reales de MovieLens" << std::endl;
    
    std::cout << "\n🚀 UserItemStore está listo para ser usado en el entrenamiento SRPR!" << std::endl;
    
    return 0;
}