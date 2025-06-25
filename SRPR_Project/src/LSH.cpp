#include "../include/LSH.h"
#include <iostream>
#include <iomanip>

// === Implementación de la clase base LSH ===

std::string LSH::generate_code(const Vector& vec) const {
    std::string code = "";
    code.reserve(b);  // Reservar espacio para eficiencia
    
    for (int i = 0; i < b; ++i) {
        code += hash_to_bit(vec, i);
    }
    return code;
}

// === Implementación de SRPHasher ===

SRPHasher::SRPHasher(int dimensions, int num_hashes, unsigned int seed) 
    : LSH(dimensions, num_hashes), initialized(false) {
    
    // Usar seed 0 significa usar random_device, cualquier otro valor usa seed fijo
    std::mt19937 rng;
    if (seed == 0) {
        rng.seed(std::random_device{}());
    } else {
        rng.seed(seed);
    }
    
    std::normal_distribution<double> dist(0.0, 1.0);

    random_vectors.resize(num_hashes);
    for (int i = 0; i < num_hashes; ++i) {
        random_vectors[i].resize(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            random_vectors[i][j] = dist(rng);
        }
    }
    
    initialized = true;
}

char SRPHasher::hash_to_bit(const Vector& vec, int hash_function_index) const {
    // Verificaciones silenciosas para mayor eficiencia
    if (!initialized || hash_function_index < 0 || hash_function_index >= b || 
        vec.size() != static_cast<size_t>(d)) {
        return '0';  // Retorno seguro sin logging excesivo
    }
    
    // Calcular producto punto a^T * x
    double projection = std::inner_product(
        vec.begin(), 
        vec.end(), 
        random_vectors[hash_function_index].begin(), 
        0.0
    );
    
    // h(x) = sign(a^T * x). Mapeamos a '1' si es >= 0, y a '0' si es < 0.
    return (projection >= 0.0) ? '1' : '0';
}

void SRPHasher::print_hash_info() const {
    std::cout << "SRPHasher Información:" << std::endl;
    std::cout << "  - Dimensiones: " << d << std::endl;
    std::cout << "  - Número de funciones hash: " << b << std::endl;
    std::cout << "  - Inicializado: " << (initialized ? "Sí" : "No") << std::endl;
    
    if (initialized && !random_vectors.empty()) {
        std::cout << "  - Vectores aleatorios generados: " << random_vectors.size() << std::endl;
        
        // Mostrar estadísticas del primer vector aleatorio como ejemplo
        if (!random_vectors[0].empty()) {
            double sum = 0.0;
            for (double val : random_vectors[0]) {
                sum += val;
            }
            double mean = sum / random_vectors[0].size();
            
            double variance_sum = 0.0;
            for (double val : random_vectors[0]) {
                variance_sum += (val - mean) * (val - mean);
            }
            double variance = variance_sum / random_vectors[0].size();
            double std_dev = std::sqrt(variance);
            
            std::cout << "  - Ejemplo (vector 0): media=" << std::fixed << std::setprecision(4) 
                      << mean << ", std=" << std_dev << std::endl;
        }
    }
}

bool SRPHasher::is_initialized() const {
    return initialized && random_vectors.size() == static_cast<size_t>(b);
}