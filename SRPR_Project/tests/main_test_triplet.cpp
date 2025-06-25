#include "../include/Triplet.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Probando Carga de Tripletas ===" << std::endl;

    // === PRUEBA 1: Archivo CSV básico ===
    std::cout << "\n--- Prueba 1: Archivo CSV básico ---" << std::endl;
    
    // Crear un archivo CSV de prueba
    std::ofstream test_file("triplets_test.csv");
    test_file << "101,1,2\n";
    test_file << "101,1,3\n";
    test_file << "102,5,6\n";
    test_file.close();

    std::vector<Triplet> triplets = load_triplets("triplets_test.csv");

    if (triplets.empty()) {
        std::cerr << "Prueba 1 fallida: No se cargaron las tripletas." << std::endl;
        return 1;
    }

    std::cout << "✓ Se cargaron " << triplets.size() << " tripletas del CSV básico." << std::endl;
    for (const auto& t : triplets) {
        std::cout << "  Usuario: " << t.user_id
                  << ", Prefiere: " << t.preferred_item_id
                  << " sobre: " << t.less_preferred_item_id << std::endl;
    }

    // === PRUEBA 2: Muestra de MovieLens ===
    std::cout << "\n--- Prueba 2: Muestra de MovieLens ---" << std::endl;
    
    std::vector<Triplet> movielens_triplets = load_triplets("data/movielens_sample.csv");
    
    if (movielens_triplets.empty()) {
        std::cout << "⚠️ No se encontró la muestra de MovieLens (data/movielens_sample.csv)" << std::endl;
        std::cout << "   Esto es normal si no has ejecutado el test de MovieLens antes." << std::endl;
    } else {
        std::cout << "✓ Se cargaron " << movielens_triplets.size() << " tripletas de MovieLens." << std::endl;
        std::cout << "  Primeras 3 tripletas de MovieLens:" << std::endl;
        for (int i = 0; i < std::min(3, (int)movielens_triplets.size()); ++i) {
            const auto& t = movielens_triplets[i];
            std::cout << "    Usuario: " << t.user_id
                      << ", Prefiere película: " << t.preferred_item_id
                      << " sobre película: " << t.less_preferred_item_id << std::endl;
        }
    }

    // === PRUEBA 3: Carga directa desde MovieLens (muestra pequeña) ===
    std::cout << "\n--- Prueba 3: Carga directa desde MovieLens (muestra pequeña) ---" << std::endl;
    
    std::vector<Triplet> direct_movielens = load_movielens_triplets(
        "data/movielens/ml-20m/ratings.csv", 
        1000,  // Solo 1000 ratings para prueba rápida
        5      // Solo 5 tripletas por usuario
    );
    
    if (direct_movielens.empty()) {
        std::cout << "⚠️ No se pudo cargar directamente desde MovieLens." << std::endl;
        std::cout << "   Verifica que existe: data/movielens/ml-20m/ratings.csv" << std::endl;
    } else {
        std::cout << "✓ Carga directa exitosa: " << direct_movielens.size() << " tripletas generadas." << std::endl;
    }

    std::cout << "\n🎉 Todas las pruebas de Tripletas completadas!" << std::endl;
    return 0;
}