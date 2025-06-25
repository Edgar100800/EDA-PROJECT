#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>

int main() {
    std::cout << "--- Probando Carga de Datos Reales ---" << std::endl;

    // Cargar el archivo de datos real
    std::vector<Triplet> triplets = load_triplets("data/triplets.csv");

    if (triplets.empty()) {
        std::cerr << "Prueba fallida: No se cargaron las tripletas del archivo real." << std::endl;
        return 1;
    }

    std::cout << "Se cargaron " << triplets.size() << " tripletas del archivo real." << std::endl;
    
    // Mostrar las primeras 5 tripletas
    std::cout << "\nPrimeras 5 tripletas:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), triplets.size()); ++i) {
        const auto& t = triplets[i];
        std::cout << "  " << (i+1) << ". Usuario: " << t.user_id
                  << ", Prefiere: " << t.preferred_item_id
                  << " sobre: " << t.less_preferred_item_id << std::endl;
    }

    // Estadísticas básicas
    std::set<int> unique_users, unique_items;
    for (const auto& t : triplets) {
        unique_users.insert(t.user_id);
        unique_items.insert(t.preferred_item_id);
        unique_items.insert(t.less_preferred_item_id);
    }

    std::cout << "\nEstadísticas:" << std::endl;
    std::cout << "  - Total de tripletas: " << triplets.size() << std::endl;
    std::cout << "  - Usuarios únicos: " << unique_users.size() << std::endl;
    std::cout << "  - Items únicos: " << unique_items.size() << std::endl;

    std::cout << "\nPrueba de datos reales completada con éxito." << std::endl;
    return 0;
}