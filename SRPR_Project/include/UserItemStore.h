#ifndef USER_ITEM_STORE_H
#define USER_ITEM_STORE_H

#include <vector>
#include <unordered_map>
#include <random>
#include "Triplet.h"

using Vector = std::vector<double>;

class UserItemStore {
public:
    UserItemStore(int dimensions);

    // Inicializa los vectores para todos los usuarios e ítems encontrados en las tripletas.
    void initialize(const std::vector<Triplet>& triplets);

    // Obtiene una referencia modificable a un vector.
    Vector& get_user_vector(int user_id);
    Vector& get_item_vector(int item_id);

    // Obtiene una referencia constante.
    const Vector& get_user_vector(int user_id) const;
    const Vector& get_item_vector(int item_id) const;

    const std::unordered_map<int, Vector>& get_all_item_vectors() const;

    void print_summary() const;

private:
    int d; // Dimensionalidad de los vectores latentes
    std::unordered_map<int, Vector> user_vectors; // Matriz X
    std::unordered_map<int, Vector> item_vectors; // Matriz Y

    // Generador de números aleatorios para la inicialización.
    std::mt19937 rng;
    std::normal_distribution<double> dist;
};

#endif // USER_ITEM_STORE_H