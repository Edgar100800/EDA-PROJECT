#ifndef LSH_H
#define LSH_H

#include <vector>
#include <string>
#include <random>
#include <numeric> // Para std::inner_product

using Vector = std::vector<double>;

// Clase base abstracta para familias LSH
class LSH {
public:
    LSH(int dimensions, int num_hashes) : d(dimensions), b(num_hashes) {}
    virtual ~LSH() = default;

    // Genera un código binario de longitud b para un vector dado.
    std::string generate_code(const Vector& vec) const;

    // Getters para información de configuración
    int get_dimensions() const { return d; }
    int get_num_hashes() const { return b; }

protected:
    // Genera un único bit de hash (0 o 1). Debe ser implementado por las clases hijas.
    virtual char hash_to_bit(const Vector& vec, int hash_function_index) const = 0;

    int d; // Dimensiones del vector
    int b; // Número de funciones de hash (longitud del código)
};

// Implementación de Sign Random Projection (SRP)
class SRPHasher : public LSH {
public:
    SRPHasher(int dimensions, int num_hashes, unsigned int seed = 0);

    // Función para obtener información de debug
    void print_hash_info() const;
    
    // Función para verificar la configuración
    bool is_initialized() const;

protected:
    char hash_to_bit(const Vector& vec, int hash_function_index) const override;

private:
    std::vector<Vector> random_vectors; // Parámetros 'a' de las funciones de hash
    bool initialized;
};

#endif // LSH_H