#include "../include/UserItemStore.h"
#include <iostream>
#include <set>

UserItemStore::UserItemStore(int dimensions) : d(dimensions), rng(std::random_device{}()), dist(0.0, 0.1) {}

void UserItemStore::initialize(const std::vector<Triplet>& triplets) {
    std::set<int> user_ids;
    std::set<int> item_ids;

    for (const auto& t : triplets) {
        user_ids.insert(t.user_id);
        item_ids.insert(t.preferred_item_id);
        item_ids.insert(t.less_preferred_item_id);
    }

    for (int id : user_ids) {
        user_vectors[id] = Vector(d);
        for (int i = 0; i < d; ++i) {
            user_vectors[id][i] = dist(rng);
        }
    }

    for (int id : item_ids) {
        item_vectors[id] = Vector(d);
        for (int i = 0; i < d; ++i) {
            item_vectors[id][i] = dist(rng);
        }
    }
}

Vector& UserItemStore::get_user_vector(int user_id) {
    return user_vectors.at(user_id);
}

Vector& UserItemStore::get_item_vector(int item_id) {
    return item_vectors.at(item_id);
}

const Vector& UserItemStore::get_user_vector(int user_id) const {
    return user_vectors.at(user_id);
}

const Vector& UserItemStore::get_item_vector(int item_id) const {
    return item_vectors.at(item_id);
}

const std::unordered_map<int, Vector>& UserItemStore::get_all_item_vectors() const {
    return item_vectors;
}

void UserItemStore::print_summary() const {
    std::cout << "UserItemStore Resumen:" << std::endl;
    std::cout << "  - " << user_vectors.size() << " usuarios." << std::endl;
    std::cout << "  - " << item_vectors.size() << " items." << std::endl;
    std::cout << "  - Dimensiones: " << d << std::endl;
}