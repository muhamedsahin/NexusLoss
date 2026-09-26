/**
 * @file point_cloud.hpp
 * @brief Iki nokta bulutu arasindaki Chamfer ve Earth Mover mesafeleri.
 *
 * Bulutlar duz koordinat dizisidir, varsayilan boyut 3'tur. EMD esit sayida
 * nokta ister ve Hungarian eslestirmesi O(n^3) oldugu icin kucuk bulutlara
 * yoneliktir. Gradyan en yakin (Chamfer) veya eslesen (EMD) komsuya akar.
 */
#ifndef NEXUSLOSS_POINT_CLOUD_HPP
#define NEXUSLOSS_POINT_CLOUD_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace nexusloss::point_cloud {

template<typename T>
void validate(std::span<const T> a, std::span<const T> b, size_t dimensions) {
    if (!dimensions || a.empty() || b.empty() || a.size() % dimensions || b.size() % dimensions)
        throw std::invalid_argument("NexusLoss: point clouds must be flattened coordinate arrays");
}

template<typename T>
T point_distance(std::span<const T> a, size_t ai, std::span<const T> b, size_t bi, size_t dimensions) {
    T sum = T(0);
    for (size_t d = 0; d < dimensions; ++d) {
        const T delta = a[ai * dimensions + d] - b[bi * dimensions + d];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}

template<typename T>
T chamfer_distance(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    validate(a, b, dimensions);
    const size_t na = a.size() / dimensions, nb = b.size() / dimensions;
    T sum = T(0);
    for (size_t i = 0; i < na; ++i) {
        T best = std::numeric_limits<T>::max();
        for (size_t j = 0; j < nb; ++j) best = std::min(best, point_distance(a, i, b, j, dimensions));
        sum += best;
    }
    for (size_t j = 0; j < nb; ++j) {
        T best = std::numeric_limits<T>::max();
        for (size_t i = 0; i < na; ++i) best = std::min(best, point_distance(a, i, b, j, dimensions));
        sum += best;
    }
    return sum / static_cast<T>(na + nb);
}

template<typename T>
T earth_movers_distance(std::span<const T> a, std::span<const T> b, size_t dimensions = 3,
                        std::vector<size_t>* assignment = nullptr) {
    validate(a, b, dimensions);
    const size_t n = a.size() / dimensions;
    if (b.size() / dimensions != n)
        throw std::invalid_argument("NexusLoss: exact EMD requires point clouds with equal point counts");
    // Hungarian assignment gives the exact minimum-cost matching in O(n^3).
    std::vector<T> u(n + 1), v(n + 1);
    std::vector<size_t> matching(n + 1), path(n + 1);
    for (size_t i = 1; i <= n; ++i) {
        matching[0] = i;
        size_t j0 = 0;
        std::vector<T> minimum(n + 1, std::numeric_limits<T>::max());
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            const size_t i0 = matching[j0];
            T delta = std::numeric_limits<T>::max();
            size_t j1 = 0;
            for (size_t j = 1; j <= n; ++j) if (!used[j]) {
                const T cost = point_distance(a, i0 - 1, b, j - 1, dimensions) - u[i0] - v[j];
                if (cost < minimum[j]) { minimum[j] = cost; path[j] = j0; }
                if (minimum[j] < delta) { delta = minimum[j]; j1 = j; }
            }
            for (size_t j = 0; j <= n; ++j) {
                if (used[j]) { u[matching[j]] += delta; v[j] -= delta; }
                else minimum[j] -= delta;
            }
            j0 = j1;
        } while (matching[j0] != 0);
        do {
            const size_t j1 = path[j0];
            matching[j0] = matching[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    if (assignment) {
        assignment->assign(n, 0);
        for (size_t column = 1; column <= n; ++column) (*assignment)[matching[column] - 1] = column - 1;
    }
    return -v[0] / static_cast<T>(n);
}

/// @brief Chamfer kaybinin koordinat turevi. Cikis [grad_a | grad_b], girdiyle ayni duzen.
template<typename T>
std::vector<T> chamfer_distance_gradient(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    validate(a, b, dimensions);
    const size_t na = a.size() / dimensions, nb = b.size() / dimensions;
    std::vector<T> grad(a.size() + b.size(), T(0));
    const T scale = T(1) / static_cast<T>(na + nb);
    auto accumulate = [&](size_t source_cloud, size_t source, size_t neighbor) {
        const std::span<const T> from = source_cloud == 0 ? a : b;
        const std::span<const T> to = source_cloud == 0 ? b : a;
        T distance = point_distance(from, source, to, neighbor, dimensions);
        if (distance <= T(0)) return;
        distance = std::max(distance, std::numeric_limits<T>::min());
        for (size_t d = 0; d < dimensions; ++d) {
            const T delta = (from[source * dimensions + d] - to[neighbor * dimensions + d]) / distance * scale;
            if (source_cloud == 0) {
                grad[source * dimensions + d] += delta;
                grad[a.size() + neighbor * dimensions + d] -= delta;
            } else {
                grad[a.size() + source * dimensions + d] += delta;
                grad[neighbor * dimensions + d] -= delta;
            }
        }
    };
    for (size_t i = 0; i < na; ++i) {
        size_t best = 0;
        T best_distance = std::numeric_limits<T>::max();
        for (size_t j = 0; j < nb; ++j) {
            const T distance = point_distance(a, i, b, j, dimensions);
            if (distance < best_distance) { best_distance = distance; best = j; }
        }
        accumulate(0, i, best);
    }
    for (size_t j = 0; j < nb; ++j) {
        size_t best = 0;
        T best_distance = std::numeric_limits<T>::max();
        for (size_t i = 0; i < na; ++i) {
            const T distance = point_distance(a, i, b, j, dimensions);
            if (distance < best_distance) { best_distance = distance; best = i; }
        }
        accumulate(1, j, best);
    }
    return grad;
}

template<typename T>
std::vector<T> earth_movers_gradient(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    std::vector<size_t> assignment;
    earth_movers_distance(a, b, dimensions, &assignment);
    const size_t n = assignment.size();
    std::vector<T> grad(a.size() + b.size(), T(0));
    const T scale = T(1) / static_cast<T>(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t j = assignment[i];
        const T distance = std::max(point_distance(a, i, b, j, dimensions), std::numeric_limits<T>::min());
        if (point_distance(a, i, b, j, dimensions) == T(0)) continue;
        for (size_t d = 0; d < dimensions; ++d) {
            const T delta = (a[i * dimensions + d] - b[j * dimensions + d]) / distance * scale;
            grad[i * dimensions + d] += delta;
            grad[a.size() + j * dimensions + d] -= delta;
        }
    }
    return grad;
}

template<typename T>
class ChamferDistanceLoss {
public:
    explicit ChamferDistanceLoss(size_t dimensions = 3) : dimensions_(dimensions) {}
    T forward(std::span<const T> a, std::span<const T> b) {
        a_.assign(a.begin(), a.end());
        b_.assign(b.begin(), b.end());
        ready_ = true;
        return chamfer_distance(a, b, dimensions_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return chamfer_distance_gradient<T>(a_, b_, dimensions_);
    }
private:
    size_t dimensions_;
    std::vector<T> a_, b_;
    bool ready_ = false;
};

template<typename T>
class EarthMoverDistanceLoss {
public:
    explicit EarthMoverDistanceLoss(size_t dimensions = 3) : dimensions_(dimensions) {}
    /// @note Exact EMD, Hungarian eslestirme, O(n^3). Buyuk bulutlarda yaklasik EMD tercih edilir.
    T forward(std::span<const T> a, std::span<const T> b) {
        a_.assign(a.begin(), a.end());
        b_.assign(b.begin(), b.end());
        ready_ = true;
        return earth_movers_distance(a, b, dimensions_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return earth_movers_gradient<T>(a_, b_, dimensions_);
    }
private:
    size_t dimensions_;
    std::vector<T> a_, b_;
    bool ready_ = false;
};

} // namespace nexusloss::point_cloud

#endif
