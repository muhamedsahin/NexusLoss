/**
 * @file point_cloud.hpp
 * @brief Iki nokta bulutu arasindaki Chamfer ve Earth Mover mesafeleri.
 *
 * Bulutlar duz koordinat dizisidir, varsayilan boyut 3'tur. EMD esit sayida
 * nokta ister ve Hungarian eslestirmesi O(n^3) oldugu icin kucuk bulutlara
 * yoneliktir. Gradyan en yakin (Chamfer) veya eslesen (EMD) komsuya akar.
 *
 * Gradyan duzeni her yerde [grad_a | grad_b] seklindedir; uzunluk
 * a.size() + b.size(). `*_gradient_into` cagiranin tamponuna yazar. EMD'nin
 * Hungarian tablolari `HungarianWorkspace` icinde tutulur; ayni boyutta
 * tekrar cagrida yeni bellek ayrilmaz.
 */
#ifndef NEXUSLOSS_POINT_CLOUD_HPP
#define NEXUSLOSS_POINT_CLOUD_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
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
T squared_point_distance(std::span<const T> a, size_t ai, std::span<const T> b, size_t bi, size_t dimensions) {
    T sum = T(0);
    for (size_t d = 0; d < dimensions; ++d) {
        const T delta = a[ai * dimensions + d] - b[bi * dimensions + d];
        sum += delta * delta;
    }
    return sum;
}

template<typename T>
T point_distance(std::span<const T> a, size_t ai, std::span<const T> b, size_t bi, size_t dimensions) {
    return std::sqrt(squared_point_distance(a, ai, b, bi, dimensions));
}

/// En yakin komsu karsilastirmasi karekok almadan yapilir; sira degismez.
template<typename T>
size_t nearest_neighbor(std::span<const T> from, size_t index, std::span<const T> to, size_t dimensions) {
    const size_t count = to.size() / dimensions;
    size_t best = 0;
    T best_distance = std::numeric_limits<T>::max();
    for (size_t j = 0; j < count; ++j) {
        const T distance = squared_point_distance(from, index, to, j, dimensions);
        if (distance < best_distance) { best_distance = distance; best = j; }
    }
    return best;
}

template<typename T>
T chamfer_distance(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    validate(a, b, dimensions);
    const size_t na = a.size() / dimensions, nb = b.size() / dimensions;
    T sum = T(0);
    for (size_t i = 0; i < na; ++i) sum += point_distance(a, i, b, nearest_neighbor(a, i, b, dimensions), dimensions);
    for (size_t j = 0; j < nb; ++j) sum += point_distance(b, j, a, nearest_neighbor(b, j, a, dimensions), dimensions);
    return sum / static_cast<T>(na + nb);
}

/**
 * @brief Hungarian eslestirmesinin tekrar kullanilan tablolari.
 *
 * `prepare(n)` kapasiteyi bir kez buyutur. Sonraki ayni veya daha kucuk
 * boyuttaki cagrilar `assign` ile yalnizca deger yazar, bellek ayirmaz.
 */
template<typename T>
struct HungarianWorkspace {
    std::vector<T> u, v, minimum;
    std::vector<size_t> matching, path, assignment;
    std::vector<unsigned char> used;

    void prepare(size_t n) {
        u.assign(n + 1, T(0));
        v.assign(n + 1, T(0));
        minimum.resize(n + 1);
        matching.assign(n + 1, 0);
        path.assign(n + 1, 0);
        used.resize(n + 1);
        assignment.resize(n);
    }
};

/**
 * @brief Esit sayili iki bulut arasinda tam EMD. `workspace.assignment[i]`,
 *        a'nin i. noktasinin eslestigi b noktasidir.
 */
template<typename T>
T earth_movers_distance(std::span<const T> a, std::span<const T> b, size_t dimensions,
                        HungarianWorkspace<T>& workspace) {
    validate(a, b, dimensions);
    const size_t n = a.size() / dimensions;
    if (b.size() / dimensions != n)
        throw std::invalid_argument("NexusLoss: exact EMD requires point clouds with equal point counts");
    workspace.prepare(n);
    auto& u = workspace.u;
    auto& v = workspace.v;
    auto& minimum = workspace.minimum;
    auto& matching = workspace.matching;
    auto& path = workspace.path;
    auto& used = workspace.used;
    for (size_t i = 1; i <= n; ++i) {
        matching[0] = i;
        size_t j0 = 0;
        std::fill(minimum.begin(), minimum.end(), std::numeric_limits<T>::max());
        std::fill(used.begin(), used.end(), static_cast<unsigned char>(0));
        do {
            used[j0] = 1;
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
    for (size_t column = 1; column <= n; ++column) workspace.assignment[matching[column] - 1] = column - 1;
    return -v[0] / static_cast<T>(n);
}

template<typename T>
T earth_movers_distance(std::span<const T> a, std::span<const T> b, size_t dimensions = 3,
                        std::vector<size_t>* assignment = nullptr) {
    HungarianWorkspace<T> workspace;
    const T distance = earth_movers_distance(a, b, dimensions, workspace);
    if (assignment) *assignment = workspace.assignment;
    return distance;
}

namespace detail {

/// d||x - y|| / dx = (x - y) / ||x - y||. Cakisan noktada subgradyan 0 secilir.
template<typename T>
void accumulate_distance_gradient(std::span<const T> from, size_t source, std::span<const T> to, size_t target,
                                  size_t dimensions, T scale, std::span<T> grad_from, std::span<T> grad_to) {
    const T distance = point_distance(from, source, to, target, dimensions);
    if (!(distance > T(0))) return;
    const T factor = scale / distance;
    for (size_t d = 0; d < dimensions; ++d) {
        const T delta = (from[source * dimensions + d] - to[target * dimensions + d]) * factor;
        grad_from[source * dimensions + d] += delta;
        grad_to[target * dimensions + d] -= delta;
    }
}

template<typename T>
void require_cloud_gradient(std::span<const T> a, std::span<const T> b, std::span<T> grad_out) {
    if (grad_out.size() != a.size() + b.size())
        throw std::invalid_argument("NexusLoss: point-cloud grad_out must hold [grad_a | grad_b]");
}

} // namespace detail

/// @brief Chamfer kaybinin koordinat turevi. Cikis [grad_a | grad_b], ayirma yok.
template<typename T>
void chamfer_distance_gradient_into(std::span<const T> a, std::span<const T> b, std::span<T> grad_out,
                                    size_t dimensions = 3) {
    validate(a, b, dimensions);
    detail::require_cloud_gradient(a, b, grad_out);
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    const size_t na = a.size() / dimensions, nb = b.size() / dimensions;
    const T scale = T(1) / static_cast<T>(na + nb);
    const std::span<T> grad_a = grad_out.first(a.size());
    const std::span<T> grad_b = grad_out.subspan(a.size());
    for (size_t i = 0; i < na; ++i)
        detail::accumulate_distance_gradient(a, i, b, nearest_neighbor(a, i, b, dimensions), dimensions, scale,
                                             grad_a, grad_b);
    for (size_t j = 0; j < nb; ++j)
        detail::accumulate_distance_gradient(b, j, a, nearest_neighbor(b, j, a, dimensions), dimensions, scale,
                                             grad_b, grad_a);
}

template<typename T>
std::vector<T> chamfer_distance_gradient(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    std::vector<T> grad(a.size() + b.size());
    chamfer_distance_gradient_into<T>(a, b, grad, dimensions);
    return grad;
}

/// @brief Eslesme sabitken EMD turevi. Hungarian sonucu `workspace` icinden okunur.
template<typename T>
void earth_movers_gradient_into(std::span<const T> a, std::span<const T> b, std::span<T> grad_out,
                                HungarianWorkspace<T>& workspace, size_t dimensions = 3) {
    detail::require_cloud_gradient(a, b, grad_out);
    earth_movers_distance(a, b, dimensions, workspace);
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    const size_t n = workspace.assignment.size();
    const T scale = T(1) / static_cast<T>(n);
    const std::span<T> grad_a = grad_out.first(a.size());
    const std::span<T> grad_b = grad_out.subspan(a.size());
    for (size_t i = 0; i < n; ++i)
        detail::accumulate_distance_gradient(a, i, b, workspace.assignment[i], dimensions, scale, grad_a, grad_b);
}

template<typename T>
std::vector<T> earth_movers_gradient(std::span<const T> a, std::span<const T> b, size_t dimensions = 3) {
    std::vector<T> grad(a.size() + b.size());
    HungarianWorkspace<T> workspace;
    earth_movers_gradient_into<T>(a, b, grad, workspace, dimensions);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return a_.size() + b_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        chamfer_distance_gradient_into<T>(a_, b_, grad_out, dimensions_);
    }
private:
    size_t dimensions_;
    std::vector<T> a_, b_;
    bool ready_ = false;
};

/**
 * @class EarthMoverDistanceLoss
 * @note Exact EMD, Hungarian eslestirme, O(n^3). forward() eslesmeyi saklar;
 *       backward_into ayni eslesmeyi kullanir, Hungarian'i tekrar cozmez.
 *       Ayni nesneye iki is parcacigindan es zamanli forward cagrilmaz.
 */
template<typename T>
class EarthMoverDistanceLoss {
public:
    explicit EarthMoverDistanceLoss(size_t dimensions = 3) : dimensions_(dimensions) {}
    T forward(std::span<const T> a, std::span<const T> b) {
        a_.assign(a.begin(), a.end());
        b_.assign(b.begin(), b.end());
        const T distance = earth_movers_distance<T>(a, b, dimensions_, workspace_);
        ready_ = true;
        return distance;
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return a_.size() + b_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        const std::span<const T> a(a_), b(b_);
        detail::require_cloud_gradient(a, b, grad_out);
        std::fill(grad_out.begin(), grad_out.end(), T(0));
        const size_t n = workspace_.assignment.size();
        const T scale = T(1) / static_cast<T>(n);
        const std::span<T> grad_a = grad_out.first(a.size());
        const std::span<T> grad_b = grad_out.subspan(a.size());
        for (size_t i = 0; i < n; ++i)
            detail::accumulate_distance_gradient(a, i, b, workspace_.assignment[i], dimensions_, scale, grad_a, grad_b);
    }
private:
    size_t dimensions_;
    std::vector<T> a_, b_;
    HungarianWorkspace<T> workspace_;
    bool ready_ = false;
};

} // namespace nexusloss::point_cloud

#endif
