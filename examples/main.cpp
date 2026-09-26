#include <iostream>
#include <vector>

#include <nexusloss/nexusloss.hpp>

namespace {

void print_gradient(const char* name, double loss, const std::vector<double>& gradient) {
    std::cout << name << " loss: " << loss << "  grad:";
    for (double value : gradient) std::cout << ' ' << value;
    std::cout << '\n';
}

}

int main() {
    const std::vector<double> prediction{2.0, 4.0, 3.0};
    const std::vector<double> target{1.0, 5.0, 2.5};
    nexusloss::HuberLoss<double> huber(1.0);
    const double huber_loss = huber.forward(prediction, target).front();
    print_gradient("Huber", huber_loss, huber.backward());

    const std::vector<double> logits{0.0, 2.0};
    const std::vector<double> binary_target{1.0, 0.0};
    nexusloss::BCEWithLogitsLoss<double> bce;
    const double bce_loss = bce.forward(logits, binary_target).front();
    print_gradient("BCE-with-logits", bce_loss, bce.backward());

    const std::vector<double> class_logits{1.0, 2.0, 3.0};
    nexusloss::classification::CrossEntropyLoss<double> cross_entropy;
    const double cross_entropy_loss = cross_entropy.forward(class_logits, 2);
    print_gradient("Cross-entropy", cross_entropy_loss, cross_entropy.backward());
    return 0;
}
