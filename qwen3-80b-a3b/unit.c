#include <math.h>
#include <stddef.h>

void conv1d_simple(__bf16 *output, const __bf16 *input, int length, const __bf16 *weight, int kernel_size) {

    int output_length = length + kernel_size - 1;

    // Perform convolution - each filter processes its corresponding input channel
    for (int i = 0; i < length; ++i) {
        for (int k = 0; k < kernel_size; ++k) {
            if (i - k >= 0) {
                output[i] += input[i - k] * weight[kernel_size - 1 - k];
            }
        }
    }
}

void silu_array(__bf16 *output, const __bf16 *input, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] / (1.0f + expf(-input[i]));
    }
}

double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

int main() {
    __bf16 tensor[] = {-0.9297, -3.1250, -5.4688, -2.2344, -4.3125, 2.3125, -5.4688, -0.9297, -4.7812, -5.4688};
    __bf16 weight[] = {-0.0090, -0.0270, -0.0615, 0.0613};
    __bf16 output[100] = {0};
    __bf16 output2[100] = {0};
    conv1d_simple(output, tensor, 10, weight, 4);
    silu_array(output2, output, 10);

    /* output */
    // tensor([-0.0569, -0.1348, -0.1177,  0.2930,  0.0488,  0.5156, -0.3418,  0.2559,
    //     -0.1089,  0.0332,  0.4746,  0.1904,  0.0491], dtype=torch.bfloat16)

    __bf16 sigmoid_input[] = {0.9961,  1.8750,  3.2188, 6.1562, 1.1094,  1.1797,  1.0625,  1.9531,
                              1.1562,  0.2461,  1.8047, 1.9688, 3.3906,  1.8359,  2.2812,  2.3906,
                              0.5234,  1.9766,  1.6719, 3.8750, 1.7188,  -0.0361, -1.7422, -0.5469,
                              -0.2314, -0.2715, 0.1025, 0.9375, -0.4824, -0.5234, 5.5312,  5.3125};
    __bf16 sigmoid_output[32] = {};

    for (int i = 0; i < 32; i++) {
        sigmoid_output[i] = sigmoid(sigmoid_input[i]);
    }

    // tensor([0.7305, 0.8672, 0.9609, 0.9961, 0.7539, 0.7656, 0.7422, 0.8750, 0.7617,
    //     0.5625, 0.8594, 0.8789, 0.9688, 0.8633, 0.9062, 0.9180, 0.6289, 0.8789,
    //     0.8438, 0.9805, 0.8477, 0.4902, 0.1494, 0.3672, 0.4434, 0.4316, 0.5273,
    //     0.7188, 0.3809, 0.3711, 0.9961, 0.9961], dtype=torch.bfloat16)

    return 0;
}