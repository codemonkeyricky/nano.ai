

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

int main() {
    __bf16 tensor[] = {-0.9297, -3.1250, -5.4688, -2.2344, -4.3125, 2.3125, -5.4688, -0.9297, -4.7812, -5.4688};
    __bf16 weight[] = {-0.0090, -0.0270, -0.0615, 0.0613};
    __bf16 output[100] = {0};
    conv1d_simple(output, tensor, 10, weight, 4);

    /* output */
    // tensor([-0.0569, -0.1348, -0.1177,  0.2930,  0.0488,  0.5156, -0.3418,  0.2559,
    //     -0.1089,  0.0332,  0.4746,  0.1904,  0.0491], dtype=torch.bfloat16)
    return 0;
}