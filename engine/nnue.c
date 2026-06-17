#include <stdint.h>
#include <string.h>
#include <math.h>

#define L1_HALF 512
#define L1      1024
#define L2      64
#define L3      32
#define CP_SCALE 200.0f

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__declspec(dllexport) void accumulator_init(const float* ft_b, float* w_acc, float* b_acc) {
    memcpy(w_acc, ft_b, L1_HALF * sizeof(float));
    memcpy(b_acc, ft_b, L1_HALF * sizeof(float));
}

__declspec(dllexport) void accumulator_add(const float* ft_W, int col, float* acc) {
    const float* col_ptr = ft_W + col * L1_HALF;
    for (int i = 0; i < L1_HALF; i++)
        acc[i] += col_ptr[i];
}

__declspec(dllexport) void accumulator_sub(const float* ft_W, int col, float* acc) {
    const float* col_ptr = ft_W + col * L1_HALF;
    for (int i = 0; i < L1_HALF; i++)
        acc[i] -= col_ptr[i];
}

__declspec(dllexport) void accumulator_add_sub(const float* ft_W, int add_col, int sub_col, float* acc) {
    const float* add_ptr = ft_W + add_col * L1_HALF;
    const float* sub_ptr = ft_W + sub_col * L1_HALF;
    for (int i = 0; i < L1_HALF; i++)
        acc[i] += add_ptr[i] - sub_ptr[i];
}

__declspec(dllexport) float forward(
    const float* w_acc,
    const float* b_acc,
    const float* l1_W,
    const float* l1_b,
    const float* l2_W,
    const float* l2_b,
    const float* out_W,
    float        out_b,
    int          white_to_move
) {
    float x[L1];
    float h1[L2];
    float h2[L3];

    const float* first  = white_to_move ? w_acc : b_acc;
    const float* second = white_to_move ? b_acc : w_acc;

    for (int i = 0; i < L1_HALF; i++)
        x[i]           = clampf(first[i],  0.0f, 1.0f);
    for (int i = 0; i < L1_HALF; i++)
        x[i + L1_HALF] = clampf(second[i], 0.0f, 1.0f);

    for (int i = 0; i < L2; i++) {
        float s = l1_b[i];
        const float* row = l1_W + i * L1;
        for (int j = 0; j < L1; j++)
            s += row[j] * x[j];
        h1[i] = clampf(s, 0.0f, 1.0f);
    }

    for (int i = 0; i < L3; i++) {
        float s = l2_b[i];
        const float* row = l2_W + i * L2;
        for (int j = 0; j < L2; j++)
            s += row[j] * h1[j];
        h2[i] = clampf(s, 0.0f, 1.0f);
    }

    float out = out_b;
    for (int i = 0; i < L3; i++)
        out += out_W[i] * h2[i];

    return out * CP_SCALE;
}
