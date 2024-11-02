#include <RTNeural/RTNeural.h>
#include <iostream>

struct TestMathsProvider
{
    template <typename Matrix>
    static auto tanh (const Matrix& x)
    {
        return x.array().tanh();
    }

    template <typename Matrix>
    static auto sigmoid (const Matrix& x)
    {
        using T = typename Matrix::Scalar;
        return (T) 1 / (((T) -1 * x.array()).array().exp() + (T) 1);
    }

    template <typename Matrix>
    static auto exp (const Matrix& x)
    {
        return x.array().exp();
    }
};

int main()
{
    const auto model_path { std::string { ROOT_DIR } + "AMP Orange Nasty.json" };

    std::cout << "Loading model from path: " << model_path << std::endl;

    std::ifstream jsonStream (model_path, std::ifstream::binary);

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    static constexpr size_t N = 2048;
    std::vector<float> input;
    input.resize (N, 0.0);
    std::vector<float> output_default;
    output_default.resize (N, 0.0);
    std::vector<float> output_custom;
    output_custom.resize (N, 0.0);

    for (size_t n = 0; n < input.size(); ++n)
        input[n] = std::sin (3.14 * static_cast<double> (n) * 0.01);

    RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 16, RTNeural::SampleRateCorrectionMode::None, RTNeural::DefaultMathsProvider>, RTNeural::DenseT<float, 16, 1>> model_default;
    model_default.parseJson (modelJson, true);
    model_default.reset();

    RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 16, RTNeural::SampleRateCorrectionMode::None, TestMathsProvider>, RTNeural::DenseT<float, 16, 1>> model_custom;
    model_custom.parseJson (modelJson, true);
    model_default.reset();

    float* data = input.data();

    float err = 0;

    for (size_t n = 0; n < input.size(); ++n)
    {
        // nam_dsp->process (input.data() + n, output_nam.data() + n, 1);
        output_default[n] = model_default.forward(data + n);
        output_custom[n] = model_custom.forward (data + n);

        float delta = output_default[n] - output_custom[n];

        err += (delta * delta);
    }

    std::cout << "Err is: " << sqrt(err);

    return 0;
}
