/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <gflags/gflags.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "qwen3_5_v.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer stuff.");

DEFINE_string(prompt, "Describe this image.", "Prompt.");

DEFINE_string(image_path, "", "The path to a .jpg file.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    1024,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_int32(
    target_size,
    512,
    "Target image size for resizing. Defaults to 512 for Qwen3.5 vision encoder.");

using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;

/**
 * @brief Loads an image from a file and resizes it to target_size x target_size
 */
void load_image_qwen(const std::string& image_path, Image& image, int target_size) {
  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    ET_LOG(Fatal, "Failed to load image: %s", image_path.c_str());
    exit(1);
  }

  std::vector<uint8_t> resized_data(target_size * target_size * channels);
  stbir_resize_uint8(data, width, height, 0, resized_data.data(), target_size, target_size, 0, channels);

  std::vector<float> chw_data(channels * target_size * target_size);
  for (int h = 0; h < target_size; ++h) {
    for (int w = 0; w < target_size; ++w) {
      for (int c = 0; c < channels; ++c) {
        uint8_t pixel_value = resized_data[h * target_size * channels + w * channels + c];
        chw_data[c * target_size * target_size + h * target_size + w] = static_cast<float>(pixel_value) / 255.0f;
      }
    }
  }

  image = Image(std::move(chw_data), target_size, target_size, channels);
  stbi_image_free(data);
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer = ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  auto runner = ::executorch::extension::llm::create_multimodal_runner(FLAGS_model_path.c_str(), std::move(tokenizer));
  if (runner->load() != ::executorch::runtime::Error::Ok) return 1;

  // Qwen3.5 Static Model Support:
  // The model expects 1x1 token inputs. MultimodalRunner's prefiller defaults to parallel prefill
  // for text inputs (1xN). To bypass this without modifying core runner code, we tokenize 
  // manually and pass each token as a separate MultimodalInput.
  
  auto tokenize_to_inputs = [&](const std::string& text, std::vector<MultimodalInput>& inputs) {
    auto tokens_res = runner->get_tokenizer()->encode(text, 0, 0);
    if (tokens_res.ok()) {
      for (auto t : *tokens_res) {
        inputs.emplace_back(std::vector<uint64_t>{t});
      }
    }
  };

  std::vector<MultimodalInput> inputs;
  tokenize_to_inputs("<|im_start|>user\n", inputs);
  
  if (!FLAGS_image_path.empty()) {
    Image image;
    load_image_qwen(FLAGS_image_path, image, FLAGS_target_size);
    inputs.emplace_back(make_image_input(std::move(image)));
  }
  
  tokenize_to_inputs("\n" + FLAGS_prompt + "<|im_end|>\n<|im_start|>assistant\n", inputs);

  ::executorch::extension::llm::GenerationConfig config;
  config.temperature = FLAGS_temperature;
  config.seq_len = FLAGS_seq_len;

  if (runner->generate(inputs, config) != ::executorch::runtime::Error::Ok) {
    return 1;
  }

  printf("\n");
  return 0;
}
