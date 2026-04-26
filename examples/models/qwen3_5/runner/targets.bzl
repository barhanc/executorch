load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    runtime.cxx_binary(
        name = "qwen3_5_runner",
        srcs = [
            "main.cpp",
        ],
        compiler_flags = ["-Wno-global-constructors"],
        deps = [
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
            "//executorch/extension/llm/runner:runner_lib",
        ],
        external_deps = [
            "gflags",
            "stb",
        ],
        **get_oss_build_kwargs()
    )
