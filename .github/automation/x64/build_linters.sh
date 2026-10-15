# Build oneDNN for CPU-only PR linter checks.

set -o errexit -o pipefail -o noclobber

export CC=clang
export CXX=clang++

if [[ "$ONEDNN_ACTION" == "configure" ]]; then
    if [[ "$GITHUB_JOB" == "pr-clang-tidy" ]]; then
        set -x
        cmake \
            -Bbuild -S. \
            -DCMAKE_BUILD_TYPE=Debug \
            -DONEDNN_BUILD_GRAPH=OFF \
            -DDNNL_EXPERIMENTAL=ON \
            -DDNNL_EXPERIMENTAL_PROFILING=ON \
            -DDNNL_EXPERIMENTAL_UKERNEL=ON \
            -DDNNL_CPU_RUNTIME=OMP \
            -DDNNL_GPU_RUNTIME=NONE \
            -DDNNL_WERROR=ON \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        set +x
    elif [[ "$GITHUB_JOB" == "pr-format-tags" ]]; then
        set -x
        cmake -B../build -S. -DONEDNN_BUILD_GRAPH=OFF -DDNNL_GPU_RUNTIME=NONE
        set +x
    else
        echo "Unknown linter job: $GITHUB_JOB"
        exit 1
    fi
elif [[ "$ONEDNN_ACTION" == "build" ]]; then
    set -x
    cmake --build build --parallel "$(getconf _NPROCESSORS_ONLN)"
    set +x
else
    echo "Unknown action: $ONEDNN_ACTION"
    exit 1
fi
