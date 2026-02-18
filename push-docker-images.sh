#!/usr/bin/env bash

# Docker Push Script for Text Embeddings Inference
# Builds and pushes Docker images for multiple CUDA compute capabilities
# Following GitHub Actions naming conventions

set -euo pipefail

# Configuration
VERSION="1.8.6.ner"
REGISTRIES=(
    # "registry.internal.huggingface.tech/api-inference/text-embeddings-inference"
    # "ghcr.io/huggingface/text-embeddings-inference"
    "baseten/bei_bert"
)

# Matrix configurations: prefix:compute_cap:dockerfile:grpc:sccache:extra_args
declare -A IMAGES=(
    ["turing-"]="75:Dockerfile-cuda:false:true:DEFAULT_USE_FLASH_ATTENTION=False"
    ["ampere-"]="80:Dockerfile-cuda:false:true:"
    ["86-"]="86:Dockerfile-cuda:false:true:"
    ["89-"]="89:Dockerfile-cuda:false:true:"
    ["hopper-"]="90:Dockerfile-cuda:false:true:"
    ["blackwell-"]="100:Dockerfile-cuda:false:true:"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are available
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if user is logged in to registries
    for registry in "${REGISTRIES[@]}"; do
        if [[ "$registry" == *"ghcr.io"* ]]; then
            if ! docker pull ghcr.io/huggingface/text-embeddings-inference:latest &> /dev/null; then
                log_warning "Not logged in to $registry. Please run 'docker login ghcr.io'"
            fi
        fi
    done
    
    log_success "Prerequisites check completed"
}

# Build and push image for a specific variant
build_and_push_variant() {
    local prefix="$1"
    local config="$2"
    
    IFS=':' read -r compute_cap dockerfile grpc sccache extra_args <<< "$config"
    
    log_info "Building variant: ${prefix}sm${compute_cap}"
    
    # Build arguments
    local build_args=(
        "--build-arg" "CUDA_COMPUTE_CAP=${compute_cap}"
        "--build-arg" "GIT_SHA=$(git rev-parse --short HEAD)"
        "--build-arg" "DOCKER_LABEL=sha-$(git rev-parse --short HEAD)"
        "--build-arg" "SCCACHE_GHA_ENABLED=false"
    )
    
    # Add extra build args if specified
    if [[ -n "$extra_args" ]]; then
        IFS=',' read -ra args <<< "$extra_args"
        for arg in "${args[@]}"; do
            build_args+=("--build-arg" "$arg")
        done
    fi
    
    # Tags to build
    local tags=()
    for registry in "${REGISTRIES[@]}"; do
        tags+=("-t" "${registry}:${prefix}${VERSION}")
        # tags+=("-t" "${registry}:${prefix}latest")
    done
    
    # Build standard image
    log_info "Building standard image for ${prefix}sm${compute_cap}..."
    docker buildx build \
        --platform linux/amd64 \
        --file "${dockerfile}" \
        "${build_args[@]}" \
        "${tags[@]}" \
        --push \
        .
    
    # Build gRPC image if enabled
    if [[ "$grpc" == "true" ]]; then
        log_info "Building gRPC image for ${prefix}sm${compute_cap}..."
        
        local grpc_tags=()
        for registry in "${REGISTRIES[@]}"; do
            grpc_tags+=("-t" "${registry}:${prefix}${VERSION}-grpc")
            grpc_tags+=("-t" "${registry}:${prefix}latest-grpc")
        done
        
        docker buildx build \
            --target grpc \
            --platform linux/amd64 \
            --file "${dockerfile}" \
            "${build_args[@]}" \
            "${grpc_tags[@]}" \
            --push \
            .
    fi
    
    log_success "Completed variant: ${prefix}sm${compute_cap}"
}

# Build all variants in parallel
build_all_variants() {
    log_info "Starting build for all variants..."
    
    # Create a temporary directory for job tracking
    local temp_dir=$(mktemp -d)
    local pids=()
    
    # Build each variant
    for prefix_config in "${!IMAGES[@]}"; do
        local actual_prefix="$prefix_config"
        [[ "$actual_prefix" == "ampere-" ]] && actual_prefix=""
        (
            build_and_push_variant "$actual_prefix" "${IMAGES[$prefix_config]}"
        ) &
        pids+=($!)
        
        # Limit parallel builds to avoid overwhelming the system
        if [[ ${#pids[@]} -ge 8 ]]; then
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
            pids=()
        fi
    done
    
    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    rm -rf "$temp_dir"
    log_success "All variants built successfully"
}

# Verify images were pushed
verify_images() {
    log_info "Verifying pushed images..."
    
    local failed=0
    for prefix_config in "${!IMAGES[@]}"; do
        IFS=':' read -r compute_cap dockerfile grpc sccache extra_args <<< "${IMAGES[$prefix_config]}"
        local actual_prefix="$prefix_config"
        [[ "$actual_prefix" == "ampere-" ]] && actual_prefix=""
        
        for registry in "${REGISTRIES[@]}"; do
            # Check standard image
            if ! docker pull "${registry}:${actual_prefix}${VERSION}" &> /dev/null; then
                log_error "Failed to verify ${registry}:${actual_prefix}${VERSION}"
                ((failed++))
            else
                log_success "Verified ${registry}:${actual_prefix}${VERSION}"
            fi
            
            # Check gRPC image if enabled
            if [[ "$grpc" == "true" ]]; then
                if ! docker pull "${registry}:${actual_prefix}${VERSION}-grpc" &> /dev/null; then
                    log_error "Failed to verify ${registry}:${actual_prefix}${VERSION}-grpc"
                    ((failed++))
                else
                    log_success "Verified ${registry}:${actual_prefix}${VERSION}-grpc"
                fi
            fi
        done
    done
    
    if [[ $failed -eq 0 ]]; then
        log_success "All images verified successfully"
    else
        log_error "$failed images failed verification"
        return 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verify   Only verify existing images, don't build"
    echo "  -p, --parallel Build in parallel (default: sequential)"
    echo "  --dry-run      Show what would be built without actually building"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_BUILDKIT  Set to 1 for improved build performance"
    echo "  BUILDKIT_INLINE_CACHE  Set to 1 for inline caching"
    echo ""
    echo "Supported Variants:"
    for prefix_config in "${!IMAGES[@]}"; do
        IFS=':' read -r compute_cap dockerfile grpc sccache extra_args <<< "${IMAGES[$prefix_config]}"
        local display_prefix="$prefix_config"
        [[ "$display_prefix" == "ampere-" ]] && display_prefix=""
        echo "  ${display_prefix}sm${compute_cap} (${dockerfile}, grpc: ${grpc})"
    done
}

# Main execution
main() {
    local verify_only=false
    local parallel_build=false
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verify)
                verify_only=true
                shift
                ;;
            -p|--parallel)
                parallel_build=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set environment variables for better performance
    export DOCKER_BUILDKIT=1
    export BUILDKIT_INLINE_CACHE=1
    
    log_info "Text Embeddings Inference Docker Build Script"
    log_info "Version: $VERSION"
    log_info "Git SHA: $(git rev-parse --short HEAD)"
    
    # Check prerequisites
    check_prerequisites
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN - Would build the following images:"
        for prefix_config in "${!IMAGES[@]}"; do
            IFS=':' read -r compute_cap dockerfile grpc sccache extra_args <<< "${IMAGES[$prefix_config]}"
            local display_prefix="$prefix_config"
            [[ "$display_prefix" == "ampere-" ]] && display_prefix=""
            echo "  ${display_prefix}sm${compute_cap} (${dockerfile}, grpc: ${grpc})"
        done
        exit 0
    fi
    
    if [[ "$verify_only" == "true" ]]; then
        verify_images
        exit $?
    fi
    
    # Build all variants
    if [[ "$parallel_build" == "true" ]]; then
        build_all_variants
    else
        for prefix_config in "${!IMAGES[@]}"; do
            build_and_push_variant "$prefix_config" "${IMAGES[$prefix_config]}"
        done
    fi
    
    # Verify images
    verify_images
    
    log_success "Docker build and push completed successfully!"
    log_info "Built images for version $VERSION"
    log_info "Total variants: ${#IMAGES[@]}"
}

# Run main function with all arguments
main "$@"