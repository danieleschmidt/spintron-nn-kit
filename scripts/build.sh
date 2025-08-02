#!/bin/bash

# SpinTron-NN-Kit Build Script
# Builds and packages the project for different deployment scenarios

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Default values
BUILD_TYPE="development"
VERBOSE=false
CLEAN=false
TEST=false
DOCKER=false
UPLOAD=false

# Functions
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
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build SpinTron-NN-Kit for different deployment scenarios.

OPTIONS:
    -t, --type TYPE         Build type: development, production, or all (default: development)
    -c, --clean            Clean build artifacts before building
    -T, --test             Run tests after building
    -d, --docker           Build Docker images
    -u, --upload           Upload packages to PyPI (production only)
    -v, --verbose          Verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0 --type development --test
    $0 --type production --docker --upload
    $0 --clean --type all --verbose

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check build tools
    if ! python3 -c "import build" 2>/dev/null; then
        log_warning "build package not found, installing..."
        pip3 install build
    fi
    
    if [[ "$DOCKER" == "true" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is required but not installed"
            exit 1
        fi
    fi
    
    log_success "Dependencies check passed"
}

clean_build() {
    log_info "Cleaning build artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Remove Python build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    
    # Remove test artifacts
    rm -rf .pytest_cache/
    rm -rf .coverage
    rm -rf htmlcov/
    
    # Remove documentation build
    rm -rf docs/_build/
    
    log_success "Build artifacts cleaned"
}

setup_environment() {
    log_info "Setting up build environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    mkdir -p "$DIST_DIR"
    
    # Install build dependencies
    pip3 install --upgrade pip wheel setuptools build
    
    if [[ "$BUILD_TYPE" == "development" || "$BUILD_TYPE" == "all" ]]; then
        pip3 install -e ".[dev]"
    fi
    
    log_success "Build environment ready"
}

build_python_package() {
    log_info "Building Python package..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$VERBOSE" == "true" ]]; then
        python3 -m build --verbose
    else
        python3 -m build
    fi
    
    log_success "Python package built successfully"
    
    # List built packages
    log_info "Built packages:"
    ls -la dist/
}

build_docker_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build development image
    if [[ "$BUILD_TYPE" == "development" || "$BUILD_TYPE" == "all" ]]; then
        log_info "Building development image..."
        docker build --target development -t spintron-nn-kit:dev .
        log_success "Development image built"
    fi
    
    # Build production image
    if [[ "$BUILD_TYPE" == "production" || "$BUILD_TYPE" == "all" ]]; then
        log_info "Building production image..."
        docker build --target production -t spintron-nn-kit:latest .
        log_success "Production image built"
        
        # Tag with version
        VERSION=$(python3 -c "import spintron_nn; print(spintron_nn.__version__)" 2>/dev/null || echo "0.1.0")
        docker tag spintron-nn-kit:latest spintron-nn-kit:$VERSION
        log_info "Tagged as spintron-nn-kit:$VERSION"
    fi
    
    # Build simulation image
    if [[ "$BUILD_TYPE" == "development" || "$BUILD_TYPE" == "all" ]]; then
        log_info "Building simulation image..."
        docker build --target simulation -t spintron-nn-kit:simulation .
        log_success "Simulation image built"
    fi
    
    # Build benchmarking image
    if [[ "$BUILD_TYPE" == "all" ]]; then
        log_info "Building benchmarking image..."
        docker build --target benchmarking -t spintron-nn-kit:benchmark .
        log_success "Benchmarking image built"
    fi
    
    log_success "Docker images built successfully"
    
    # List built images
    log_info "Built images:"
    docker images spintron-nn-kit
}

run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run tests with coverage
    if [[ "$VERBOSE" == "true" ]]; then
        pytest tests/ -v --cov=spintron_nn --cov-report=html --cov-report=term-missing
    else
        pytest tests/ --cov=spintron_nn --cov-report=term
    fi
    
    log_success "Tests passed"
}

upload_package() {
    if [[ "$BUILD_TYPE" != "production" ]]; then
        log_warning "Skipping upload - only production builds can be uploaded"
        return
    fi
    
    log_info "Uploading package to PyPI..."
    
    # Check if twine is available
    if ! command -v twine &> /dev/null; then
        log_info "Installing twine..."
        pip3 install twine
    fi
    
    # Upload to PyPI
    twine upload dist/*
    
    log_success "Package uploaded to PyPI"
}

generate_build_info() {
    log_info "Generating build information..."
    
    cd "$PROJECT_ROOT"
    
    BUILD_INFO_FILE="$BUILD_DIR/build_info.json"
    
    cat > "$BUILD_INFO_FILE" << EOF
{
    "build_type": "$BUILD_TYPE",
    "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "python_version": "$(python3 --version)",
    "platform": "$(uname -a)",
    "packages": $(ls dist/*.whl dist/*.tar.gz 2>/dev/null | jq -R . | jq -s . || echo '[]')
}
EOF
    
    log_success "Build info generated: $BUILD_INFO_FILE"
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            -T|--test)
                TEST=true
                shift
                ;;
            -d|--docker)
                DOCKER=true
                shift
                ;;
            -u|--upload)
                UPLOAD=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate build type
    if [[ "$BUILD_TYPE" != "development" && "$BUILD_TYPE" != "production" && "$BUILD_TYPE" != "all" ]]; then
        log_error "Invalid build type: $BUILD_TYPE"
        usage
        exit 1
    fi
    
    log_info "Starting SpinTron-NN-Kit build (type: $BUILD_TYPE)"
    
    # Execute build steps
    check_dependencies
    
    if [[ "$CLEAN" == "true" ]]; then
        clean_build
    fi
    
    setup_environment
    build_python_package
    
    if [[ "$DOCKER" == "true" ]]; then
        build_docker_images
    fi
    
    if [[ "$TEST" == "true" ]]; then
        run_tests
    fi
    
    generate_build_info
    
    if [[ "$UPLOAD" == "true" ]]; then
        upload_package
    fi
    
    log_success "Build completed successfully!"
    
    # Display summary
    echo
    log_info "Build Summary:"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Packages: $(ls dist/ 2>/dev/null | wc -l) files in dist/"
    if [[ "$DOCKER" == "true" ]]; then
        echo "  Docker Images: $(docker images spintron-nn-kit --format "table {{.Tag}}" | tail -n +2 | wc -l) images built"
    fi
    echo "  Build Info: $BUILD_DIR/build_info.json"
}

# Run main function
main "$@"