#!/bin/bash

# SpinTron-NN-Kit Release Script
# Automates the release process with semantic versioning

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

# Default values
RELEASE_TYPE="patch"
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false
SKIP_DOCKER=false
PUSH_DOCKER=false

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

Automate the release process for SpinTron-NN-Kit.

OPTIONS:
    -t, --type TYPE        Release type: major, minor, or patch (default: patch)
    -d, --dry-run          Perform a dry run without making changes
    -s, --skip-tests       Skip test execution
    -b, --skip-build       Skip build process
    -D, --skip-docker      Skip Docker image building
    -p, --push-docker      Push Docker images to registry
    -h, --help             Show this help message

EXAMPLES:
    $0 --type minor
    $0 --type patch --push-docker
    $0 --dry-run --type major

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    # Check if we're on main branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "main" ]]; then
        log_warning "Not on main branch (current: $current_branch)"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check required tools
    for tool in python3 pip3 git; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

get_current_version() {
    if [[ -f "$PROJECT_ROOT/spintron_nn/__init__.py" ]]; then
        python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    import spintron_nn
    print(spintron_nn.__version__)
except:
    print('0.1.0')
"
    else
        echo "0.1.0"
    fi
}

calculate_new_version() {
    local current_version=$1
    local release_type=$2
    
    # Parse version (assuming semantic versioning)
    IFS='.' read -ra VERSION_PARTS <<< "$current_version"
    major=${VERSION_PARTS[0]}
    minor=${VERSION_PARTS[1]}
    patch=${VERSION_PARTS[2]}
    
    case $release_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid release type: $release_type"
            exit 1
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

update_version_files() {
    local new_version=$1
    
    log_info "Updating version to $new_version..."
    
    # Update __init__.py
    if [[ -f "$PROJECT_ROOT/spintron_nn/__init__.py" ]]; then
        sed -i.bak "s/__version__ = .*/__version__ = \"$new_version\"/" "$PROJECT_ROOT/spintron_nn/__init__.py"
        rm "$PROJECT_ROOT/spintron_nn/__init__.py.bak"
    fi
    
    # Update pyproject.toml
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        sed -i.bak "s/version = .*/version = \"$new_version\"/" "$PROJECT_ROOT/pyproject.toml"
        rm "$PROJECT_ROOT/pyproject.toml.bak"
    fi
    
    # Update setup.cfg
    if [[ -f "$PROJECT_ROOT/setup.cfg" ]]; then
        sed -i.bak "s/version = .*/version = $new_version/" "$PROJECT_ROOT/setup.cfg"
        rm "$PROJECT_ROOT/setup.cfg.bak"
    fi
    
    # Update package.json
    if [[ -f "$PROJECT_ROOT/package.json" ]]; then
        python3 -c "
import json
import sys

with open('$PROJECT_ROOT/package.json', 'r') as f:
    data = json.load(f)

data['version'] = '$new_version'

with open('$PROJECT_ROOT/package.json', 'w') as f:
    json.dump(data, f, indent=2)
"
    fi
    
    log_success "Version files updated"
}

generate_changelog() {
    local current_version=$1
    local new_version=$2
    
    log_info "Generating changelog..."
    
    # Get commits since last tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    
    if [[ -n "$last_tag" ]]; then
        commits=$(git log --pretty=format:"- %s" "$last_tag"..HEAD)
    else
        commits=$(git log --pretty=format:"- %s")
    fi
    
    # Create changelog entry
    changelog_entry="## [$new_version] - $(date +%Y-%m-%d)

### Changes
$commits

"
    
    # Prepend to CHANGELOG.md
    if [[ -f "$PROJECT_ROOT/CHANGELOG.md" ]]; then
        temp_file=$(mktemp)
        echo "$changelog_entry" > "$temp_file"
        cat "$PROJECT_ROOT/CHANGELOG.md" >> "$temp_file"
        mv "$temp_file" "$PROJECT_ROOT/CHANGELOG.md"
    else
        echo "# Changelog

$changelog_entry" > "$PROJECT_ROOT/CHANGELOG.md"
    fi
    
    log_success "Changelog updated"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests (--skip-tests)"
        return
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run comprehensive test suite
    python3 -m pytest tests/ -v --cov=spintron_nn --cov-report=term
    
    log_success "Tests passed"
}

build_package() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping build (--skip-build)"
        return
    fi
    
    log_info "Building package..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build package
    python3 -m build
    
    log_success "Package built"
}

build_docker_images() {
    if [[ "$SKIP_DOCKER" == "true" ]]; then
        log_info "Skipping Docker build (--skip-docker)"
        return
    fi
    
    local new_version=$1
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build --target production -t spintron-nn-kit:$new_version .
    docker tag spintron-nn-kit:$new_version spintron-nn-kit:latest
    
    log_success "Docker images built"
}

push_docker_images() {
    if [[ "$PUSH_DOCKER" != "true" ]]; then
        log_info "Skipping Docker push (use --push-docker to enable)"
        return
    fi
    
    local new_version=$1
    
    log_info "Pushing Docker images..."
    
    # Push to registry (assumes you're logged in)
    docker push spintron-nn-kit:$new_version
    docker push spintron-nn-kit:latest
    
    log_success "Docker images pushed"
}

create_git_tag() {
    local new_version=$1
    
    log_info "Creating git tag..."
    
    cd "$PROJECT_ROOT"
    
    # Commit version changes
    git add .
    git commit -m "chore: bump version to $new_version"
    
    # Create annotated tag
    git tag -a "v$new_version" -m "Release version $new_version"
    
    log_success "Git tag created: v$new_version"
}

push_changes() {
    log_info "Pushing changes to remote..."
    
    cd "$PROJECT_ROOT"
    
    # Push commits and tags
    git push origin main
    git push origin --tags
    
    log_success "Changes pushed to remote"
}

create_github_release() {
    local new_version=$1
    
    # Check if GitHub CLI is available
    if ! command -v gh &> /dev/null; then
        log_warning "GitHub CLI not found, skipping GitHub release creation"
        return
    fi
    
    log_info "Creating GitHub release..."
    
    cd "$PROJECT_ROOT"
    
    # Extract changelog for this version
    changelog_content=$(sed -n "/## \[$new_version\]/,/## \[/p" CHANGELOG.md | head -n -1)
    
    # Create release
    gh release create "v$new_version" \
        --title "Release $new_version" \
        --notes "$changelog_content" \
        dist/*
    
    log_success "GitHub release created"
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                RELEASE_TYPE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -b|--skip-build)
                SKIP_BUILD=true
                shift
                ;;
            -D|--skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            -p|--push-docker)
                PUSH_DOCKER=true
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
    
    # Validate release type
    if [[ "$RELEASE_TYPE" != "major" && "$RELEASE_TYPE" != "minor" && "$RELEASE_TYPE" != "patch" ]]; then
        log_error "Invalid release type: $RELEASE_TYPE"
        usage
        exit 1
    fi
    
    log_info "Starting release process (type: $RELEASE_TYPE)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Execute release steps
    check_prerequisites
    
    current_version=$(get_current_version)
    new_version=$(calculate_new_version "$current_version" "$RELEASE_TYPE")
    
    log_info "Version: $current_version → $new_version"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would perform the following actions:"
        echo "  1. Update version files"
        echo "  2. Generate changelog"
        echo "  3. Run tests"
        echo "  4. Build package"
        echo "  5. Build Docker images"
        echo "  6. Create git tag"
        echo "  7. Push changes"
        echo "  8. Create GitHub release"
        exit 0
    fi
    
    # Confirm release
    read -p "Create release $new_version? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Release cancelled"
        exit 0
    fi
    
    # Execute release
    update_version_files "$new_version"
    generate_changelog "$current_version" "$new_version"
    run_tests
    build_package
    build_docker_images "$new_version"
    create_git_tag "$new_version"
    push_changes
    push_docker_images "$new_version"
    create_github_release "$new_version"
    
    log_success "Release $new_version completed successfully!"
    
    # Display summary
    echo
    log_info "Release Summary:"
    echo "  Version: $current_version → $new_version"
    echo "  Git Tag: v$new_version"
    echo "  Packages: $(ls dist/ 2>/dev/null | wc -l) files in dist/"
    if [[ "$SKIP_DOCKER" != "true" ]]; then
        echo "  Docker Images: spintron-nn-kit:$new_version, spintron-nn-kit:latest"
    fi
    echo
    echo "Next steps:"
    echo "  - Verify the release on GitHub"
    echo "  - Update documentation if needed"
    echo "  - Announce the release"
}

# Run main function
main "$@"