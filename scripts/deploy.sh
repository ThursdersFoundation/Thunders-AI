#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Deploy Script
# =============================================================================
# Usage:
#   ./scripts/deploy.sh                          # Build & deploy locally
#   ./scripts/deploy.sh --registry ghcr.io/org   # Push to specific registry
#   ./scripts/deploy.sh --kubernetes             # Deploy to Kubernetes
#   ./scripts/deploy.sh --cloud aws              # Deploy to AWS
#   ./scripts/deploy.sh --tag v1.0.0             # Tag the deployment
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & Logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------------------------
REGISTRY=""
TAG="latest"
CLOUD=""
DEPLOY_KUBERNETES=false
DEPLOY_COMPOSE=false
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_HEALTHCHECK=false
NAMESPACE="thunders-ai"
REPLICAS=3
HEALTH_URL=""
HEALTH_TIMEOUT=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry)        REGISTRY="$2"; shift ;;
        --tag)             TAG="$2"; shift ;;
        --cloud)           CLOUD="$2"; shift ;;
        --kubernetes)      DEPLOY_KUBERNETES=true ;;
        --compose)         DEPLOY_COMPOSE=true ;;
        --skip-build)      SKIP_BUILD=true ;;
        --skip-push)       SKIP_PUSH=true ;;
        --skip-health)     SKIP_HEALTHCHECK=true ;;
        --namespace)       NAMESPACE="$2"; shift ;;
        --replicas)        REPLICAS="$2"; shift ;;
        --health-url)      HEALTH_URL="$2"; shift ;;
        --health-timeout)  HEALTH_TIMEOUT="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Docker:"
            echo "  --registry URL       Docker registry (e.g., ghcr.io/org)"
            echo "  --tag TAG            Image tag (default: latest)"
            echo "  --skip-build         Skip Docker build"
            echo "  --skip-push          Skip Docker push"
            echo ""
            echo "Deployment:"
            echo "  --kubernetes         Deploy to Kubernetes"
            echo "  --compose            Deploy with Docker Compose"
            echo "  --cloud PROVIDER     Deploy to cloud (aws|azure|gcp)"
            echo "  --namespace NS       K8s namespace (default: thunders-ai)"
            echo "  --replicas N         Number of replicas (default: 3)"
            echo ""
            echo "Health Check:"
            echo "  --health-url URL     Health check URL"
            echo "  --health-timeout N   Health check timeout in seconds (default: 120)"
            echo "  --skip-health        Skip health check"
            echo ""
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Determine Project Root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_NAME="thunders-ai"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE="${REGISTRY}/${FULL_IMAGE}"
fi

info "Thunders AI Deployment"
info "Image: $FULL_IMAGE"

# ---------------------------------------------------------------------------
# Step 1: Build Docker Image
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == false ]]; then
    info "Building Docker image: $FULL_IMAGE ..."

    docker build \
        -t "$FULL_IMAGE" \
        -f deployment/docker/Dockerfile \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$TAG" \
        .

    # Also tag as latest if a specific tag was given
    if [[ "$TAG" != "latest" ]] && [[ -n "$REGISTRY" ]]; then
        docker tag "$FULL_IMAGE" "${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    ok "Docker image built: $FULL_IMAGE"
else
    warn "Skipping Docker build"
fi

# ---------------------------------------------------------------------------
# Step 2: Push to Registry
# ---------------------------------------------------------------------------
if [[ "$SKIP_PUSH" == false ]] && [[ -n "$REGISTRY" ]]; then
    info "Pushing image to registry: $REGISTRY ..."

    # Check if logged in
    if ! docker pull "$REGISTRY/$IMAGE_NAME:latest" &>/dev/null 2>&1; then
        warn "May need to authenticate: docker login $REGISTRY"
    fi

    docker push "$FULL_IMAGE"

    if [[ "$TAG" != "latest" ]]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    ok "Image pushed to registry"
elif [[ -z "$REGISTRY" ]]; then
    warn "No registry specified, skipping push"
else
    warn "Skipping Docker push"
fi

# ---------------------------------------------------------------------------
# Step 3: Deploy
# ---------------------------------------------------------------------------
if [[ "$DEPLOY_KUBERNETES" == true ]]; then
    info "Deploying to Kubernetes (namespace: $NAMESPACE)..."

    # Check kubectl
    if ! command -v kubectl &>/dev/null; then
        error "kubectl not found. Install kubectl to deploy to Kubernetes."
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" 2>/dev/null || true

    # Update image in deployment
    if [[ -f deployment/docker/kubernetes.yaml ]]; then
        # Apply manifests
        kubectl apply -f deployment/docker/kubernetes.yaml -n "$NAMESPACE"

        # Update image if registry is specified
        if [[ -n "$REGISTRY" ]]; then
            kubectl set image deployment/thunders-ai \
                thunders-ai="${FULL_IMAGE}" \
                -n "$NAMESPACE"
        fi

        # Scale replicas
        kubectl scale deployment thunders-ai --replicas="$REPLICAS" -n "$NAMESPACE"

        ok "Kubernetes deployment applied"
    else
        error "deployment/docker/kubernetes.yaml not found"
    fi

elif [[ "$DEPLOY_COMPOSE" == true ]]; then
    info "Deploying with Docker Compose..."

    if [[ -f deployment/docker/docker-compose.yml ]]; then
        # Set image tag
        export THUNDERS_AI_IMAGE="$FULL_IMAGE"

        docker compose -f deployment/docker/docker-compose.yml up -d

        ok "Docker Compose deployment started"
    else
        error "deployment/docker/docker-compose.yml not found"
    fi

elif [[ -n "$CLOUD" ]]; then
    info "Deploying to $CLOUD..."

    case "$CLOUD" in
        aws)
            if [[ -f deployment/aws/deploy.sh ]]; then
                bash deployment/aws/deploy.sh --image "$FULL_IMAGE" --tag "$TAG"
            else
                info "Deploying to AWS ECS/EKS..."
                # Push to ECR
                AWS_REGION="${AWS_REGION:-us-east-1}"
                AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")

                if [[ -n "$AWS_ACCOUNT_ID" ]]; then
                    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
                    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" 2>/dev/null || true
                    docker tag "$FULL_IMAGE" "$ECR_URI:$TAG"
                    docker push "$ECR_URI:$TAG"
                    ok "Pushed to AWS ECR: $ECR_URI:$TAG"
                else
                    warn "AWS credentials not configured. Run 'aws configure' first."
                fi
            fi
            ;;

        azure)
            if [[ -f deployment/azure/deploy.sh ]]; then
                bash deployment/azure/deploy.sh --image "$FULL_IMAGE" --tag "$TAG"
            else
                info "Deploying to Azure Container Instances..."
                warn "Azure deployment script not found. Use --kubernetes with AKS instead."
            fi
            ;;

        gcp)
            if [[ -f deployment/gcp/deploy.sh ]]; then
                bash deployment/gcp/deploy.sh --image "$FULL_IMAGE" --tag "$TAG"
            else
                info "Deploying to Google Cloud Run..."
                if command -v gcloud &>/dev/null; then
                    GCR_URI="gcr.io/$(gcloud config get-value project 2>/dev/null || echo 'PROJECT_ID')/${IMAGE_NAME}"
                    docker tag "$FULL_IMAGE" "$GCR_URI:$TAG"
                    docker push "$GCR_URI:$TAG"
                    ok "Pushed to GCR: $GCR_URI:$TAG"
                else
                    warn "gcloud CLI not found. Install Google Cloud SDK."
                fi
            fi
            ;;

        *)
            error "Unknown cloud provider: $CLOUD. Supported: aws, azure, gcp"
            ;;
    esac

    ok "Cloud deployment initiated"
else
    warn "No deployment target specified. Use --kubernetes, --compose, or --cloud PROVIDER"
    info "Image is ready: $FULL_IMAGE"
fi

# ---------------------------------------------------------------------------
# Step 4: Health Check
# ---------------------------------------------------------------------------
if [[ "$SKIP_HEALTHCHECK" == false ]] && [[ -n "$HEALTH_URL" ]]; then
    info "Running health check against $HEALTH_URL ..."

    ELAPSED=0
    while [[ $ELAPSED -lt $HEALTH_TIMEOUT ]]; do
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            ok "Health check passed!"
            break
        fi
        info "Waiting for service... (${ELAPSED}s / ${HEALTH_TIMEOUT}s)"
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    if [[ $ELAPSED -ge $HEALTH_TIMEOUT ]]; then
        error "Health check timed out after ${HEALTH_TIMEOUT}s"
    fi
elif [[ "$DEPLOY_KUBERNETES" == true ]]; then
    info "Waiting for Kubernetes rollout..."
    if command -v kubectl &>/dev/null; then
        if kubectl rollout status deployment/thunders-ai -n "$NAMESPACE" --timeout="${HEALTH_TIMEOUT}s"; then
            ok "Kubernetes rollout complete"
        else
            warn "Kubernetes rollout may still be in progress"
        fi
    fi
elif [[ "$DEPLOY_COMPOSE" == true ]]; then
    info "Waiting for Docker Compose services..."
    sleep 10
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        ok "Health check passed on localhost:8000"
    else
        warn "Service may still be starting up"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
ok "Deployment completed!"
echo "  Image:      $FULL_IMAGE"
echo "  Namespace:  $NAMESPACE"
echo "  Replicas:   $REPLICAS"
if [[ "$DEPLOY_KUBERNETES" == true ]]; then
    echo "  Platform:   Kubernetes"
    echo "  Check:      kubectl get all -n $NAMESPACE"
elif [[ "$DEPLOY_COMPOSE" == true ]]; then
    echo "  Platform:   Docker Compose"
    echo "  Check:      docker compose ps"
elif [[ -n "$CLOUD" ]]; then
    echo "  Platform:   $CLOUD"
fi
echo "============================================"
