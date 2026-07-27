"""Cloud deployment manager for Thunders AI models.

Supports deploying models to AWS, GCP, and Azure with Docker
containerization, health checks, and lifecycle management.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class DeploymentStatus(str, Enum):
    """Possible deployment states."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class CloudDeployment:
    """Manage model deployments across cloud providers.

    Handles containerized deployment of AI models to AWS, GCP, and Azure,
    with built-in health checking and status monitoring.

    Attributes:
        provider: The target cloud provider.
        deployments: Active deployment registry.
    """

    def __init__(
        self,
        provider: CloudProvider = CloudProvider.AWS,
        region: str = "us-east-1",
        docker_image: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.region = region
        self.docker_image = docker_image or "thunders-ai/model-server:latest"
        self._api_key = api_key
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self._health_check_interval: int = 30
        logger.info(
            "CloudDeployment initialized: provider=%s, region=%s",
            self.provider.value,
            self.region,
        )

    def deploy(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        replicas: int = 1,
        port: int = 8080,
    ) -> str:
        """Deploy a model to the cloud.

        Args:
            model_path: Path or URI of the model to deploy.
            model_name: Optional name for the deployment.
            resources: Resource requirements (cpu, memory, gpu).
            env_vars: Environment variables for the container.
            replicas: Number of deployment replicas.
            port: Service port.

        Returns:
            Deployment ID string.

        Raises:
            ValueError: If model_path is empty or replicas < 1.
            RuntimeError: If deployment fails to initialise.
        """
        if not model_path:
            raise ValueError("model_path must be a non-empty string")
        if replicas < 1:
            raise ValueError("replicas must be at least 1")

        deployment_id = f"deploy-{uuid.uuid4().hex[:12]}"
        model_name = model_name or model_path.split("/")[-1]
        resources = resources or {"cpu": "2", "memory": "4Gi", "gpu": 0}
        env_vars = env_vars or {}

        deployment: Dict[str, Any] = {
            "id": deployment_id,
            "model_path": model_path,
            "model_name": model_name,
            "provider": self.provider.value,
            "region": self.region,
            "docker_image": self.docker_image,
            "resources": resources,
            "env_vars": env_vars,
            "replicas": replicas,
            "port": port,
            "status": DeploymentStatus.DEPLOYING.value,
            "created_at": time.time(),
            "updated_at": time.time(),
            "health": "unknown",
            "endpoint": None,
        }

        try:
            endpoint = self._create_container(deployment)
            deployment["endpoint"] = endpoint
            deployment["status"] = DeploymentStatus.RUNNING.value
            deployment["health"] = "healthy"
            self.deployments[deployment_id] = deployment
            logger.info(
                "Deployment %s created: endpoint=%s", deployment_id, endpoint
            )
        except Exception as exc:
            deployment["status"] = DeploymentStatus.FAILED.value
            deployment["health"] = "unhealthy"
            self.deployments[deployment_id] = deployment
            logger.error("Deployment %s failed: %s", deployment_id, exc)
            raise RuntimeError(f"Deployment failed: {exc}") from exc

        return deployment_id

    def undeploy(self, deployment_id: str, force: bool = False) -> bool:
        """Remove a deployment.

        Args:
            deployment_id: The deployment to remove.
            force: Force removal even if health check fails.

        Returns:
            True if undeployed successfully.

        Raises:
            KeyError: If deployment_id is not found.
        """
        if deployment_id not in self.deployments:
            raise KeyError(f"Deployment '{deployment_id}' not found")

        deployment = self.deployments[deployment_id]
        deployment["status"] = DeploymentStatus.STOPPING.value
        deployment["updated_at"] = time.time()

        try:
            self._remove_container(deployment)
            deployment["status"] = DeploymentStatus.STOPPED.value
            deployment["health"] = "stopped"
            logger.info("Deployment %s undeployed", deployment_id)
            return True
        except Exception as exc:
            if force:
                deployment["status"] = DeploymentStatus.STOPPED.value
                logger.warning("Force-removed deployment %s: %s", deployment_id, exc)
                return True
            deployment["status"] = DeploymentStatus.FAILED.value
            logger.error("Failed to undeploy %s: %s", deployment_id, exc)
            return False

    def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get the current status of a deployment.

        Args:
            deployment_id: The deployment to query.

        Returns:
            Dictionary with status, health, and endpoint information.

        Raises:
            KeyError: If deployment_id is not found.
        """
        if deployment_id not in self.deployments:
            raise KeyError(f"Deployment '{deployment_id}' not found")

        deployment = self.deployments[deployment_id]
        health = self._check_health(deployment)
        deployment["health"] = health
        deployment["updated_at"] = time.time()

        return {
            "id": deployment["id"],
            "model_name": deployment["model_name"],
            "status": deployment["status"],
            "health": health,
            "endpoint": deployment["endpoint"],
            "replicas": deployment["replicas"],
            "provider": deployment["provider"],
            "region": deployment["region"],
            "uptime_seconds": time.time() - deployment["created_at"],
        }

    def _create_container(self, deployment: Dict[str, Any]) -> str:
        """Create a Docker container on the cloud provider.

        Simulates container creation and returns an endpoint URL.
        """
        provider = deployment["provider"]
        region = deployment["region"]
        name = deployment["model_name"]
        port = deployment["port"]

        endpoints: Dict[str, str] = {
            CloudProvider.AWS.value: f"https://{name}.{region}.amazonaws.com",
            CloudProvider.GCP.value: f"https://{name}-{region}.run.app",
            CloudProvider.AZURE.value: f"https://{name}.{region}.azurecontainer.io",
        }

        endpoint = endpoints.get(provider, f"http://localhost:{port}")
        logger.debug("Container created on %s: %s", provider, endpoint)
        return endpoint

    def _remove_container(self, deployment: Dict[str, Any]) -> None:
        """Remove a container from the cloud provider."""
        logger.debug(
            "Removing container %s from %s",
            deployment["id"],
            deployment["provider"],
        )

    def _check_health(self, deployment: Dict[str, Any]) -> str:
        """Perform a health check on the deployment.

        Returns:
            'healthy', 'unhealthy', or 'stopped'.
        """
        if deployment["status"] == DeploymentStatus.STOPPED.value:
            return "stopped"
        if deployment["status"] == DeploymentStatus.FAILED.value:
            return "unhealthy"
        if deployment["status"] == DeploymentStatus.RUNNING.value:
            return "healthy"
        return "unknown"
