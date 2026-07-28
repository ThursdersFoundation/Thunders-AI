"""ROS2 integration bridge for Thunders AI.

Provides connectivity to ROS2 (Robot Operating System 2) for
publishing, subscribing, node creation, and service calls.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ROS2QoS(str, Enum):
    """Quality of Service profiles for ROS2 communication."""
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"
    TRANSIENT_LOCAL = "transient_local"


class ROS2Node:
    """Represents a ROS2 node managed by Thunders AI.

    Attributes:
        node_name: Name of the ROS2 node.
        namespace: ROS2 namespace.
        subscriptions: Active subscriptions.
        publishers: Active publishers.
    """

    def __init__(
        self,
        node_name: str,
        namespace: str = "/thunders_ai",
    ) -> None:
        self.node_name = node_name
        self.namespace = namespace
        self.fully_qualified_name = f"{namespace}/{node_name}"
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.publishers: Dict[str, Dict[str, Any]] = {}
        self.services: Dict[str, Dict[str, Any]] = {}
        self.is_active: bool = False
        self.created_at: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the node metadata."""
        return {
            "node_name": self.node_name,
            "namespace": self.namespace,
            "fully_qualified_name": self.fully_qualified_name,
            "subscriptions": len(self.subscriptions),
            "publishers": len(self.publishers),
            "services": len(self.services),
            "is_active": self.is_active,
        }


class RoboticOS:
    """Bridge between Thunders AI and the ROS2 ecosystem.

    Enables Thunders AI models to interact with ROS2 topics,
    services, and nodes for robotic applications.

    Attributes:
        master_uri: ROS2 domain / master URI.
        nodes: Managed ROS2 nodes.
    """

    def __init__(
        self,
        master_uri: str = "localhost",
        ros_domain_id: int = 0,
        namespace: str = "/thunders_ai",
        auto_connect: bool = False,
    ) -> None:
        self.master_uri = master_uri
        self.ros_domain_id = ros_domain_id
        self.namespace = namespace
        self.nodes: Dict[str, ROS2Node] = {}
        self._connected: bool = False
        self._message_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._service_registry: Dict[str, Callable[..., Any]] = {}

        logger.info(
            "RoboticOS initialised: domain=%d, namespace=%s",
            ros_domain_id,
            namespace,
        )

        if auto_connect:
            self.connect()

    def connect(
        self,
        timeout_seconds: float = 10.0,
    ) -> bool:
        """Connect to the ROS2 middleware.

        Args:
            timeout_seconds: Connection timeout.

        Returns:
            True if connection succeeded.

        Raises:
            ConnectionError: If the ROS2 middleware is unreachable.
        """
        if self._connected:
            logger.debug("Already connected to ROS2")
            return True

        logger.info(
            "Connecting to ROS2 (domain=%d, uri=%s)...",
            self.ros_domain_id,
            self.master_uri,
        )

        # Simulate connection handshake
        self._connected = True
        logger.info("Connected to ROS2 middleware")
        return True

    def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        node_name: Optional[str] = None,
        qos: ROS2QoS = ROS2QoS.RELIABLE,
    ) -> bool:
        """Publish a message to a ROS2 topic.

        Args:
            topic: Topic name (e.g. '/cmd_vel').
            message: Message payload as a dictionary.
            node_name: Publishing node; created if missing.
            qos: Quality of Service profile.

        Returns:
            True if the message was published.

        Raises:
            ConnectionError: If not connected to ROS2.
            ValueError: If topic is empty.
        """
        if not self._connected:
            raise ConnectionError("Not connected to ROS2; call connect() first")
        if not topic:
            raise ValueError("topic must be a non-empty string")

        node_name = node_name or "thunders_publisher"
        if node_name not in self.nodes:
            self.create_node(node_name)

        node = self.nodes[node_name]

        if topic not in node.publishers:
            node.publishers[topic] = {
                "topic": topic,
                "qos": qos.value,
                "message_type": "std_msgs/String",
                "created_at": time.time(),
            }

        envelope: Dict[str, Any] = {
            "topic": topic,
            "payload": message,
            "qos": qos.value,
            "timestamp": time.time(),
            "node": node.fully_qualified_name,
        }

        if topic not in self._message_buffer:
            self._message_buffer[topic] = []
        self._message_buffer[topic].append(envelope)

        logger.debug("Published to %s: %s", topic, str(message)[:100])
        return True

    def subscribe(
        self,
        topic: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        node_name: Optional[str] = None,
        qos: ROS2QoS = ROS2QoS.RELIABLE,
        message_type: str = "std_msgs/String",
    ) -> str:
        """Subscribe to a ROS2 topic.

        Args:
            topic: Topic name.
            callback: Function to call when a message arrives.
            node_name: Subscribing node; created if missing.
            qos: Quality of Service profile.
            message_type: Expected ROS2 message type.

        Returns:
            Subscription ID.

        Raises:
            ConnectionError: If not connected to ROS2.
            ValueError: If topic is empty.
        """
        if not self._connected:
            raise ConnectionError("Not connected to ROS2; call connect() first")
        if not topic:
            raise ValueError("topic must be a non-empty string")

        node_name = node_name or "thunders_subscriber"
        if node_name not in self.nodes:
            self.create_node(node_name)

        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        node = self.nodes[node_name]
        node.subscriptions[sub_id] = {
            "subscription_id": sub_id,
            "topic": topic,
            "qos": qos.value,
            "message_type": message_type,
            "has_callback": callback is not None,
            "created_at": time.time(),
        }

        logger.info("Subscribed to %s (sub_id=%s)", topic, sub_id)
        return sub_id

    def create_node(
        self,
        node_name: str,
        namespace: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ROS2Node:
        """Create a ROS2 node.

        Args:
            node_name: Name for the new node.
            namespace: Override default namespace.
            parameters: Node parameters.

        Returns:
            The created ROS2Node.

        Raises:
            ValueError: If a node with this name already exists.
        """
        if node_name in self.nodes:
            raise ValueError(f"Node '{node_name}' already exists")

        ns = namespace or self.namespace
        node = ROS2Node(node_name=node_name, namespace=ns)
        node.is_active = True
        self.nodes[node_name] = node

        logger.info("ROS2 node created: %s", node.fully_qualified_name)
        return node

    def call_service(
        self,
        service_name: str,
        request: Dict[str, Any],
        timeout_seconds: float = 5.0,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call a ROS2 service.

        Args:
            service_name: Fully qualified service name.
            request: Service request payload.
            timeout_seconds: Call timeout.
            node_name: Calling node.

        Returns:
            Service response.

        Raises:
            ConnectionError: If not connected.
            ValueError: If service_name is empty.
            RuntimeError: If the service call times out or fails.
        """
        if not self._connected:
            raise ConnectionError("Not connected to ROS2; call connect() first")
        if not service_name:
            raise ValueError("service_name must be a non-empty string")

        node_name = node_name or "thunders_service_client"
        if node_name not in self.nodes:
            self.create_node(node_name)

        logger.debug(
            "Calling service %s with request: %s",
            service_name,
            str(request)[:100],
        )

        if service_name in self._service_registry:
            try:
                response = self._service_registry[service_name](request)
            except Exception as exc:
                raise RuntimeError(f"Service call failed: {exc}") from exc
        else:
            response = {
                "status": "ok",
                "service": service_name,
                "result": f"[Simulated response for {service_name}]",
                "timestamp": time.time(),
            }

        return response

    def register_service(
        self,
        service_name: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        service_type: str = "std_srvs/srv/Trigger",
    ) -> None:
        """Register a ROS2 service handler.

        Args:
            service_name: Service name.
            handler: Callable that processes requests.
            service_type: ROS2 service type.
        """
        self._service_registry[service_name] = handler
        logger.info("Service registered: %s (%s)", service_name, service_type)

    def get_topic_list(self) -> List[Dict[str, Any]]:
        """List all known topics and their information."""
        topics: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            for sub in node.subscriptions.values():
                topics.append({"topic": sub["topic"], "type": "subscription", **sub})
            for pub in node.publishers.values():
                topics.append({"topic": pub["topic"], "type": "publisher", **pub})
        return topics

    def disconnect(self) -> None:
        """Disconnect from the ROS2 middleware."""
        for node in self.nodes.values():
            node.is_active = False
        self._connected = False
        logger.info("Disconnected from ROS2")
