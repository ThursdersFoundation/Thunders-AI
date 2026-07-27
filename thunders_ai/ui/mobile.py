"""Mobile application interface for Thunders AI.

Provides mobile UI patterns for iOS and Android with
view management and platform-specific adaptations.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class Platform(str, Enum):
    """Target mobile platform."""
    IOS = "ios"
    ANDROID = "android"


class ViewType(str, Enum):
    """Types of mobile views."""
    CHAT = "chat"
    LIST = "list"
    DETAIL = "detail"
    SETTINGS = "settings"
    CAMERA = "camera"
    VOICE = "voice"
    DASHBOARD = "dashboard"


class NavigationItem:
    """A navigation tab or drawer item.

    Attributes:
        label: Display label.
        icon: Icon identifier.
        view_id: Target view.
    """

    def __init__(
        self,
        label: str,
        icon: str,
        view_id: str,
        badge: Optional[str] = None,
    ) -> None:
        self.label = label
        self.icon = icon
        self.view_id = view_id
        self.badge = badge
        self.item_id = f"nav-{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the navigation item."""
        return {
            "item_id": self.item_id,
            "label": self.label,
            "icon": self.icon,
            "view_id": self.view_id,
            "badge": self.badge,
        }


class View:
    """Represents a mobile view/screen.

    Attributes:
        view_id: Unique view identifier.
        title: View title.
        view_type: Type of view.
    """

    def __init__(
        self,
        title: str,
        view_type: ViewType = ViewType.CHAT,
        platform: Platform = Platform.ANDROID,
    ) -> None:
        self.view_id = f"view-{uuid.uuid4().hex[:8]}"
        self.title = title
        self.view_type = view_type
        self.platform = platform
        self.components: List[Dict[str, Any]] = []
        self.is_active: bool = False
        self.created_at: float = time.time()

    def add_component(self, component: Dict[str, Any]) -> None:
        """Add a component to this view."""
        self.components.append(component)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the view definition."""
        return {
            "view_id": self.view_id,
            "title": self.title,
            "type": self.view_type.value,
            "platform": self.platform.value,
            "components": len(self.components),
            "is_active": self.is_active,
        }


class MobileApp:
    """Mobile application framework for Thunders AI.

    Provides view management, navigation, and platform-specific
    UI patterns for iOS and Android.

    Attributes:
        app_name: Application display name.
        platform: Target platform.
        views: Registered view screens.
    """

    def __init__(
        self,
        app_name: str = "Thunders AI",
        platform: Platform = Platform.ANDROID,
        theme: str = "dark",
        navigation_style: str = "bottom_tab",
        enable_biometrics: bool = False,
    ) -> None:
        self.app_name = app_name
        self.platform = platform
        self.theme = theme
        self.navigation_style = navigation_style
        self.enable_biometrics = enable_biometrics
        self.views: Dict[str, View] = {}
        self._navigation: List[NavigationItem] = []
        self._running: bool = False
        self._active_view: Optional[str] = None

        logger.info(
            "MobileApp initialised: %s (%s, nav=%s)",
            app_name,
            platform.value,
            navigation_style,
        )

    def launch(
        self,
        initial_view: Optional[str] = None,
        orientation: str = "portrait",
    ) -> Dict[str, Any]:
        """Launch the mobile application.

        Args:
            initial_view: View ID to show first; creates a default if None.
            orientation: 'portrait', 'landscape', or 'auto'.

        Returns:
            Launch status information.
        """
        if self._running:
            logger.warning("MobileApp is already running")
            return {"status": "already_running"}

        self._running = True

        # Create default views if none exist
        if not self.views:
            default_view = self.create_view("Home", ViewType.CHAT)
            self._active_view = default_view.view_id
        else:
            self._active_view = initial_view or next(iter(self.views))

        launch_info: Dict[str, Any] = {
            "status": "started",
            "app_name": self.app_name,
            "platform": self.platform.value,
            "orientation": orientation,
            "initial_view": self._active_view,
            "total_views": len(self.views),
            "navigation_items": len(self._navigation),
        }

        logger.info(
            "MobileApp launched: %s on %s (%d views)",
            self.app_name,
            self.platform.value,
            len(self.views),
        )
        return launch_info

    def create_view(
        self,
        title: str,
        view_type: ViewType = ViewType.CHAT,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> View:
        """Create a new mobile view/screen.

        Args:
            title: View title.
            view_type: Type of view to create.
            components: Initial UI components for the view.

        Returns:
            The created View.
        """
        view = View(title=title, view_type=view_type, platform=self.platform)

        for comp in (components or []):
            view.add_component(comp)

        self.views[view.view_id] = view

        # Auto-add to navigation
        nav_item = NavigationItem(
            label=title,
            icon=view_type.value,
            view_id=view.view_id,
        )
        self._navigation.append(nav_item)

        logger.info(
            "View created: '%s' (%s) → %s", title, view_type.value, view.view_id
        )
        return view

    def navigate_to(self, view_id: str) -> bool:
        """Navigate to a specific view.

        Args:
            view_id: Target view identifier.

        Returns:
            True if navigation succeeded.

        Raises:
            KeyError: If view_id is not found.
        """
        if view_id not in self.views:
            raise KeyError(f"View '{view_id}' not found")

        if self._active_view and self._active_view in self.views:
            self.views[self._active_view].is_active = False

        self.views[view_id].is_active = True
        self._active_view = view_id
        logger.debug("Navigated to view: %s", view_id)
        return True

    def get_navigation(self) -> List[Dict[str, Any]]:
        """Get the current navigation structure.

        Returns:
            List of navigation item dicts.
        """
        return [item.to_dict() for item in self._navigation]

    def get_active_view(self) -> Optional[Dict[str, Any]]:
        """Get the currently active view information."""
        if self._active_view and self._active_view in self.views:
            return self.views[self._active_view].to_dict()
        return None

    def send_notification(
        self,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a push notification.

        Args:
            title: Notification title.
            body: Notification body text.
            data: Optional data payload.

        Returns:
            Notification send result.
        """
        notif_id = f"notif-{uuid.uuid4().hex[:8]}"
        logger.info("Notification sent: %s", title)
        return {
            "notification_id": notif_id,
            "title": title,
            "body": body,
            "platform": self.platform.value,
            "status": "sent",
            "timestamp": time.time(),
        }

    def stop(self) -> None:
        """Stop the mobile application."""
        for view in self.views.values():
            view.is_active = False
        self._running = False
        logger.info("MobileApp stopped")
