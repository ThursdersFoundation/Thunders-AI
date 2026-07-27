"""Desktop application interface for Thunders AI.

Provides a desktop GUI framework with window management,
widget system, and system tray integration.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class WindowState(str, Enum):
    """Possible window states."""
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    CLOSED = "closed"


class Widget:
    """Represents a UI widget within a desktop window.

    Attributes:
        widget_id: Unique widget identifier.
        widget_type: Type of widget (button, label, chart, etc.).
    """

    def __init__(
        self,
        name: str,
        widget_type: str,
        parent_window: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.widget_id = f"widget-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.widget_type = widget_type
        self.parent_window = parent_window
        self.config = config or {}
        self.visible: bool = True
        self.enabled: bool = True
        self.created_at: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise widget metadata."""
        return {
            "widget_id": self.widget_id,
            "name": self.name,
            "type": self.widget_type,
            "parent_window": self.parent_window,
            "visible": self.visible,
            "enabled": self.enabled,
        }


class Window:
    """Represents a desktop application window.

    Attributes:
        window_id: Unique window identifier.
        title: Window title.
        state: Current window state.
    """

    def __init__(
        self,
        title: str,
        width: int = 800,
        height: int = 600,
        resizable: bool = True,
    ) -> None:
        self.window_id = f"win-{uuid.uuid4().hex[:8]}"
        self.title = title
        self.width = width
        self.height = height
        self.resizable = resizable
        self.state: WindowState = WindowState.NORMAL
        self.widgets: Dict[str, Widget] = {}
        self.created_at: float = time.time()

    def add_widget(self, widget: Widget) -> None:
        """Add a widget to this window."""
        self.widgets[widget.widget_id] = widget

    def to_dict(self) -> Dict[str, Any]:
        """Serialise window metadata."""
        return {
            "window_id": self.window_id,
            "title": self.title,
            "size": f"{self.width}x{self.height}",
            "state": self.state.value,
            "widgets": len(self.widgets),
        }


class SystemTrayIcon:
    """System tray icon integration.

    Attributes:
        icon_path: Path to the tray icon image.
        tooltip: Hover tooltip text.
    """

    def __init__(
        self,
        icon_path: Optional[str] = None,
        tooltip: str = "Thunders AI",
        menu_items: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.icon_path = icon_path
        self.tooltip = tooltip
        self.menu_items = menu_items or [
            {"label": "Open", "action": "show"},
            {"label": "Quit", "action": "quit"},
        ]
        self.is_visible: bool = False

    def show(self) -> None:
        """Show the tray icon."""
        self.is_visible = True
        logger.debug("System tray icon shown")

    def hide(self) -> None:
        """Hide the tray icon."""
        self.is_visible = False
        logger.debug("System tray icon hidden")

    def notify(self, title: str, message: str) -> None:
        """Display a system notification."""
        logger.info("Notification: %s - %s", title, message)


class DesktopApp:
    """Desktop application framework for Thunders AI.

    Manages windows, widgets, and system tray integration
    for a native desktop experience.

    Attributes:
        app_name: Application display name.
        windows: Managed application windows.
    """

    def __init__(
        self,
        app_name: str = "Thunders AI",
        version: str = "1.0.0",
        single_instance: bool = True,
        enable_tray: bool = True,
        theme: str = "dark",
    ) -> None:
        self.app_name = app_name
        self.version = version
        self.single_instance = single_instance
        self.enable_tray = enable_tray
        self.theme = theme
        self.windows: Dict[str, Window] = {}
        self._tray: Optional[SystemTrayIcon] = None
        self._running: bool = False
        self._event_handlers: Dict[str, List[Callable[..., Any]]] = {}

        logger.info(
            "DesktopApp initialised: %s v%s (theme=%s)",
            app_name,
            version,
            theme,
        )

    def launch(
        self,
        splash: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Launch the desktop application.

        Args:
            splash: Show a splash screen on startup.
            debug: Enable debug mode.

        Returns:
            Launch status information.
        """
        if self._running:
            logger.warning("DesktopApp is already running")
            return {"status": "already_running"}

        self._running = True

        if self.enable_tray:
            self._tray = SystemTrayIcon()
            self._tray.show()

        # Create main window if none exist
        if not self.windows:
            self.create_window(f"{self.app_name} - Main")

        launch_info: Dict[str, Any] = {
            "status": "started",
            "app_name": self.app_name,
            "version": self.version,
            "windows": len(self.windows),
            "tray_enabled": self._tray is not None,
            "debug": debug,
            "pid": uuid.uuid4().hex[:8],
        }

        logger.info(
            "DesktopApp launched: %d windows, tray=%s",
            len(self.windows),
            self._tray is not None,
        )
        return launch_info

    def create_window(
        self,
        title: str,
        width: int = 800,
        height: int = 600,
        resizable: bool = True,
        on_close: Optional[Callable[[], None]] = None,
    ) -> Window:
        """Create a new application window.

        Args:
            title: Window title.
            width: Initial width in pixels.
            height: Initial height in pixels.
            resizable: Whether the window can be resized.
            on_close: Callback when the window is closed.

        Returns:
            The created Window object.
        """
        window = Window(
            title=title, width=width, height=height, resizable=resizable
        )
        self.windows[window.window_id] = window

        logger.info("Window created: '%s' (%dx%d)", title, width, height)
        return window

    def add_widget(
        self,
        window_id: str,
        name: str,
        widget_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Widget:
        """Add a widget to a specific window.

        Args:
            window_id: Target window identifier.
            name: Widget display name.
            widget_type: Type of widget.
            config: Widget configuration.

        Returns:
            The created Widget.

        Raises:
            KeyError: If window_id is not found.
        """
        if window_id not in self.windows:
            raise KeyError(f"Window '{window_id}' not found")

        widget = Widget(
            name=name,
            widget_type=widget_type,
            parent_window=window_id,
            config=config,
        )
        self.windows[window_id].add_widget(widget)

        logger.debug(
            "Widget '%s' (%s) added to window %s", name, widget_type, window_id
        )
        return widget

    def close_window(self, window_id: str) -> bool:
        """Close a specific window.

        Args:
            window_id: The window to close.

        Returns:
            True if the window was closed.

        Raises:
            KeyError: If window_id is not found.
        """
        if window_id not in self.windows:
            raise KeyError(f"Window '{window_id}' not found")

        window = self.windows[window_id]
        window.state = WindowState.CLOSED
        logger.info("Window closed: '%s'", window.title)
        return True

    def quit(self) -> None:
        """Quit the entire application."""
        for window in self.windows.values():
            window.state = WindowState.CLOSED

        if self._tray:
            self._tray.hide()

        self._running = False
        logger.info("DesktopApp quit")
