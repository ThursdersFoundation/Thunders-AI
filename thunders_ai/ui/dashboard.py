"""Dashboard framework for Thunders AI.

Provides real-time data visualisation with panels, charts,
and live-updating data feeds.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


class Panel:
    """A single panel within a dashboard.

    Attributes:
        panel_id: Unique panel identifier.
        title: Panel display title.
        chart_type: Type of visualisation.
    """

    def __init__(
        self,
        title: str,
        chart_type: ChartType = ChartType.LINE,
        width: int = 6,
        height: int = 4,
        refresh_interval: int = 5,
        data_source: Optional[str] = None,
    ) -> None:
        self.panel_id = f"panel-{uuid.uuid4().hex[:8]}"
        self.title = title
        self.chart_type = chart_type
        self.width = width
        self.height = height
        self.refresh_interval = refresh_interval
        self.data_source = data_source
        self.data: Dict[str, Any] = {}
        self.updated_at: Optional[float] = None
        self.created_at: float = time.time()

    def update_data(self, data: Dict[str, Any]) -> None:
        """Update the panel's data payload."""
        self.data = data
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the panel definition and data."""
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "chart_type": self.chart_type.value,
            "width": self.width,
            "height": self.height,
            "refresh_interval": self.refresh_interval,
            "data_source": self.data_source,
            "has_data": bool(self.data),
            "updated_at": self.updated_at,
        }


class Chart:
    """A chart configuration within a dashboard panel.

    Attributes:
        chart_id: Unique chart identifier.
        chart_type: Type of chart.
    """

    def __init__(
        self,
        chart_type: ChartType,
        title: str,
        x_axis: Optional[Dict[str, Any]] = None,
        y_axis: Optional[Dict[str, Any]] = None,
        series: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chart_id = f"chart-{uuid.uuid4().hex[:8]}"
        self.chart_type = chart_type
        self.title = title
        self.x_axis = x_axis or {"label": "X"}
        self.y_axis = y_axis or {"label": "Y"}
        self.series = series or []
        self.options = options or {}
        self.created_at: float = time.time()

    def add_series(self, name: str, data: List[Any], **kwargs: Any) -> None:
        """Add a data series to the chart.

        Args:
            name: Series name.
            data: Data points.
            **kwargs: Additional series options.
        """
        self.series.append({"name": name, "data": data, **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the chart definition."""
        return {
            "chart_id": self.chart_id,
            "type": self.chart_type.value,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "series_count": len(self.series),
            "series": self.series,
            "options": self.options,
        }


class Dashboard:
    """Real-time dashboard for Thunders AI.

    Manages panels, charts, and live data feeds for monitoring
    and visualising AI system performance.

    Attributes:
        title: Dashboard display title.
        panels: Registered panels.
        charts: Registered charts.
    """

    def __init__(
        self,
        title: str = "Thunders AI Dashboard",
        columns: int = 12,
        auto_refresh: bool = True,
        refresh_interval: int = 5,
        theme: str = "dark",
    ) -> None:
        self.title = title
        self.columns = columns
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self.theme = theme
        self.panels: Dict[str, Panel] = {}
        self.charts: Dict[str, Chart] = {}
        self._data_sources: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._last_update: Optional[float] = None

        logger.info("Dashboard initialised: '%s' (theme=%s)", title, theme)

    def create(
        self,
        layout: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create the dashboard with an optional layout specification.

        Args:
            layout: List of panel configurations for initial layout.

        Returns:
            Dashboard creation metadata.
        """
        dashboard_id = f"dash-{uuid.uuid4().hex[:8]}"

        for panel_cfg in (layout or []):
            self.add_panel(
                title=panel_cfg.get("title", "Untitled"),
                chart_type=ChartType(panel_cfg.get("chart_type", "line")),
                width=panel_cfg.get("width", 6),
                height=panel_cfg.get("height", 4),
            )

        result: Dict[str, Any] = {
            "dashboard_id": dashboard_id,
            "title": self.title,
            "columns": self.columns,
            "panels": len(self.panels),
            "charts": len(self.charts),
            "theme": self.theme,
            "auto_refresh": self.auto_refresh,
        }

        logger.info(
            "Dashboard created: %s (%d panels)", dashboard_id, len(self.panels)
        )
        return result

    def add_panel(
        self,
        title: str,
        chart_type: ChartType = ChartType.LINE,
        width: int = 6,
        height: int = 4,
        refresh_interval: Optional[int] = None,
        data_source: Optional[str] = None,
    ) -> Panel:
        """Add a panel to the dashboard.

        Args:
            title: Panel title.
            chart_type: Visualisation type.
            width: Panel width in grid columns.
            height: Panel height in grid rows.
            refresh_interval: Per-panel refresh interval.
            data_source: Named data source for auto-updates.

        Returns:
            The created Panel.
        """
        panel = Panel(
            title=title,
            chart_type=chart_type,
            width=width,
            height=height,
            refresh_interval=refresh_interval or self.refresh_interval,
            data_source=data_source,
        )
        self.panels[panel.panel_id] = panel

        logger.info("Panel added: '%s' (%s)", title, chart_type.value)
        return panel

    def add_chart(
        self,
        chart_type: ChartType,
        title: str,
        x_axis: Optional[Dict[str, Any]] = None,
        y_axis: Optional[Dict[str, Any]] = None,
        series: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        link_to_panel: Optional[str] = None,
    ) -> Chart:
        """Add a chart to the dashboard.

        Args:
            chart_type: Type of chart.
            title: Chart title.
            x_axis: X-axis configuration.
            y_axis: Y-axis configuration.
            series: Initial data series.
            options: Chart-specific options.
            link_to_panel: Panel ID to link this chart to.

        Returns:
            The created Chart.
        """
        chart = Chart(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            series=series,
            options=options,
        )
        self.charts[chart.chart_id] = chart

        if link_to_panel and link_to_panel in self.panels:
            self.panels[link_to_panel].update_data({"chart_id": chart.chart_id})

        logger.info("Chart added: '%s' (%s)", title, chart_type.value)
        return chart

    def update(
        self,
        panel_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update dashboard data.

        Args:
            panel_id: Specific panel to update; updates all if None.
            data: New data payload.

        Returns:
            Update summary.
        """
        now = time.time()
        self._last_update = now
        updated_panels: List[str] = []

        if panel_id:
            if panel_id not in self.panels:
                raise KeyError(f"Panel '{panel_id}' not found")
            self.panels[panel_id].update_data(data or {})
            updated_panels.append(panel_id)
        else:
            for pid, panel in self.panels.items():
                if panel.data_source and panel.data_source in self._data_sources:
                    try:
                        fresh_data = self._data_sources[panel.data_source]()
                        panel.update_data(fresh_data)
                    except Exception as exc:
                        logger.error("Data source error for %s: %s", panel.data_source, exc)
                elif data:
                    panel.update_data(data)
                updated_panels.append(pid)

        update_event: Dict[str, Any] = {
            "updated_panels": updated_panels,
            "timestamp": now,
        }

        for callback in self._update_callbacks:
            try:
                callback(update_event)
            except Exception as exc:
                logger.error("Update callback error: %s", exc)

        logger.debug("Dashboard updated: %d panels", len(updated_panels))
        return update_event

    def render(self, format: str = "json") -> Any:
        """Render the dashboard in the specified format.

        Args:
            format: Output format ('json', 'html', 'dict').

        Returns:
            Rendered dashboard representation.

        Raises:
            ValueError: If format is unsupported.
        """
        dashboard_data: Dict[str, Any] = {
            "title": self.title,
            "columns": self.columns,
            "theme": self.theme,
            "auto_refresh": self.auto_refresh,
            "refresh_interval": self.refresh_interval,
            "last_update": self._last_update,
            "panels": [p.to_dict() for p in self.panels.values()],
            "charts": [c.to_dict() for c in self.charts.values()],
            "rendered_at": time.time(),
        }

        if format == "json":
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format == "html":
            return self._render_html(dashboard_data)
        elif format == "dict":
            return dashboard_data
        else:
            raise ValueError(f"Unsupported render format: {format}")

    def register_data_source(
        self,
        name: str,
        provider: Callable[[], Dict[str, Any]],
    ) -> None:
        """Register a named data source for automatic panel updates.

        Args:
            name: Data source name.
            provider: Callable that returns fresh data.
        """
        self._data_sources[name] = provider
        logger.info("Data source registered: '%s'", name)

    def on_update(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for dashboard update events."""
        self._update_callbacks.append(callback)

    # -- Internal helpers ---------------------------------------------------

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render a simple HTML dashboard."""
        panels_html = ""
        for panel in data["panels"]:
            panels_html += (
                f'<div class="panel" style="grid-column: span {panel["width"]}; '
                f'grid-row: span {panel["height"]};">'
                f"<h3>{panel['title']}</h3>"
                f"<p>Type: {panel['chart_type']}</p>"
                f"</div>"
            )

        return (
            f"<!DOCTYPE html><html><head><title>{data['title']}</title></head>"
            f"<body><h1>{data['title']}</h1>"
            f'<div class="dashboard-grid">{panels_html}</div>'
            f"</body></html>"
        )
