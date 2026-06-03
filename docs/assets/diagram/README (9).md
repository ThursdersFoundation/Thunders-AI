# Thunders AI — Diagram Collection

A comprehensive set of architectural and system diagrams for the **Thunders AI** platform. These diagrams illustrate the core infrastructure, AI pipelines, security flows, and deployment strategies that power Thunders AI's products.

---

## Diagram Index

| # | Category | File | Description |
|---|----------|------|-------------|
| 1 | Architecture | [`architecture/cloud_infrastructure.drawio`](architecture/cloud_infrastructure.drawio) | AWS cloud infrastructure — Region, VPC, Availability Zones, EC2, ALB, RDS, S3 |
| 2 | LLM | [`llm/reasoning_engine.drawio`](llm/reasoning_engine.drawio) | Reasoning engine — Strategy selection, CoT/ToT/ReAct branches, step execution, reflection |
| 3 | Multimodal | [`multimodal/image_understanding.drawio`](multimodal/image_understanding.drawio) | Image understanding pipeline — Pre-processing, feature extraction, detection, OCR, description generation |
| 4 | Robotics | [`robotics/sensor_fusion.drawio`](robotics/sensor_fusion.drawio) | Sensor fusion — Camera, LiDAR, Radar, IMU inputs through Kalman filtering to fused output |
| 5 | Cloud | [`cloud/distributed_training.drawio`](cloud/distributed_training.drawio) | Distributed training — Parameter server, worker nodes, gradient all-reduce, checkpointing |
| 6 | Security | [`security/threat_detection_flow.drawio`](security/threat_detection_flow.drawio) | Threat detection — Pattern matching, ML detection, risk scoring, block/allow/quarantine |
| 7 | API | [`api/request_pipeline.drawio`](api/request_pipeline.drawio) | API request pipeline — Rate limiting, auth, caching, routing, error handling |
| 8 | Database | [`database/storage_architecture.drawio`](database/storage_architecture.drawio) | Storage architecture — Hot/Warm/Cold tiers, model storage, data lifecycle |
| 9 | Deployment | [`deployment/edge_deployment.drawio`](deployment/edge_deployment.drawio) | Edge deployment — Cloud training, optimization, packaging, distribution, OTA updates |

---

## File Formats

Each diagram is available in the following formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| Draw.io Source | `.drawio` | Editable XML source — use draw.io to modify |
| Vector | `.svg` | Scalable vector graphic — ideal for docs and web |
| Raster | `.png` | High-resolution raster — ideal for presentations |

> The `.drawio` files in this directory are the **source of truth**. SVG and PNG files should be regenerated from these sources after any edits.

---

## Color Scheme

All diagrams use the **Thunders AI dark theme**:

| Color | Hex Code | Usage |
|-------|----------|-------|
| Background | `#0D1117` | Container and page backgrounds |
| Primary Blue | `#00B4FF` | Primary shapes, borders, accent elements |
| Accent Purple | `#6C3CE1` | Edges, secondary shapes, connections |
| Foreground Text | `#E6EDF3` | All text labels and values |

Additional status colors used contextually:

| Color | Hex Code | Usage |
|-------|----------|-------|
| Success / Allow | `#2ED573` | Positive outcomes, "allow" paths |
| Warning / Quarantine | `#FFA502` | Medium-risk, caution states |
| Danger / Block | `#FF4757` | Critical alerts, "block" paths |
| Hot Storage | `#FF6348` | Hot-tier data indicators |

---

## How to Edit `.drawio` Files

1. **Online editor** — Go to [app.diagrams.net](https://app.diagrams.net) and open the `.drawio` file from your local filesystem or GitHub.
2. **Desktop app** — Download [draw.io Desktop](https://github.com/jgraph/drawio-desktop/releases) and open the file directly.
3. **VS Code** — Install the **Draw.io Integration** extension (`hediet.vscode-drawio`) and edit files inline.

When editing:
- Maintain the dark theme styles defined above.
- Keep shape IDs unique and incrementing.
- Use `rounded=1` for all shapes.
- Use `orthogonalEdgeStyle` for all edges.

---

## How to Modify SVG Files

1. Edit the corresponding `.drawio` source file first.
2. Export to SVG from draw.io: **File → Export as → SVG**.
3. Alternatively, edit the SVG directly in a text editor or vector tool (Inkscape, Figma) for minor tweaks.
4. Ensure the SVG retains the dark theme color scheme.

---

## How to Regenerate PNG Files

1. Open the `.drawio` file in draw.io.
2. Go to **File → Export as → PNG**.
3. Set the following export options:
   - **Zoom:** 200% (for retina-quality output)
   - **Transparent Background:** No
   - **Include a copy of my diagram:** No
4. Save the PNG alongside the source `.drawio` file.

**Batch export** (command line with draw.io desktop):

```bash
# Export all .drawio files to PNG at 2x zoom
for f in **/*.drawio; do
  drawio --export --format png --scale 2 --output "${f%.drawio}.png" "$f"
done
```

**Batch export to SVG:**

```bash
# Export all .drawio files to SVG
for f in **/*.drawio; do
  drawio --export --format svg --output "${f%.drawio}.svg" "$f"
done
```

---

## Directory Structure

```
diagrams/
├── README.md                            ← You are here
├── architecture/
│   └── cloud_infrastructure.drawio
├── llm/
│   └── reasoning_engine.drawio
├── multimodal/
│   └── image_understanding.drawio
├── robotics/
│   └── sensor_fusion.drawio
├── cloud/
│   └── distributed_training.drawio
├── security/
│   └── threat_detection_flow.drawio
├── api/
│   └── request_pipeline.drawio
├── database/
│   └── storage_architecture.drawio
└── deployment/
    └── edge_deployment.drawio
```

---

## Contributing

When adding new diagrams:

1. Place the `.drawio` file in the appropriate category subdirectory.
2. Follow the dark theme style conventions described above.
3. Use the standard XML header with `agent="Thunders AI"`.
4. Update this README with the new diagram entry in the index table.
5. Export corresponding SVG and PNG files.

---

*Last updated: 2026-05-31 · Thunders AI Team*
