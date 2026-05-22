# OpenClaw Adapter Contract

OpenClaw and Feishu are not part of the v0.1.1 main demo path. They are v0.2+
integration targets after the Web + CV demo is stable on K3.

## Positioning

```text
RVClaw Agent Core
  -> Safety Guard
  -> Skill Registry
  -> Device Adapter API
  -> OpenClaw / ROS2 / Feishu sidecar
```

OpenClaw should be treated as a device adapter, not as the product center. The
RVClaw product center remains the RISC-V edge Agent Runtime: planning, memory,
safety, observability, benchmark, and deployment.

## Adapter Boundary

The adapter must expose the same skill-level contract already used by the mock
device:

```text
move_to(target)
capture_image(target, mode)
detect_status(target, image_ref)
speak(text)
upload_report(title)
stop(reason)
```

Each call must return JSON-serializable output and must write enough metadata
for `trace.jsonl`, `metrics.json`, and `report.md`.

## Safety Gate

- Skill name must exist in the registry.
- Target zone must be configured in `configs/zones.yaml`.
- Motion or actuator skills must support `requires_confirmation`.
- `stop` must always remain available.
- Feishu or other chat ingress must never bypass Safety Guard.

## Feishu Sidecar

The Feishu bot should be a sidecar ingress:

```text
Feishu message
  -> sidecar validates sender and workspace
  -> RVClaw Web/API POST /api/runs
  -> RVClaw returns run summary and report link
```

Do not let the Feishu sidecar call robot/device APIs directly. It should only
submit tasks and read run artifacts through RVClaw APIs.

## v0.2 Readiness Checklist

- OpenClaw SDK can run on K3 or a companion host.
- Node/runtime dependencies are pinned.
- Token and app secrets are loaded from environment variables only.
- A dry-run adapter can replay actions without hardware.
- Human confirmation is implemented for controlled motion and actuator actions.
- A demo can be stopped by both Web UI and sidecar command.
