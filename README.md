# Touchless Controller

Gesture-controlled desktop input using OpenCV + MediaPipe: move the cursor,
click, scroll, navigate slides, and adjust volume with hand gestures over a
webcam.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python touchless_controller.py
```

## Development

Install dev tools (lint, tests, packaging) as well:

```bash
pip install -r requirements-dev.txt
```

Before pushing:

```bash
black .              # auto-format
flake8 .             # lint
PYTHONPATH=. pytest -v tests/   # run tests
```

These three checks also run automatically in CI (`.github/workflows/ci.yml`)
on every push and pull request. The live camera/GUI loop isn't covered by
automated tests (no camera in CI) — the math and gesture-detection logic in
`tests/` is what's covered; the interactive loop is verified manually.

## Releasing a build

Releases are built automatically as a standalone Windows `.exe` whenever you
push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers `.github/workflows/release.yml`, which packages the app with
PyInstaller and attaches `TouchlessController.exe` to a new GitHub Release.
Use [semantic versioning](https://semver.org/) for tags (`vMAJOR.MINOR.PATCH`).
