#!/usr/bin/env python3
"""Install robot user units as regular files; never start a service."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SERVICES = ('rvclaw-nav2-persistent.service', 'rvclaw-vendor-command.service',
            'rvclaw-meter-gauge.service')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--with-camera-api', action='store_true',
                        help='install a standalone camera API only when no shared Studio API owns it')
    args = parser.parse_args()
    root = args.root.resolve()
    if any(character in str(root) for character in ('\n', '\r', '%', ' ', '"')):
        raise ValueError('service root must not contain systemd expansion or quoting characters')
    units = Path.home() / '.config/systemd/user'
    services = SERVICES
    if args.with_camera_api:
        result = subprocess.run(['systemctl', '--user', 'is-active', 'rvclaw-studio.service'],
                                capture_output=True, text=True)
        if result.stdout.strip() in {'active', 'activating', 'deactivating', 'reloading'}:
            raise RuntimeError('shared Studio API owns the camera service')
        import socket
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', 8017))
            except OSError as exc:
                raise RuntimeError('camera API port is already owned') from exc
        services += ('rvclaw-camera-api.service',)
    # Keep unrelated units and drop-ins intact. A running robot is not a
    # deployment target; prepare a separate root and stop owned units first.
    for service in services:
        result = subprocess.run(['systemctl', '--user', 'is-active', service],
                                capture_output=True, text=True)
        if result.stdout.strip() in {'active', 'activating', 'deactivating', 'reloading'}:
            raise RuntimeError('owned service is not stopped: ' + service)
        dropins = units / (service + '.d')
        if dropins.exists() and any(dropins.glob('*.conf')):
            raise RuntimeError('existing service overrides require separate review: ' + service)
        if service in SERVICES and not (root / 'runtime/k3/environment' / (service + '.env')).is_file():
            raise RuntimeError('prepare runtime before installing services')
    backup = root / 'runtime/k3/service-backup'
    backup.mkdir(parents=True, exist_ok=False)
    units.mkdir(parents=True, exist_ok=True)
    for service in services:
        destination = units / service
        if destination.is_file() or destination.is_symlink():
            shutil.copy2(destination, backup / service, follow_symlinks=False)
        text = (root / 'deploy/k3/systemd-user' / service).read_text(encoding='utf-8')
        text = text.replace('@ROBOT_ROOT@', str(root))
        with tempfile.NamedTemporaryFile(dir=units, prefix='.rvclaw-unit-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.chmod(0o644)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    for service in services:
        subprocess.run(['systemctl', '--user', 'enable', service], check=True)
    print('Robot units installed; no service was started')


if __name__ == '__main__':
    main()
