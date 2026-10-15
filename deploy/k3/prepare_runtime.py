#!/usr/bin/env python3
"""Assemble the robot runtime without starting services or accessing devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil


def child(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('path must remain relative: ' + name)
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes runtime root: ' + name)
    return target


def copy_file(source, target, mode=0o644):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(mode)


def workspace_link(target, destination):
    if destination.exists() or destination.is_symlink():
        raise RuntimeError('runtime destination already exists: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(os.path.relpath(target, destination.parent), target_is_directory=True)
    except OSError:
        # Source-only assembly can also be inspected on Windows without link
        # privileges. Board builds use the linked workspace on Linux.
        shutil.copytree(target, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--asset-root', type=Path,
                        help='compatible robot assets in the same relative layout')
    parser.add_argument('--source-only', action='store_true',
                        help='assemble source paths without model/library assets')
    args = parser.parse_args()
    root = args.root.resolve()
    runtime = root / 'runtime/k3'
    if runtime.exists():
        raise RuntimeError('assemble only a new runtime; active trees must not be overwritten')
    layout = json.loads((root / 'deploy/k3/runtime_layout.json').read_text(encoding='utf-8'))
    if not args.source_only:
        if args.asset_root is None:
            raise RuntimeError('--asset-root is required for a runnable deployment')
        assets = args.asset_root.resolve()
        manifest = json.loads((root / 'models/runtime_assets.json').read_text(encoding='utf-8'))
        for item in manifest['assets']:
            repository_source = child(root, item['path'])
            source = repository_source if repository_source.is_file() else child(assets, item['path'])
            if not source.is_file():
                raise RuntimeError('missing runtime asset: ' + item['path'])
            if source.stat().st_size != item['bytes']:
                raise RuntimeError('runtime asset size differs: ' + item['path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError('runtime asset hash differs: ' + item['path'])
        for item in manifest['external_resources']['vision_model_cache']['assets']:
            source = child(Path.home() / '.cache/models/vision', item['path'])
            if not source.is_file() or source.stat().st_size != item['bytes']:
                raise RuntimeError('missing or incompatible vision model: ' + item['path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError('vision model hash differs: ' + item['path'])
        for name in manifest['directory_assets']:
            if not child(assets, name).is_dir():
                raise RuntimeError('missing dependency directory: ' + name)
        for item in manifest['assets']:
            source, target = child(assets, item['path']), child(root, item['path'])
            if not target.is_file():
                copy_file(source, target)
        # Mutable localization/task state is initialized on first use.
        for name in manifest['directory_assets']:
            source, target = child(assets, name), child(root, name)
            if source != target:
                shutil.copytree(source, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtime.mkdir(parents=True, exist_ok=True)
    for item in layout['scripts']:
        copy_file(child(root, item['source']),
                  child(runtime / item['profile'], item['destination']), item['mode'])
    for item in layout['profiles']:
        profile = root / 'robot/profiles' / item['profile']
        copy_file(child(root, item['source']), child(profile, item['destination']), item['mode'])
    for profile in ('base', 'mission'):
        workspace_link(root / 'robot/ros2_ws', runtime / profile / 'ros2_ws')
    for profile in ('navigation', 'mapping'):
        workspace_link(root / 'robot/profiles' / profile / 'ros2_ws', runtime / profile / 'ros2_ws')
    for profile in ('base', 'navigation', 'mission', 'mapping'):
        workspace_link(root / 'deploy/k3/env', runtime / profile / 'env')
        (runtime / profile / 'state').mkdir(parents=True, exist_ok=True)
        (runtime / profile / 'runs').mkdir(parents=True, exist_ok=True)
    vision = runtime / 'vision'
    for item in layout['vision']:
        copy_file(child(root, item['source']), child(vision / 'services/rvclaw_api', item['destination']), item['mode'])
    for item in layout['contracts']:
        copy_file(child(root, item['source']), child(vision, item['destination']))
    for item in layout['algorithms']:
        copy_file(child(root, item['source']),
                  child(vision / 'robot/ros2_ws/src/rvclaw_algo', item['destination']), item['mode'])
    # The stable API resolves its meter reference store relative to its root.
    for source_name, destination_name in (
        ('configs/device/meter', 'configs/meter'),
        ('configs/device/vision', 'configs/vision'),
        ('configs/thresholds', 'configs/thresholds'),
    ):
        shutil.copytree(root / source_name, vision / destination_name, dirs_exist_ok=True)
    for template in (root / 'deploy/k3/environment').glob('*.env'):
        destination = runtime / 'environment' / template.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(template.read_text(encoding='utf-8').replace('@ROBOT_ROOT@', str(root)),
                               encoding='utf-8')
    print('Source paths prepared' if args.source_only else 'Runtime prepared; services remain stopped')


if __name__ == '__main__':
    main()
