"""Single-frame meter diagnostics and live state; independent of the web handler."""
import json
import time
import uuid


class TrialError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def runtime_reading(status):
    camera = (status.get('cameras') or [{}])[0]
    inference = camera.get('inference') or {}
    mode = str(status.get('active_mode', ''))
    meter_mode = mode.startswith('astra_meter')
    digital = meter_mode and inference.get('meter_type', 'digital') == 'digital'
    running = meter_mode and bool(camera.get('active'))
    online = running and bool(camera.get('online'))
    switching = bool(status.get('switching'))
    ready = online and bool(inference.get('ready')) and not switching
    state = ('starting' if switching and meter_mode else
             'inactive' if not running else
             'offline' if not online else
             'starting' if not ready else inference.get('meter_status') or 'checking')
    region = inference.get('display_roi')
    bbox = dict(zip(('x', 'y', 'width', 'height'), region)) if isinstance(region, list) and len(region) == 4 else None
    return {
        'camera': 'astra', 'mode': status.get('active_mode'),
        'running': running, 'online': online, 'switching': switching, 'ready': ready,
        'meter_id': inference.get('meter_id') if digital else None,
        'meter_name': inference.get('meter_name') if digital else None,
        'meter_type': inference.get('meter_type') if meter_mode else None,
        'status': state, 'value': inference.get('meter_value') if ready else None,
        'confidence': inference.get('meter_confidence') if ready else None,
        'raw_text': inference.get('meter_raw_text', '') if ready else '',
        'unit': inference.get('unit'), 'inference_ms': inference.get('inference_ms'),
        'frame_age_ms': camera.get('frame_age_ms'),
        'error': inference.get('error') or camera.get('last_error'),
        'angle_deg': inference.get('meter_angle_deg') if ready else None,
        'text_items': inference.get('meter_text_items', []) if ready else [],
        'display_value': ({'text': inference.get('meter_raw_text', '') if ready else '',
                          'value': inference.get('meter_value') if ready else None,
                          'confidence': inference.get('meter_confidence') if ready else None,
                          'method': inference.get('meter_value_method') or None,
                          'bbox': bbox} if digital else None),
    }


def require_meter(status, meter_id):
    current = runtime_reading(status)
    if current['switching']:
        raise TrialError('相机正在切换，请稍后重试')
    if current['mode'] != 'astra_meter_digital' or current['meter_id'] != meter_id or not current['running']:
        raise TrialError('请先启动所选仪表的数字检测')
    if not current['ready']:
        raise TrialError('相机离线或模型尚未就绪，请稍后重试', 503)


def run_trial(controller, configuration, root, timeout=15):
    """Caller holds the shared mutation lock throughout this bounded request."""
    meter_id = configuration['meter_id']
    require_meter(controller.status(), meter_id)
    root.mkdir(parents=True, exist_ok=True)
    trial_id = uuid.uuid4().hex
    requested_at = time.time()
    started = time.monotonic()
    request = root / 'request'
    temporary = root / 'request.tmp'
    temporary.write_text(f'{trial_id} {requested_at:.9f}\n', encoding='ascii')
    temporary.replace(request)
    native_result = root / f'{trial_id}.json'
    try:
        while time.monotonic() - started < timeout:
            require_meter(controller.status(), meter_id)
            if native_result.is_file():
                trial = json.loads(native_result.read_text(encoding='utf-8'))
                if trial.get('trial_id') != trial_id or trial.get('captured_at', 0) < requested_at:
                    raise TrialError('试读返回了过期帧，请重新拍照', 502)
                if not all((root / f'{trial_id}-{name}').is_file() for name in ('scene.jpg', 'crop.png')):
                    raise TrialError('试读照片保存不完整', 502)
                trial.update({
                    'meter_id': meter_id, 'meter_name': configuration['name'],
                    'configuration_updated_at': configuration.get('updated_at'),
                    'unit': configuration.get('unit') or '',
                    'total_ms': round((time.monotonic() - started) * 1000, 1),
                    'image_url': f'/api/meter-trials/{trial_id}/scene.jpg',
                    'crop_url': f'/api/meter-trials/{trial_id}/crop.png',
                })
                (root / f'{trial_id}-result.json').write_text(
                    json.dumps(trial, ensure_ascii=False), encoding='utf-8')
                return trial
            time.sleep(0.05)
        raise TrialError('等待本次识别超时，请检查相机后重试', 504)
    finally:
        request.unlink(missing_ok=True)
