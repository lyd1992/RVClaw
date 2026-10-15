"""Read a current pointer against registered, configured static scale geometry."""
import base64
from collections import OrderedDict
import hashlib
import json
import math
import os
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np
from .meter_reference import ReferenceRegistrationError
from .pointer_gauge_adapter import _INFERENCE_DEADLINE

_REFERENCE_FEATURES = OrderedDict()

def _pose_current(image, *, enhance=False):
    if enhance:
        image = _enhance(image)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ReferenceRegistrationError('pointer image cannot be encoded')
    configured_url = os.getenv('RVCLAW_POINTER_GAUGE_URL', 'http://127.0.0.1:8018/api/read')
    url = configured_url.rsplit('/', 1)[0] + '/pointer-pose'
    body = json.dumps({'image_b64':base64.b64encode(encoded).decode('ascii')}).encode()
    remaining = _remaining()
    with urlopen(Request(url,data=body,headers={'Content-Type':'application/json'}),
                 timeout=min(5.,remaining)) as response:
        result = json.loads(response.read())
    _remaining()
    if result.get('status') != 'pointer_geometry' or result.get('n_meters') != 1:
        if not enhance and int(result.get('n_meters') or 0) <= 1:
            return _pose_current(image, enhance=True)
        raise ReferenceRegistrationError('current pointer geometry unavailable')
    if result.get('image_size') != [image.shape[1],image.shape[0]]:
        raise ReferenceRegistrationError('pointer frame size mismatch')
    scores = [float(result['score']), *map(float,result['keypoint_scores'])]
    if len(scores) != 3 or not all(math.isfinite(v) for v in scores) or scores[0]<.8 or min(scores[1:])<.9:
        raise ReferenceRegistrationError('current pointer confidence insufficient')
    x,y,w,h = map(float,result['meter_box']['rect'])
    score = float(result['meter_box']['score'])
    if (not all(math.isfinite(v) for v in (x,y,w,h,score)) or not .65<=score<=1
            or not 0<=x<x+w<=image.shape[1] or not 0<=y<y+h<=image.shape[0]):
        raise ReferenceRegistrationError('invalid current dial locator')
    return result


def _enhance(image):
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)


def _remaining():
    deadline = _INFERENCE_DEADLINE.get()
    remaining = 8. if deadline is None else deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('pointer registration budget exhausted')
    return remaining


def register_static_scale(reference, live, geometry, live_box=None, *, method='sift'):
    """Require corroborated homographies after erasing reference pointer pixels."""
    is_orb = method.startswith('orb')
    registration_deadline = time.monotonic()+(3.0 if is_orb else 2.0)
    rh,rw = reference.shape[:2]
    detector = (cv2.ORB_create(nfeatures=5000,edgeThreshold=12,fastThreshold=5)
                if is_orb else cv2.SIFT_create(nfeatures=5000,contrastThreshold=.01))
    feature_image = live
    left,top = 0,0
    if live_box is not None:
        # Match the instrument and its surrounding printed body, rather than
        # unrelated signs and wall texture elsewhere in the camera frame.
        x,y,w,h = map(float,live_box)
        pad = .65*max(w,h)
        left,top = max(0,math.floor(x-pad)),max(0,math.floor(y-pad))
        right,bottom = min(live.shape[1],math.ceil(x+w+pad)),min(live.shape[0],math.ceil(y+h+pad))
        feature_image = live[top:bottom,left:right]
    clahe = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
    live_gray = cv2.cvtColor(feature_image,cv2.COLOR_BGR2GRAY)
    if method=='orb':
        live_gray = clahe.apply(live_gray)
    lk,ld = detector.detectAndCompute(live_gray,None)
    if left or top:
        for point in lk:
            point.pt = (point.pt[0]+left,point.pt[1]+top)
    if ld is None:
        raise ReferenceRegistrationError('current instrument features unavailable')
    source_points = np.float32([[geometry[k]['x']*rw,geometry[k]['y']*rh]
                                for k in ('center','start','end')])
    source_box = geometry['dial_region']
    box = np.float32([[source_box['x']*rw,source_box['y']*rh],
                     [(source_box['x']+source_box['width'])*rw,source_box['y']*rh],
                     [(source_box['x']+source_box['width'])*rw,(source_box['y']+source_box['height'])*rh],
                     [source_box['x']*rw,(source_box['y']+source_box['height'])*rh]])
    accepted = []
    mutable = geometry['mutable_region']
    clean = reference.copy()
    cv2.ellipse(clean,(round(mutable['cx']*rw),round(mutable['cy']*rh)),
                (round(mutable['rx']*rw),round(mutable['ry']*rh)),0,0,360,(128,128,128),-1)
    reference_key = (method,hashlib.sha256(clean.tobytes()).digest(),rh,rw,
                     tuple(float(mutable[k]) for k in ('cx','cy','rx','ry')))
    factors = [1/3,1/4,5/12,1/2,2/3,3/4,1/5,1.]
    if live_box is not None:
        # Start near the observed dial scale. A fixed sequence can exhaust the
        # budget on incompatible scales when the stop distance changes.
        _,_,bw,bh = map(float,live_box)
        observed_scale = .5*(bw/(source_box['width']*rw)+bh/(source_box['height']*rh))
        factors = [observed_scale*v for v in (1.,.85,1.15)] + factors
    tried_widths = set()
    for factor in factors:
        _remaining()
        if time.monotonic() >= registration_deadline:
            break
        width = max(240,16*round(rw*factor/16)); scale = width/rw
        if width in tried_widths:
            continue
        tried_widths.add(width)
        key = reference_key + (width,)
        cached = _REFERENCE_FEATURES.get(key)
        if cached is None:
            ref = cv2.resize(clean,(width,round(rh*scale)),interpolation=cv2.INTER_AREA)
            h,w = ref.shape[:2]
            mask = np.full((h,w),255,np.uint8)
            cv2.ellipse(mask,(round(mutable['cx']*w),round(mutable['cy']*h)),
                        (round(mutable['rx']*w),round(mutable['ry']*h)),0,0,360,0,-1)
            # Cache static descriptors only, after erasing mutable pixels.
            mask = cv2.erode(mask,np.ones((7,7),np.uint8))
            ref[mask==0] = 128
            gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
            if method=='orb':
                gray = clahe.apply(gray)
            cached = detector.detectAndCompute(gray,mask)
            _REFERENCE_FEATURES[key] = cached
            while len(_REFERENCE_FEATURES)>48:
                _REFERENCE_FEATURES.popitem(last=False)
        _REFERENCE_FEATURES.move_to_end(key)
        rk,rd = cached
        if rd is None:continue
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING if is_orb else cv2.NORM_L2).knnMatch(rd,ld,k=2)
        ratio = .77 if method=='orb' else (.75 if is_orb else .8)
        good = [p[0] for p in pairs if len(p)==2 and p[0].distance<ratio*p[1].distance]
        unique = {}
        for match in sorted(good,key=lambda m:m.distance):
            unique.setdefault(tuple(round(v) for v in lk[match.trainIdx].pt),match)
        good = list(unique.values())
        if len(good)<10:continue
        src = np.float32([rk[m.queryIdx].pt for m in good])/scale
        dst = np.float32([lk[m.trainIdx].pt for m in good])
        H, keep = cv2.findHomography(src,dst,cv2.RANSAC,1.5)
        if H is None or not np.isfinite(H).all():continue
        use = keep.ravel().astype(bool)
        coverage = cv2.contourArea(cv2.convexHull(src[use]))/(rw*rh)
        if int(use.sum())<10 or coverage<.15 or use.mean()<.2:continue
        mapped = cv2.perspectiveTransform(source_points[None],H)[0]
        quad = cv2.perspectiveTransform(box[None],H)[0]
        if (not np.isfinite(mapped).all() or not np.isfinite(quad).all()
                or not cv2.isContourConvex(quad) or cv2.contourArea(quad,oriented=True)<=0
                or np.any(quad[:,0]<0) or np.any(quad[:,0]>=live.shape[1])
                or np.any(quad[:,1]<0) or np.any(quad[:,1]>=live.shape[0])):continue
        diameter = float(np.mean(np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)))
        if diameter<80:continue
        item = dict(H=H,points=mapped,quad=quad,diameter=diameter,
                    matches=len(good),inliers=int(use.sum()),coverage=coverage,method=method)
        # Corroborate the complete calibrated geometry at another source scale.
        for previous in accepted:
            error = float(np.max(np.linalg.norm(previous['points']-mapped,axis=1)))
            if error <= .02*min(diameter,previous['diameter']):
                return previous,item
        accepted.append(item)
    raise ReferenceRegistrationError('two static scale registrations did not agree')


def value_from_pointer(pose, registration, geometry, calibration):
    H = registration['H']
    center,tip = np.float32([pose['center'],pose['tip']])
    diameter = registration['diameter']
    if not np.isfinite([center,tip]).all():
        raise ReferenceRegistrationError('nonfinite current pointer')
    if np.linalg.norm(center-registration['points'][0])>.04*diameter:
        raise ReferenceRegistrationError('pointer center disagrees with static scale')
    quad = registration.get('quad')
    if quad is not None and (cv2.pointPolygonTest(quad,tuple(map(float,center)),False)<0
                             or cv2.pointPolygonTest(quad,tuple(map(float,tip)),False)<0):
        raise ReferenceRegistrationError('current pointer outside registered dial')
    length = float(np.linalg.norm(tip-center))
    if not .2*diameter <= length <= .5*diameter:
        raise ReferenceRegistrationError('pointer length outside dial geometry')
    mapped = cv2.perspectiveTransform(np.float32([[center,tip]]),np.linalg.inv(H))[0]
    # Scale endpoints are transformed back into the reference dial plane;
    # measurement comes solely from the current center-to-tip vector.
    endpoints = cv2.perspectiveTransform(registration['points'][None],np.linalg.inv(H))[0]
    def angle(point):
        v = point-mapped[0]
        return math.atan2(float(v[1]),float(v[0]))%(2*math.pi)
    start,end,needle = angle(endpoints[1]),angle(endpoints[2]),angle(mapped[1])
    sign = 1 if geometry.get('direction','clockwise')=='clockwise' else -1
    sweep = (sign*(end-start))%(2*math.pi)
    travelled = (sign*(needle-start))%(2*math.pi)
    if not math.pi/2<sweep<2*math.pi-.1 or travelled>sweep:
        raise ReferenceRegistrationError('pointer outside configured scale')
    lower,upper = map(float,(calibration['min_value'],calibration['max_value']))
    if not (math.isfinite(lower) and math.isfinite(upper) and upper>lower):
        raise ReferenceRegistrationError('invalid pointer range')
    return lower+(upper-lower)*travelled/sweep


def read_registered_pointer(reference,live,record,calibration,pose_reader=None):
    geometry = {**record['pointer_scale'],'dial_region':record['display_region']}
    pose = (pose_reader or _pose_current)(live)
    try:
        try:
            pair = register_static_scale(reference,live,geometry,pose['meter_box']['rect'])
        except ReferenceRegistrationError:
            try:
                pair = register_static_scale(reference,live,geometry,pose['meter_box']['rect'],method='orb')
            except ReferenceRegistrationError:
                pair = register_static_scale(reference,live,geometry,pose['meter_box']['rect'],method='orb_plain')
        values = [value_from_pointer(pose,r,geometry,calibration) for r in pair]
    except (ReferenceRegistrationError,cv2.error,np.linalg.LinAlgError,TimeoutError) as error:
        x,y,w,h = map(float,pose['meter_box']['rect'])
        return dict(kind='pointer',status='unreadable',value=None,raw_text=None,
                    confidence=0.,gauge_box=[round(x),round(y),round(x+w),round(y+h)],
                    reason='registered pointer refused: '+str(error),
                    notes={'locator_only':True,'reference_pointer_used':False})
    span = float(calibration['max_value'])-float(calibration['min_value'])
    if abs(values[0]-values[1])>.02*span:
        x,y,w,h = map(float,pose['meter_box']['rect'])
        return dict(kind='pointer',status='unreadable',value=None,raw_text=None,
                    confidence=0.,gauge_box=[round(x),round(y),round(x+w),round(y+h)],
                    reason='independent scale registrations read different values',
                    notes={'locator_only':True,'reference_pointer_used':False})
    value = sum(values)/2
    _remaining()
    decimals = int(calibration.get('decimal_places',2))
    quad = pair[0]['quad']
    return dict(kind='pointer',status='stable',value=value,raw_text=f'{value:.{decimals}f}',
                confidence=min(float(pose['score']),float(pose['meter_box']['score'])),
                gauge_box=[round(float(quad[:,0].min())),round(float(quad[:,1].min())),
                           round(float(quad[:,0].max())),round(float(quad[:,1].max()))],
                unit=record.get('unit',''),reason='',
                model='NanoDet + YOLOv8s-Pose (current pointer, configured static scale)',
                value_method='registered_scale_current_pointer',
                notes=dict(current_pointer=pose,static_scale_checks=[
                    {key:r[key] for key in ('matches','inliers','coverage')}
                    for r in pair],reference_pointer_used=False))
