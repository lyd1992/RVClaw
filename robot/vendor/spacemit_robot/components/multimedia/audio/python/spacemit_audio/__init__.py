# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
SpacemitAudio - 音频采集和播放库
"""

from ._spacemit_audio import AudioCapture, AudioPlayer, init, get_config

__all__ = ['AudioCapture', 'AudioPlayer', 'init', 'get_config']
__version__ = '1.0.0'
