# -*- coding: utf-8 -*-
"""
FaceBlur Web Application Package
"""

from __future__ import absolute_import, unicode_literals

# Celery app import
from .celery import app as celery_app

__all__ = ('celery_app',)
