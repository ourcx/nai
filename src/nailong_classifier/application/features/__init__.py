"""可插拔的应用功能协议。"""

from .admin import AdminFeature, AdminRequest
from .base import FeatureHandler, Workload
from .callback import CallbackFeature
from .help import HelpFeature
from .social import SocialImportFeature
from .speech import SpeechFeature, SpeechRequest
from .system import InitializationFeature

__all__ = [
    "FeatureHandler",
    "AdminFeature",
    "AdminRequest",
    "CallbackFeature",
    "HelpFeature",
    "InitializationFeature",
    "SocialImportFeature",
    "SpeechFeature",
    "SpeechRequest",
    "Workload",
]
