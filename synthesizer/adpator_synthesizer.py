from .base_synthesizer import BaseSynthesizer
from utils.file_util import *


class AdaptorSynthesizer(BaseSynthesizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
