class SBDError(RuntimeError):
    """Actionable application error, safe to display locally."""


class VideoError(SBDError):
    pass


class ModelError(SBDError):
    pass
