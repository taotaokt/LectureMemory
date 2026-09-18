"""Domain-specific persistence errors shared by repositories."""


class CourseNotFoundError(ValueError):
    """Raised when an operation references a course that does not exist."""


class LectureNotFoundError(ValueError):
    """Raised when an operation references a lecture that does not exist."""


class SlidePageNotFoundError(ValueError):
    """Raised when an operation references a slide page that does not exist."""


class NotePageMismatchError(ValueError):
    """Raised when a note references a page from a different lecture."""


class DuplicateSlidePageError(ValueError):
    """Raised when a lecture already has the requested page number."""
