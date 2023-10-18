from enum import Enum


class ErrorType(str, Enum):
    BACKEND = "Backend"
    OVERLOADED = "Overloaded"
    TOKENIZER = "Tokenizer"
    UNHEALTHY = "Unhealthy"
    VALIDATION = "Validation"

    def __str__(self) -> str:
        return str(self.value)
