from typing import Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.error_type import ErrorType

T = TypeVar("T", bound="OpenAICompatErrorResponse")


@_attrs_define
class OpenAICompatErrorResponse:
    """
    Attributes:
        code (int):
        error_type (ErrorType):
        message (str):
    """

    code: int
    error_type: ErrorType
    message: str
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        code = self.code
        error_type = self.error_type.value

        message = self.message

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
                "error_type": error_type,
                "message": message,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        code = d.pop("code")

        error_type = ErrorType(d.pop("error_type"))

        message = d.pop("message")

        open_ai_compat_error_response = cls(
            code=code,
            error_type=error_type,
            message=message,
        )

        open_ai_compat_error_response.additional_properties = d
        return open_ai_compat_error_response

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
