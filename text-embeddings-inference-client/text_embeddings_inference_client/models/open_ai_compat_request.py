from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="OpenAICompatRequest")


@_attrs_define
class OpenAICompatRequest:
    """
    Attributes:
        input_ (Union[List[str], str]):
        model (Union[Unset, None, str]):  Example: null.
        user (Union[Unset, None, str]):  Example: null.
    """

    input_: Union[List[str], str]
    model: Union[Unset, None, str] = UNSET
    user: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_: Union[List[str], str]

        if isinstance(self.input_, list):
            input_ = self.input_

        else:
            input_ = self.input_

        model = self.model
        user = self.user

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
            }
        )
        if model is not UNSET:
            field_dict["model"] = model
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_input_(data: object) -> Union[List[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                componentsschemas_input_type_1 = cast(List[str], data)

                return componentsschemas_input_type_1
            except:  # noqa: E722
                pass
            return cast(Union[List[str], str], data)

        input_ = _parse_input_(d.pop("input"))

        model = d.pop("model", UNSET)

        user = d.pop("user", UNSET)

        open_ai_compat_request = cls(
            input_=input_,
            model=model,
            user=user,
        )

        open_ai_compat_request.additional_properties = d
        return open_ai_compat_request

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
