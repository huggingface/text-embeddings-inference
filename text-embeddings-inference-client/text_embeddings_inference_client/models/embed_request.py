from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EmbedRequest")


@_attrs_define
class EmbedRequest:
    """
    Attributes:
        inputs (Union[List[str], str]):
        truncate (Union[Unset, bool]):  Default: True. Example: false.
    """

    inputs: Union[List[str], str]
    truncate: Union[Unset, bool] = True
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        inputs: Union[List[str], str]

        if isinstance(self.inputs, list):
            inputs = self.inputs

        else:
            inputs = self.inputs

        truncate = self.truncate

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "inputs": inputs,
            }
        )
        if truncate is not UNSET:
            field_dict["truncate"] = truncate

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_inputs(data: object) -> Union[List[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                componentsschemas_input_type_1 = cast(List[str], data)

                return componentsschemas_input_type_1
            except:  # noqa: E722
                pass
            return cast(Union[List[str], str], data)

        inputs = _parse_inputs(d.pop("inputs"))

        truncate = d.pop("truncate", UNSET)

        embed_request = cls(
            inputs=inputs,
            truncate=truncate,
        )

        embed_request.additional_properties = d
        return embed_request

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
