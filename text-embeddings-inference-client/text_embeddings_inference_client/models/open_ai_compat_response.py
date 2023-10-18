from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.open_ai_compat_embedding import OpenAICompatEmbedding
    from ..models.open_ai_compat_usage import OpenAICompatUsage


T = TypeVar("T", bound="OpenAICompatResponse")


@_attrs_define
class OpenAICompatResponse:
    """
    Attributes:
        data (List['OpenAICompatEmbedding']):
        model (str):  Example: thenlper/gte-base.
        object_ (str):  Example: list.
        usage (OpenAICompatUsage):
    """

    data: List["OpenAICompatEmbedding"]
    model: str
    object_: str
    usage: "OpenAICompatUsage"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = []
        for data_item_data in self.data:
            data_item = data_item_data.to_dict()

            data.append(data_item)

        model = self.model
        object_ = self.object_
        usage = self.usage.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
                "model": model,
                "object": object_,
                "usage": usage,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.open_ai_compat_embedding import OpenAICompatEmbedding
        from ..models.open_ai_compat_usage import OpenAICompatUsage

        d = src_dict.copy()
        data = []
        _data = d.pop("data")
        for data_item_data in _data:
            data_item = OpenAICompatEmbedding.from_dict(data_item_data)

            data.append(data_item)

        model = d.pop("model")

        object_ = d.pop("object")

        usage = OpenAICompatUsage.from_dict(d.pop("usage"))

        open_ai_compat_response = cls(
            data=data,
            model=model,
            object_=object_,
            usage=usage,
        )

        open_ai_compat_response.additional_properties = d
        return open_ai_compat_response

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
