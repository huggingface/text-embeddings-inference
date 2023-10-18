from typing import Any, Dict, List, Type, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="OpenAICompatEmbedding")


@_attrs_define
class OpenAICompatEmbedding:
    """
    Attributes:
        embedding (List[float]):  Example: ['0.0', '1.0', '2.0'].
        index (int):  Example: 0.
        object_ (str):  Example: embedding.
    """

    embedding: List[float]
    index: int
    object_: str
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        embedding = self.embedding

        index = self.index
        object_ = self.object_

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "embedding": embedding,
                "index": index,
                "object": object_,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        embedding = cast(List[float], d.pop("embedding"))

        index = d.pop("index")

        object_ = d.pop("object")

        open_ai_compat_embedding = cls(
            embedding=embedding,
            index=index,
            object_=object_,
        )

        open_ai_compat_embedding.additional_properties = d
        return open_ai_compat_embedding

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
