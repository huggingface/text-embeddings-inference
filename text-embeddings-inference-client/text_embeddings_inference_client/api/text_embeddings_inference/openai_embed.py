from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.open_ai_compat_error_response import OpenAICompatErrorResponse
from ...models.open_ai_compat_request import OpenAICompatRequest
from ...models.open_ai_compat_response import OpenAICompatResponse
from ...types import Response


def _get_kwargs(
    *,
    json_body: OpenAICompatRequest,
) -> Dict[str, Any]:
    pass

    json_json_body = json_body.to_dict()

    return {
        "method": "post",
        "url": "/openai",
        "json": json_json_body,
    }


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = OpenAICompatResponse.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        response_413 = OpenAICompatErrorResponse.from_dict(response.json())

        return response_413
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = OpenAICompatErrorResponse.from_dict(response.json())

        return response_422
    if response.status_code == HTTPStatus.FAILED_DEPENDENCY:
        response_424 = OpenAICompatErrorResponse.from_dict(response.json())

        return response_424
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        response_429 = OpenAICompatErrorResponse.from_dict(response.json())

        return response_429
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    json_body: OpenAICompatRequest,
) -> Response[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    """OpenAI compatible route

     OpenAI compatible route

    Args:
        json_body (OpenAICompatRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]
    """

    kwargs = _get_kwargs(
        json_body=json_body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    json_body: OpenAICompatRequest,
) -> Optional[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    """OpenAI compatible route

     OpenAI compatible route

    Args:
        json_body (OpenAICompatRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[OpenAICompatErrorResponse, OpenAICompatResponse]
    """

    return sync_detailed(
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    json_body: OpenAICompatRequest,
) -> Response[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    """OpenAI compatible route

     OpenAI compatible route

    Args:
        json_body (OpenAICompatRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]
    """

    kwargs = _get_kwargs(
        json_body=json_body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    json_body: OpenAICompatRequest,
) -> Optional[Union[OpenAICompatErrorResponse, OpenAICompatResponse]]:
    """OpenAI compatible route

     OpenAI compatible route

    Args:
        json_body (OpenAICompatRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[OpenAICompatErrorResponse, OpenAICompatResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            json_body=json_body,
        )
    ).parsed
