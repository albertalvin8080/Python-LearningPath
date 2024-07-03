# from unittest.mock import patch
import unittest.mock as mock
from source.registry import Registry
import pytest, requests

# pytest tests/test_registry.py


@mock.patch("requests.get")
def test_get_users_success(mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    # You should not use `mock_response.json = ` direcly because json() is a method.
    mock_response.json.return_value = {"id": 1, "name": "Emil Sebe"}

    mock_get.return_value = mock_response

    assert Registry().get_users() == {"id": 1, "name": "Emil Sebe"}


@pytest.fixture
def registry_instance():
    return Registry()


@mock.patch("requests.get")
def test_get_users_HTTPError(mock_get, registry_instance):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    # mock_response.status_code = 200 # Uncomment to make it fail.

    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        registry_instance.get_users()
