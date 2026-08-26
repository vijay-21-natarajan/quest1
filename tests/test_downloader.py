import pytest

from video.downloader import NetworkPermissionError, _raise_helpful_network_error


def test_windows_socket_permission_error_has_actionable_message():
    with pytest.raises(NetworkPermissionError, match="Windows blocked") as caught:
        _raise_helpful_network_error(RuntimeError("connection failed: [WinError 10013]"))

    assert "not a problem with the video URL" in str(caught.value)


def test_unrelated_error_is_preserved():
    original = RuntimeError("video unavailable")

    with pytest.raises(RuntimeError) as caught:
        _raise_helpful_network_error(original)

    assert caught.value is original
