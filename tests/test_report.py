from report import to_console
from schema import Result


def test_missing_dialogue_has_clear_user_guidance():
    output = to_console(Result(found=False), "a dialogue that is not present")

    assert "RESULT: Target dialogue is not in the video" in output
    assert "Please give the correct dialogue." in output


def test_audio_only_result_keeps_its_more_specific_explanation():
    output = to_console(Result(found=False, modality="audio_only"), "hello")

    assert "spoken in audio, not found as on-screen text" in output
    assert "Please give the correct dialogue." not in output
