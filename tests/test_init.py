"""Basic unittests to test functioning of module's top-level"""

try:
    from fastcan import *  # noqa: F403

    _TOP_IMPORT_ERROR = None

except Exception as e:
    _TOP_IMPORT_ERROR = e


def test_import_fastcan():
    """Test either above import has failed for some reason
    "import *" is discouraged outside of the module level, hence we
    rely on setting up the variable above
    """
    assert _TOP_IMPORT_ERROR is None
