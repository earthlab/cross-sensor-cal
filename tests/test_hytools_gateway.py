from src.third_party.hytools_api import import_hytools, HyToolsInfo


def test_hytools_import_gateway():
    module, info = import_hytools()
    assert module is not None
    assert isinstance(info, HyToolsInfo)
    assert isinstance(info.version, str)
