def test_configuration():
    from core import Configuration

    @Configuration.register_module(schema_name="test")
    def tst():
        return { "name": "test" }

    mod = Configuration.build_module("test")

    assert isinstance(mod, dict)
    assert "name" in mod.keys()
    assert mod["name"] == "test"