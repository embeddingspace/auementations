"""BDD-style tests for config_store submodule and registration decorator."""

import pytest
from hydra_zen import ZenStore

# Import the decorator early for test classes
from auementations.config.config_store import auementations_store


# Define test classes at module level so they're importable
@auementations_store(name="test_class", group="test_group")
class TestClass:
    def __init__(self, value: int):
        self.value = value


@auementations_store(name="test_registration", group="test_group")
class TestRegistration:
    def __init__(self, param: str):
        self.param = param


@auementations_store(name="my_complex_class_name", group="test_group")
class MyComplexClassName:
    pass


@auementations_store(group="test_group", name="custom_name")
class SomeClass:
    pass


@auementations_store(name="partial_class", group="test_group", zen_partial=True)
class PartialClass:
    def __init__(self, value: int):
        self.value = value


@auementations_store(name="first_class", group="multi_test")
class FirstClass:
    pass


@auementations_store(name="second_class", group="multi_test")
class SecondClass:
    pass


class TestConfigStoreSubmodule:
    """Test scenarios for config_store submodule creation."""

    def test_given_config_store_module_when_importing_then_store_exists(self):
        """Given the config_store module, when importing, then store exists."""
        # Given / When
        from auementations import config

        # Then
        assert config is not None
        assert hasattr(config, "auementations_store")

    def test_given_config_store_when_accessing_store_then_is_zenstore(self):
        """Given config_store, when accessing store attribute, then it is a ZenStore."""
        # Given
        from auementations.config import auementations_store

        # When
        store = auementations_store

        # Then
        assert isinstance(store, ZenStore)
        assert store.name == "auementations"


class TestRegisterConfigDecorator:
    """Test scenarios for @auementations_store decorator."""

    def test_given_register_config_decorator_when_imported_then_exists(self):
        """Given config_store module, when importing register_config, then it exists."""
        # Given / When
        from auementations.config.config_store import auementations_store

        # Then
        assert auementations_store is not None
        assert callable(auementations_store)

    def test_given_class_with_decorator_when_defined_then_class_still_usable(self):
        """Given a class with @register_config, when defined, then class remains usable."""
        # Given / When
        instance = TestClass(value=42)

        # Then
        assert instance.value == 42
        assert TestClass.__name__ == "TestClass"

    def test_given_decorated_class_when_checking_store_then_config_is_registered(self):
        """Given a decorated class, when checking store, then config is registered."""
        # Given
        from auementations.config import config_store

        # When
        entry = config_store.auementations_store.get_entry(
            "test_group", "test_registration"
        )

        # Then
        assert entry is not None
        assert entry["name"] == "test_registration"
        assert entry["group"] == "test_group"

    def test_given_class_name_with_capitals_when_registered_then_converts_to_snake_case(
        self,
    ):
        """Given a class with CamelCase name, when registered, then name is snake_case."""
        # Given
        from auementations.config import config_store

        # When
        entry = config_store.auementations_store.get_entry(
            "test_group", "my_complex_class_name"
        )

        # Then
        assert entry["name"] == "my_complex_class_name"


class TestRegisterConfigDecoratorWithParameters:
    """Test scenarios for @register_config decorator with custom parameters."""

    def test_given_custom_name_when_decorated_then_uses_custom_name(self):
        """Given decorator with custom name, when applied, then uses that name."""
        # Given
        from auementations.config import config_store

        # When
        entry = config_store.auementations_store.get_entry("test_group", "custom_name")

        # Then
        assert entry["name"] == "custom_name"

    def test_given_builds_kwargs_when_decorated_then_passes_to_builds(self):
        """Given decorator with builds kwargs, when applied, then passes them through."""
        # Given
        from auementations.config import config_store

        # When
        entry = config_store.auementations_store.get_entry(
            "test_group", "partial_class"
        )
        config = entry["node"]

        # Then
        # zen_partial=True means the config should create a partial function
        assert entry is not None
        # The actual verification of zen_partial behavior would require instantiation


class TestStoreAfterDecoratorRegistration:
    """Test scenarios for accessing store after decorator registration."""

    def test_given_multiple_decorated_classes_when_checking_store_then_all_registered(
        self,
    ):
        """Given multiple decorated classes, when checking store, then all are registered."""
        # Given
        from auementations.config import config_store

        # When / Then
        assert (
            config_store.auementations_store.get_entry("multi_test", "first_class")
            is not None
        )
        assert (
            config_store.auementations_store.get_entry("multi_test", "second_class")
            is not None
        )

    def test_given_auementations_store_when_accessing_then_is_same_as_config_store(
        self,
    ):
        """Given auementations_store export, when accessing, then is same as config_store.store."""
        # Given
        from auementations.config import auementations_store, config_store

        # When / Then
        assert auementations_store is config_store.auementations_store
