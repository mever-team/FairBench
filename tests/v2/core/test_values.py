import fairbench as fb
import random
import pytest


def test_number():
    for _ in range(10):
        value = random.random()
        units = f"unit {random.randint(0, 6)}"
        number = fb.core.Number(value, units)
        number = fb.core.Number.from_dict(number.to_dict())
        assert float(number) == value
        assert number.units == units


def test_targeted_number():
    for _ in range(10):
        value = random.random()
        target = random.random()
        units = f"unit {random.randint(0, 6)}"
        number = fb.core.TargetedNumber(value, target, units)
        number = fb.core.Number.from_dict(number.to_dict())
        assert float(number) == value
        assert number.target == target
        assert number.units == units


def test_str():
    for _ in range(10):
        name = f"name {random.randint(0, 6)}"
        role = f"role {random.randint(0, 6)}"
        original = fb.core.Descriptor(name, role)
        descriptor = fb.core.Descriptor.from_dict(original.to_dict())
        assert original == descriptor
        assert name in str(descriptor)
        assert role in str(descriptor)
        assert name in descriptor.__repr__()
        assert role in descriptor.__repr__()
        assert name in descriptor.details
        assert role in descriptor.details
        assert descriptor.preferred_units is not None
        assert descriptor.prototype is not None
        assert descriptor.alias is not None


def test_conversion():
    for _ in range(10):
        name = f"name {random.randint(0, 6)}"
        role = f"role {random.randint(0, 6)}"
        descriptor = fb.core.Descriptor(name, role)
        original = descriptor(0.5)
        value = fb.core.Value.from_dict(original.to_dict())
        assert value == original
        assert value.descriptor == descriptor
        assert value.units == descriptor.preferred_units
        assert len(value.depends) == 0

        # check that if we add something to the descriptor it can be flattened
        descriptor_child = fb.core.Descriptor(
            f"name {random.randint(0, 6)}", f"role {random.randint(0, 6)}"
        )
        value = descriptor(0.5, depends=[descriptor_child(0.3)])
        value = fb.core.Value.from_dict(value.to_dict())
        assert len(value.depends) == 1
        assert value.details.flatten(True)[0] == 0.3


def test_flatten():
    for _ in range(10):
        name = f"name {random.randint(0, 6)}"
        role = f"role {random.randint(0, 6)}"
        descriptor = fb.core.Descriptor(name, role)
        value = descriptor(0.5)

        # cannot flatten if we have a value already
        with pytest.raises(Exception):
            _ = value.flatten(True)

        # check that we can flatten by skipping an intermediate child
        value = descriptor(
            depends=[
                fb.core.Descriptor(role=f"subrole1", name=f"role {name}")(
                    random.random()
                )
                for name in range(10)
            ]
            + [
                fb.core.Descriptor(role=f"subrole2", name=f"role {10+name}")(
                    random.random()
                )
                for name in range(5)
            ]
        )
        assert value.exists()
        assert not descriptor().exists()
        assert len(value.flatten(True)) == 15
        assert len(list(value.values("subrole1"))) == 10
        assert "subrole1" in str(value)
        assert "subrole2" in str(value)


def test_conflicting_dependencies():
    for _ in range(10):
        name = f"name {random.randint(0, 6)}"
        role = f"role {random.randint(0, 6)}"
        descriptor = fb.core.Descriptor(name, role)
        descriptor_child = fb.core.Descriptor(
            f"name {random.randint(0, 6)}", f"role {random.randint(0, 6)}"
        )

        # no error for identical conflicts
        _ = descriptor(0.5, depends=[descriptor_child(0.3), descriptor_child(0.3)])

        with pytest.raises(Exception) as e_info:
            _ = descriptor(0.5, depends=[descriptor_child(0.3), descriptor_child(0.5)])
        assert descriptor.name in str(e_info.value)
        assert descriptor.role in str(e_info.value)
        assert descriptor_child.name in str(e_info.value)
        assert descriptor_child.role in str(e_info.value)
        assert descriptor_child.alias in str(e_info.value)
