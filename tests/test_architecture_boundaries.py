from scripts.check_architecture import FEATURES_DIR, check_architecture


def test_application_features_respect_architecture_boundaries() -> None:
    violations = check_architecture(FEATURES_DIR)

    assert not violations, "\n".join(violation.format() for violation in violations)
