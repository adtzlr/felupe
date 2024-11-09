def test_import():
    import felupe as fem

    assert hasattr(fem, "constitution")


if __name__ == "__main__":
    test_import()
