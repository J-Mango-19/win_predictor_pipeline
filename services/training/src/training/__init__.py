def main() -> None:
    # Lazy import so `import training` stays cheap and circular-import free.
    from training.pipeline import main as _main

    _main()
