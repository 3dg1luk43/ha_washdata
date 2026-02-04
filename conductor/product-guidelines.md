# Product Guidelines

## Architectural Style
- **Modular Monolith:** The project is organized by domain features within a single package structure, following Home Assistant integration best practices.

## Error Handling
- **Fail-Fast with Logging:** We prioritize immediate error detection with detailed logging to assist both users and developers in troubleshooting.

## Code Quality & Maintainability
- **Self-Documenting Code:** We prioritize clear naming and logical structure over excessive commenting to ensure the code remains readable and maintainable.
- **Unit Testing Focus:** We maintain high test coverage for core logic and signal processing to ensure reliability and prevent regressions.

## Data Management
- **JSON-based Storage:** We utilize Home Assistant's built-in storage helpers for persisting cycle history and user profiles, ensuring integration with the host system.

## Communication Style
- **Informative & Direct:** User-facing messages and logs provide clear and concise information without unnecessary technical jargon.
- **Transparent:** We are clear with the user about the level of uncertainty in cycle detection or time estimation to manage expectations.
