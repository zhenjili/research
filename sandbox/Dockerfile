# Sandbox Docker Image for Safe Code Execution
# Build: docker build -t code-sandbox:latest sandbox/
# Run: docker run --rm --network none code-sandbox:latest python -c "print('hello')"

FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r sandbox && useradd -r -g sandbox sandbox

# Install common Python packages (for test cases)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    sympy \
    && rm -rf /root/.cache/pip

# Security configurations
# - No network access (controlled by Docker run --network none)
# - Read-only filesystem where possible
# - Resource limits set at runtime

# Set working directory
WORKDIR /code

# Switch to non-root user
USER sandbox

# Default command
CMD ["python"]
