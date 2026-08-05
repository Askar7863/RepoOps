"""
llm_client.py

Azure OpenAI Client Factory.

Responsibility
--------------
Build an AzureOpenAI client from configuration.

This module DOES NOT:
    ✗ Make any chat completion calls itself
    ✗ Build prompts
"""

from .. import config


def get_client():
    """Return an LLM client. Swap this out if you use a different provider."""
    from openai import AzureOpenAI

    if not (config.AZURE_ENDPOINT and config.AZURE_API_KEY):
        raise RuntimeError(
            "Azure OpenAI credentials are missing. Copy .env.example to .env "
            "and fill in your values."
        )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )
