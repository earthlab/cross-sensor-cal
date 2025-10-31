"""No-op minify plugin to satisfy MkDocs config."""

from __future__ import annotations

from mkdocs.config import config_options
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin


class MinifyPlugin(BasePlugin):
    """Return config untouched; real minification runs in production."""

    config_scheme = (
        ("minify_html", config_options.Type(bool, default=True)),
    )

    def on_config(self, config: MkDocsConfig) -> MkDocsConfig:
        return config

