"""Minimal lightbox plugin for MkDocs Material."""

from __future__ import annotations

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin


class GlightboxPlugin(BasePlugin):
    """Attach glightbox assets to every page."""

    css_href = "https://cdn.jsdelivr.net/npm/glightbox/dist/css/glightbox.min.css"
    js_href = "https://cdn.jsdelivr.net/npm/glightbox/dist/js/glightbox.min.js"
    init_script = "js/glightbox-init.js"

    def on_config(self, config: MkDocsConfig) -> MkDocsConfig:
        extra_css = list(config.extra_css or [])
        if self.css_href not in extra_css:
            extra_css.append(self.css_href)
        config.extra_css = extra_css

        extra_js = list(config.extra_javascript or [])
        if self.js_href not in extra_js:
            extra_js.append(self.js_href)
        if self.init_script not in extra_js:
            extra_js.append(self.init_script)
        config.extra_javascript = extra_js
        return config
