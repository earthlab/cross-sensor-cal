# API: qa_plots

`render_flightline_panel` now accepts sampling controls and returns both the PNG path
and a metrics dictionary. When `save_json=True` (default) it writes
`<prefix>_qa.json` alongside the PNG and the returned dict matches that payload.

::: cross_sensor_cal.qa_plots.render_flightline_panel
