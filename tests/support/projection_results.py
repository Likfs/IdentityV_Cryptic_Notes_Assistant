from src import projection_workflow as workflow


def result_for(window, image, deferred=True):
    return workflow.process_screenshot(image, catalog=window.catalog, profile=window.profile, layout_override=window.locked_layout, opacity=0.3, registration_backend='template', defer_projection_preview=deferred)
