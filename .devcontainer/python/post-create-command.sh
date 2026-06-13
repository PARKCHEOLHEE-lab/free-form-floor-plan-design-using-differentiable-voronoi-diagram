#!/bin/bash

# The pre-commit config now lives under the python package dir, so the installed
# git hook must be told where to find it (-c bakes the path into .git/hooks).
pre-commit install -c free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/.pre-commit-config.yaml
