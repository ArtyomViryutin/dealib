#!/bin/bash
set -ex

declare -A output=(
    [requirements.in]=requirements.txt
    [requirements.dev.in]=requirements.dev.txt
)

for input in "${!output[@]}" ; do
    for req in ${input}; do
        sort -b "${req}" -o "${req}"
    done
    pip-compile --quiet "${@}" --output-file "${output[${input}]}" ${input}
done


pip-sync -v requirements.txt requirements.dev.txt
