#!/bin/sh

# Change to the feature_pipeline directory
cd /usr/src/app/pipeline/feature_pipeline

if [ "$DEBUG" = true ]
then
    python -m bytewax.run "tools.run_real_time:build_flow(debug=True)"
else
    if [ "$BYTEWAX_PYTHON_FILE_PATH" = "" ]
    then
        echo 'BYTEWAX_PYTHON_FILE_PATH is not set. Exiting...'
        exit 1
    fi
    RUST_BACKTRACE=full python -m bytewax.run $BYTEWAX_PYTHON_FILE_PATH
fi

echo 'Process ended.'

if [ "$BYTEWAX_KEEP_CONTAINER_ALIVE" = true ]
then
    echo 'Keeping container alive...';
    while :; do sleep 1; done
fi