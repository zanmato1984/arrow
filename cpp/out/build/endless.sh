#!/bin/bash

# Replace 'your_program' with the command you want to run endlessly.
while /Users/zanmato/dev/arrow/cpp/out/build/ninja-debug/debug/arrow-acero-asof-join-node-test --gtest_filter="AsofJoinTest.Flaky"; do
    echo "Program exited with status $?. Restarting..."
    sleep 0.5  # Delay to prevent system overload, can be adjusted or removed as needed.
done

echo "Program terminated with a non-zero exit status."
