#! /bin/bash

main () {
    local -A pids=()
    local -A tasks=([task1]="./train.sh"
                    [task2]="./train.sh"
                    [task3]="./train.sh"
                    [task4]="./train.sh"
                    [task5]="./train.sh"
                    [task6]="./train.sh")
    local max_concurrent_tasks=6

    for key in "${!tasks[@]}"; do
        while [ $(jobs 2>&1 | grep -c Running) -ge "$max_concurrent_tasks" ]; do
            sleep 0.1
        done
        ${tasks[$key]} &
        pids+=(["$key"]="$!")
    done

    errors=0
    for key in "${!tasks[@]}"; do
        pid=${pids[$key]}
        local cur_ret=0
        if [ -z "$pid" ]; then
            echo "No Job ID known for the $key process" # should never happen
            cur_ret=1
        else
            wait $pid
            cur_ret=$?
        fi
        if [ "$cur_ret" -ne 0 ]; then
            errors=$(($errors + 1))
            echo "$key (${tasks[$key]}) failed."
        fi
    done

    return $errors
}

main