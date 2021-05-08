#!/bin/bash

end_script() {
    exit 0
}

trap 'end_script' SIGINT

opponents=(baselineTeam myTeam1 myTeam2)
layouts=(
    alleyCapture
    bloxCapture
    crowdedCapture
    defaultCapture
    distantCapture
    fastCapture
    jumboCapture
    mediumCapture
    officeCapture
    strategicCapture
    tinyCapture
    RANDOM
)

while true;
do
    op=$[$RANDOM % ${#opponents[@]}]
    layout=$[$RANDOM % ${#layouts[@]}]
    python2 capture.py -r myTeam -b ${opponents[$op]} -q -l ${layouts[$layout]}
done
