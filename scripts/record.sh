echo "Recording for training"
arecord -c 1 -r 48000 -d 5 > train/$1/speech.wav
sleep 2 
echo "Recording for testing" 
arecord -c 1 -r 48000 -d 5 > test/$1/speech.wav
