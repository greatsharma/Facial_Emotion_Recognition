# for video run `bash run.sh`
# for webcam/image run `sampletype=webcam/image bash run.sh`

export MODEL_PATH="models/"
export TEST_SAMPLE_PATH="test_samples/"
export MODEL="model_2.h5"

if [ -z "$sampletype" ]
    then
        export SAMPLE_TYPE=video
else
    export SAMPLE_TYPE=$sampletype
fi

echo "FER on "$SAMPLE_TYPE"..."
python test.py