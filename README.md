# Facial Emotion Recognition

### How to add a new model?
For adding a new model to this repository all you have to do is create the new model, make your data compatible for new model(if required) and then mention the new model, that's it.

for example you want to create a new model named `MyModel`, then you need to change the following files :- <br>
1. **models.py** - Within model.py create a new class named `MyModel` extending `BaseModel` class. `BaseModel` is the class which needs to be extended by every model. And then simply implement two methods `model_builder` and `train`.
 All code for building you model goes inside `model_builder` function. Whereas `train` function is for training that model.
 
 ```
 class MyModel(BaseModel):
 
    def model_builder(self, **kwargs):
        # build model here
        self.model = my_builded_model
        
    def train(self, **kwargs):
        # train model here

 ```
 <sup>Refer to [model.py]() for seeing this process in action.</sup>

2. **data_builder.py** - Within this file you will make data compatible for your model. For example in your `MyModel` you are not only feeding the raw face images but also feeding some ROI(region of interest) then you need to crop these ROI for all images and convert them to numpy arrays before feeding into the network.
For this within data_builder.py you need to create a new class named `ImageToROI` extending `DataBuilder` class. `DataBuilder` is the class which needs to be extended and you should implement it's `build_from_directory` method, which returns the data in required format. You can also reuse already made classes if they support your model otherwise you need to create your own.

```
class ImageToROI(DataBuilder):

    def build_from_directory(self):
        # write code here
        return array
    
```
 <sup>Refer to [data_builder.py](https://github.com/greatsharma/Facial_Emotion_Recognition/blob/master/data_builder.py) for seeing this process in action.</sup>
 
3. **trainer.py** - This file is only for training purpose, in this file you should mention your new model.<br>
 <sup>Refer to [trainer.py](https://github.com/greatsharma/Facial_Emotion_Recognition/blob/master/trainer.py) for seeing this process in action.</sup>

These 3 are the major files you need to change, apart from this there are some already available `lr_scheduler`, `early_stopping` or `train_datagen` to use, or create your own.


### How to train a model?
For training a model all you need to do is run `trainer.py` with appropriate arguments. There are a variety of arguments which you can mention :-

**d** - Dataset to train on, `fer`, `feraligned`, `ck` and `feraligned+ck` are supported.<br>
**m** - Model to train on, currently `cnn` and `cnn+roi1+roi2` are supported.<br>
**em** - Emotions to train on, comma separated values, depending on the dataset select any subset from {Happy,Sadness,Surprise,Angry,Fear,Neutral}.<br>
**s** - 1 to Shuffle before split otherwise 0, default is 1.<br>
**rs** - Random state to use, default is 42.<br>
**tr** - Train ratio a value from 0 to 1, default is 0.85.<br>
**lrs** - Lr scheduler to use, default is None.<br>
**es** - Early stopping to use, default is None.<br>
**tg** - Train data generator to use, default is None.<br>
**bs** - Batch size to use, default is 24.<br>
**ep** - Max epochs, default is 50.<br>
**o** - Optimizer to use, `adam` and `nadam` are supported, default is adam.<br>
**lr** - Learning rate to use, default is 0.01.<br>
**sa** - 1 to save_architecture otherwise 0, default is 0.<br>
**sm** - 1 to save the model otherwise 0, default is 0.<br>
**scm** - 1 to save the confusion matrix of test set otherwise 0, default is 0.<br>
**sth** - 1 to save training history otherwise 0, default is 0.<br>

example :-<br><br>
`python trainer.py -d fer -m MyModel -em Happy,Sadness,Neutral -ep 25 -tr 0.7 -bs 16 -sa 1 -sm 1 -scm 1 -sth 1 -tg 4`<br><br>
This will train `MyModel` on the `fer` dataset for 3 classes `Happy,Sadness,Neutral`, number of `epochs` are 25, `train-ratio` is 0.7, `batch-size` is 16 and we are also saving everything.

All the outputs generated from training will go in their respective folders within the [outputs](https://github.com/greatsharma/Facial_Emotion_Recognition/tree/master/outputs) directory. The saved models follows a naming convetion :-<br>
`<model name>_<dataset trained on>_<number of emotions trained on>emo`<br><br>
For example in the above model, name would be `MyModel_fer_3emo`.



### How to test a model?
For testing a model run following command
<br><br>
`python -m model_testing.test_CNNModel -i webcam -d dnn -m CNNModel_feraligned+ck_5emo`

**i** - Input type, either `webcam` or `path to video file`.<br>
**d** - Detector to use, either `dlib` or `dnn`.<br>
**m** - Model to use, anyone present in the `output/models` directory.<br>
**he** - 1 to apply `histogram equalization` otherwise 0, default is 0.<br>
**ws** - Window size, `comma separated` values, default is None.<br>

As each tensorflow model is more than around +50mbs, so adding all the models in this repository will increase it's size drastically hence I added few good performing models. So if you want to test a new model then simply go to [this]() repository, select any model of your choice from the `models` folder and then select label2text file corresponding to that model from the `label2text` folder. Then place these files to models and label2text folder of this repository to test that particular model.
