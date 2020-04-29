# petfinder, by team petfinder
## Project Overview

Computer Vision (CS 1430) Final Project by Victoria Lin and Elliot Kang. For this project, we are designing a CNN model to take pictures of pets and predict their adoption times. This was a [competition on Kaggle ](https://www.kaggle.com/c/petfinder-adoption-prediction/overview "Kaggle Competition Page"). We use the dataset provided by the competition to train and evaluate our model.

  ### Data
The dataset provided by Kaggle is a 2 GB set with lots of data on each pet (breed information, health conditions, adoption fee, location, age, name, animal type, etc). We however are only using the images of the pets, because we are curious to see how we can extract the aforementioned information to create an accurate estimation of the pet's ``adoptibility". To download this data, we used the Kaggle API. See below for details on the process.

  ### Software
To build our model, we use TensorFlow's keras API. We also use Keras in our preprocessing, to augment our data with ImageDataGenerator. To train and evaluate our model, we use Google Cloud Platform.



## Running the program

### Downloading the datasets
1. Generate a new Kaggle API token: On Kaggle, log in then click top right user icon > My Account > scroll to API, generate new token, should download JSON file
2. On your GCP instance, install kaggle, set up directories, copy over your JSON information

```pip install --user kaggle
mkdir ~/.kaggle
vim ~/.kaggle/kaggle.json 
```
In vim, copy paste contents of JSON, save and change permissions with `chmod 600 /home/<user>/.kaggle/kaggle.json`

3. Now that the kaggle API is downloaded and the token is set up, download the data and unzip it in the right place

```.local/bin/kaggle competitions download -c petfinder-adoption-prediction
mkdir petfinder/data
mkdir petfinder/data/petfinder-adoption-prediction
unzip petfinder-adoption-prediction -d petfinder/data/petfinder-adoption-prediction
```

4. Now you can activate the venv with `source /cs1430-env/bin/activate` and run `main.py` with the create_sets uncommented to build it at first. create_sets will build the directory structure and move the images as needed according to Keras ImageDataGenerator's flow_from_directory method.

### Training and evaluating

lalala