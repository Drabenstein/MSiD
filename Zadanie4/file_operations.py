import pickle as pkl

def save_data(data, file_name):
    output = open(file_name, 'wb')
    pkl.dump(data, output)

def read_train_file():
     with open('train.pkl', 'rb') as f:
        return pkl.load(f)

def read_model(modelFileName):
   with open(modelFileName, 'rb') as f:
      return pkl.load(f)