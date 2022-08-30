from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from os.path import join
from os import listdir
import pandas as pd

from genarator import testgen, datagen
from metrics import f1macro
from model import cnnpred_2d_mine


DATADIR = "../Dataset"
TRAIN_TEST_CUTOFF = '2016-04-21'
TRAIN_VALID_RATIO = 0.75

seq_len = 60
batch_size = 128
n_epochs = 20
n_features = 82

data = {}

print("data we have: ")
for filename in listdir(DATADIR):
    if not filename.lower().endswith(".csv"):
        continue # read only the CSV files

    filepath = join(DATADIR, filename)
    X = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    # basic preprocessing: get the name, the classification
    # Save the target variable as a column in dataframe for easier dropna()
    name = X["Name"][0]
    print(X["Name"][0], end=' | ')
    del X["Name"]
    cols = X.columns
    # The line of code above is to compute the percentage change of the closing 
    # index and align the data with the previous day. Then convert the data into 
    # either 1 or 0 for whether the percentage change is positive.
    X["Target"] = (X["Close"].pct_change().shift(-1) > 0).astype(int) 
    X.dropna(inplace=True)
    # Fit the standard scaler using the training dataset
    index = X.index[X.index > TRAIN_TEST_CUTOFF]
    index = index[:int(len(index) * TRAIN_VALID_RATIO)]
    scaler = StandardScaler().fit(X.loc[index, cols])
    # Save scale transformed dataframe
    X[cols] = scaler.transform(X[cols])
    data[name] = X

# Produce CNNpred as a binary classification problem
model = cnnpred_2d_mine(seq_len, n_features)

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='mae', metrics=["acc", f1macro])
model.summary()  # print model structure to console


# Set up callbacks and fit the model
# We use custom validation score f1macro() and hence monitor for "val_f1macro"
checkpoint_path = "./cp2d-{epoch}-{val_f1macro:.2f}.h5"
callbacks = [
    ModelCheckpoint(checkpoint_path,
                    monitor='val_f1macro', mode="max",
                    verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
]
model.fit(datagen(data, seq_len, batch_size, "Target", "train", TRAIN_TEST_CUTOFF, TRAIN_VALID_RATIO),
          validation_data=datagen(data, seq_len, batch_size, "Target", "valid"),
          epochs=n_epochs, steps_per_epoch=400, validation_steps=10, verbose=1, callbacks=callbacks)

# Prepare test data
test_data, test_target = testgen(data, seq_len, "Target", TRAIN_TEST_CUTOFF)
 
# Test the model
test_out = model.predict(test_data)
test_pred = (test_out > 0.5).astype(int)
print("accuracy:", accuracy_score(test_pred, test_target))
print("MAE:", mean_absolute_error(test_pred, test_target))
print("F1:", f1_score(test_pred, test_target))