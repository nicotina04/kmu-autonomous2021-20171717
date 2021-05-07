import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    model_name = "gunnerDetector.pkl"
    df = pd.read_csv("pose_label.csv")
    x = df.drop("class", axis=1)
    y = df["class"]

    # Separate train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=324)

    # Create pipeline
    ela_pipeline = make_pipeline(RobustScaler(), KNeighborsClassifier())
    poseModel = ela_pipeline.fit(x_train, y_train)

    # Evaluate my pose model
    y_predict = ela_pipeline.predict(x_test)
    print("Accuracy for my model")
    print(accuracy_score(y_test, y_predict))

    f = open(model_name, "wb")
    pickle.dump(poseModel, f)
    f.close()
