import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Flower Classification", layout="centered")

MODEL_PATH = "trained_model.pkl"
SPECIES_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df["species"] = [SPECIES_NAMES[i] for i in iris.target]
    return iris, df


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


def to_species(value):
    try:
        return SPECIES_NAMES[int(value)]
    except (ValueError, TypeError):
        text = str(value).lower()
        for name in SPECIES_NAMES:
            if name.split()[-1].lower() in text:
                return name
        return str(value)


iris, df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Overview", "Dataset Summary", "Make a Prediction"]
)


if page == "Overview":
    st.title("Iris Flower Classification")
    st.write(
        "This project uses a trained machine learning model to predict the "
        "species of an Iris flower from four measurements: sepal length, "
        "sepal width, petal length and petal width."
    )

    st.subheader("Iris Species")
    st.write(
        "- **Iris Setosa**: small petals, easy to separate from the others.\n"
        "- **Iris Versicolor**: medium-sized petals.\n"
        "- **Iris Virginica**: the largest petals of the three."
    )

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", 150)
    col2.metric("Input features", 4)
    col3.metric("Species", 3)

    st.subheader("How the Application Works")
    st.write(
        "The Dataset Summary page shows the Iris data and a few plots. "
        "On the Make a Prediction page you enter the four measurements, "
        "and the saved model returns the predicted species."
    )


elif page == "Dataset Summary":
    st.title("Dataset Summary")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Feature names: " + ", ".join(iris.feature_names))
    st.write("Target classes: " + ", ".join(SPECIES_NAMES))

    st.write("Missing values per column:")
    st.dataframe(df.isnull().sum().rename("Missing values"))

    st.write("Descriptive statistics:")
    st.dataframe(df.describe())

    st.subheader("Exploratory Data Analysis")

    st.write("Number of samples per species")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="species", order=SPECIES_NAMES, ax=ax)
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    st.write("Petal length vs petal width")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        hue_order=SPECIES_NAMES,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        "Setosa forms a separate group with small petals. Versicolor and "
        "Virginica are close together, with only a small overlap."
    )

    st.write("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        df[iris.feature_names].corr(),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        "Petal length and petal width are very strongly correlated. "
        "Sepal width has the weakest relationship with the other features."
    )

    st.write("Petal length by species")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="species", y="petal length (cm)", order=SPECIES_NAMES, ax=ax)
    ax.set_xlabel("Species")
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        "Petal length increases from Setosa to Versicolor to Virginica, "
        "which is why it is useful for telling the species apart."
    )


else:
    st.title("Make a Prediction")
    st.write("Enter the flower measurements and click the button to get a prediction.")

    col1, col2 = st.columns(2)
    sepal_length = col1.number_input(
        "Sepal Length (cm)", min_value=0.1, max_value=10.0, value=5.8, step=0.1
    )
    sepal_width = col2.number_input(
        "Sepal Width (cm)", min_value=0.1, max_value=10.0, value=3.0, step=0.1
    )
    petal_length = col1.number_input(
        "Petal Length (cm)", min_value=0.1, max_value=10.0, value=4.3, step=0.1
    )
    petal_width = col2.number_input(
        "Petal Width (cm)", min_value=0.1, max_value=10.0, value=1.3, step=0.1
    )

    if st.button("Predict Species"):
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(
                "The file trained_model.pkl was not found. Place it in the "
                "same folder as app.py and run the app again."
            )
            st.stop()
        except Exception as e:
            st.error(f"The model file could not be loaded: {e}")
            st.stop()

        values = [[sepal_length, sepal_width, petal_length, petal_width]]

        names_in_model = getattr(model, "feature_names_in_", None)
        if names_in_model is not None and set(names_in_model) == set(iris.feature_names):
            X = pd.DataFrame(values, columns=iris.feature_names)
        else:
            X = np.array(values)

        try:
            prediction = model.predict(X)[0]
        except Exception as e:
            st.error(f"The model could not make a prediction with these inputs: {e}")
            st.stop()

        species = to_species(prediction)
        st.subheader("Result")
        st.success(f"Predicted species: {species}")
        st.write(
            f"Based on the measurements entered, the model classifies this "
            f"flower as {species}."
        )

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                labels = [to_species(c) for c in model.classes_]
                proba_df = pd.DataFrame({"Probability": proba}, index=labels)
                st.write("Prediction probabilities")
                st.bar_chart(proba_df)
            except Exception:
                pass
