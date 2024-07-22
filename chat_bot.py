import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv

# Load data
file_path = r'Training.csv'
training = pd.read_csv(file_path)

# Print column names to verify the presence of 'prognosis'
print("Column names:", training.columns)

# Ensure 'prognosis' column exists
if 'prognosis' not in training.columns:
    raise KeyError("The column 'prognosis' does not exist in the dataset.")

cols = training.columns
cols = cols[:-1]  # Assuming 'prognosis' is the last column
x = training[cols]
y = training['prognosis']

# Preprocess data
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Load additional data
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}
for index, symptom in enumerate(x.columns):
    symptoms_dict[symptom] = index

# Function to calculate condition
def calc_condition(exp, days):
    total_severity = 0
    for item in exp:
        if item in severityDictionary:
            total_severity += severityDictionary[item]
        else:
            st.warning(f"Severity information not available for {item}")

    if (total_severity * days) / (len(exp) + 1) > 13:
        st.warning("You should take consultation from a doctor.")
    else:
        st.info("It might not be that bad, but you should take precautions.")

# Function to get description
def getDescription():
    global description_list
    try:
        with open('MasterData/symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue
                if len(row) >= 2:  # Ensure there are at least two elements
                    description_list[row[0]] = row[1]
                else:
                    st.warning(f"Skipping row with insufficient data: {row}")
    except FileNotFoundError:
        st.error("Description data file not found.")
    except Exception as e:
        st.error(f"An error occurred while reading description data: {e}")

# Function to get severity dictionary
def getSeverityDict():
    global severityDictionary
    try:
        with open('MasterData/Symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue
                if len(row) >= 2:  # Ensure there are at least two elements
                    try:
                        severityDictionary[row[0]] = int(row[1])
                    except ValueError:
                        st.warning(f"Skipping row with invalid severity value: {row}")
                else:
                    st.warning(f"Skipping row with insufficient data: {row}")
    except FileNotFoundError:
        st.error("Severity data file not found.")
    except Exception as e:
        st.error(f"An error occurred while reading severity data: {e}")

# Function to get precaution dictionary
def getprecautionDict():
    global precautionDictionary
    try:
        with open('MasterData/symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue
                if len(row) >= 5:  # Ensure there are at least five elements
                    precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
                else:
                    st.warning(f"Skipping row with insufficient data: {row}")
    except FileNotFoundError:
        st.error("Precaution data file not found.")
    except Exception as e:
        st.error(f"An error occurred while reading precaution data: {e}")

# Function to get user input for symptoms
def get_user_input():
    st.sidebar.header("HealthCare ChatBot")
    name = st.sidebar.text_input("Your Name?")
    st.sidebar.text(f"Hello, {name}")

    st.sidebar.text("Select a symptom from the menu:")
    selected_symptoms = st.sidebar.multiselect("Symptoms", x.columns)

    num_days = st.sidebar.number_input("From how many days?", min_value=0, step=1)
    return name, selected_symptoms, num_days

# Main function to run the app
def main():
    st.title("HealthCare ChatBot")

    name, selected_symptoms, num_days = get_user_input()

    if st.sidebar.button("Diagnose"):
        st.header("Diagnosis Result")

        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            related_symptoms = selected_symptoms.copy()
            
            while st.sidebar.checkbox("Do you have any related symptoms?"):
                new_symptom = st.sidebar.selectbox("Select a related symptom:", x.columns)
                related_symptoms.append(new_symptom)

            calc_condition(related_symptoms, num_days)

            st.subheader("Possible Conditions:")

            symptoms_input_df = pd.DataFrame(0, index=[0], columns=x.columns)
            for symptom in related_symptoms:
                if symptom in x.columns:
                    symptoms_input_df.loc[0, symptom] = 1
                else:
                    st.warning(f"Symptom '{symptom}' not recognized.")

            try:
                second_prediction = clf.predict(symptoms_input_df)
                present_disease = le.inverse_transform(second_prediction)
                st.write(f"You may have {present_disease[0]}")
                st.write(description_list.get(present_disease[0], "No description available."))

                st.subheader("Precautions:")
                precaution_list = precautionDictionary.get(present_disease[0], [])
                if precaution_list:
                    for i, j in enumerate(precaution_list):
                        st.write(f"{i+1}. {j}")
                else:
                    st.write("No specific precautions available.")
            except Exception as e:
                st.error(f"An error occurred during diagnosis: {e}")

if __name__ == '__main__':
    getSeverityDict()
    getDescription()
    getprecautionDict()
    main()
